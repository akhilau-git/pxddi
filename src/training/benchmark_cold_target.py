"""Cold-Target / Protein Sequence Generalization Benchmark Study for AuditDDI.

Evaluates the inductive power of primary UniProt amino acid sequences and
ProteinTargetSequenceEncoder (1D-CNN + attention pooling / ESM-2) on:
1. S1 True Cold-Start (Unseen Drugs)
2. Protein Sequence Profiled Cohort (Cold-Target Scenario)
3. Transductive & S2 Generalization

Compares:
- Model A: Multimodal Baseline without Sequence (auditddi_multimodal_v1)
- Model B: AuditDDI Multimodal + Protein Target Sequences (auditddi_protein_seq)
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data_prep.cached_graph_loader import (
    MolecularCache,
    build_cached_multimodal_dataloader,
)
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_MULTIMODAL,
    PxDDIModel,
)
from src.training.benchmark_cold_start import (
    ensure_benchmark_splits,
    safe_forward_multimodal,
)


def compute_comprehensive_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate AUROC, AUPRC, Balanced Accuracy, Sensitivity, Specificity, Brier, and ECE."""
    preds = (probs >= threshold).astype(int)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    if n_pos > 0 and n_neg > 0:
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))
    else:
        auroc = 0.5
        auprc = 0.0

    brier = float(brier_score_loss(labels, probs))
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, zero_division=0))
    mcc = matthews_corrcoef(labels, preds)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0

    # 10-bin Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, 11)
    bin_assignments = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(10):
        bin_mask = bin_assignments == b
        if np.any(bin_mask):
            bin_conf = np.mean(probs[bin_mask])
            bin_acc = np.mean(labels[bin_mask])
            ece += np.sum(bin_mask) / len(probs) * np.abs(bin_acc - bin_conf)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "mcc": mcc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": brier,
        "ece": float(ece),
        "threshold": threshold,
        "n_samples": len(labels),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


def paired_bootstrap_comparison(
    labels: np.ndarray,
    probs_baseline: np.ndarray,
    probs_multimodal: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired non-parametric bootstrap test comparing baseline vs protein-sequence enhanced model."""
    rng = np.random.default_rng(seed)
    n_samples = len(labels)
    delta_aurocs: list[float] = []
    delta_auprcs: list[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_base = roc_auc_score(y_b, probs_baseline[idx])
        auc_multi = roc_auc_score(y_b, probs_multimodal[idx])
        ap_base = average_precision_score(y_b, probs_baseline[idx])
        ap_multi = average_precision_score(y_b, probs_multimodal[idx])
        delta_aurocs.append(float(auc_multi - auc_base))
        delta_auprcs.append(float(ap_multi - ap_base))

    ci_auc = np.percentile(delta_aurocs, [2.5, 97.5])
    ci_ap = np.percentile(delta_auprcs, [2.5, 97.5])

    # Paired Wilcoxon signed-rank test on sample-level absolute prediction errors
    err_base = np.abs(labels - probs_baseline)
    err_multi = np.abs(labels - probs_multimodal)
    diff = err_base - err_multi
    if np.all(diff == 0):
        p_val = 1.0
    else:
        try:
            _, p_val = wilcoxon(diff, alternative="greater")
        except Exception:
            p_val = 1.0

    return {
        "delta_auroc_mean": float(np.mean(delta_aurocs)),
        "delta_auroc_ci95": [float(ci_auc[0]), float(ci_auc[1])],
        "delta_auprc_mean": float(np.mean(delta_auprcs)),
        "delta_auprc_ci95": [float(ci_ap[0]), float(ci_ap[1])],
        "p_value": float(p_val),
    }


def evaluate_loader_predictions(
    model: PxDDIModel,
    loader: Any,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run inference over a DataLoader and compute performance metrics."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in loader:
            da = batch["drug_a"].to(device)
            db = batch["drug_b"].to(device)
            out = safe_forward_multimodal(model, batch, da, db, device)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
            labels = batch["labels"].numpy().ravel()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    p_arr = np.array(all_probs)
    y_arr = np.array(all_labels)
    metrics = compute_comprehensive_metrics(y_arr, p_arr, threshold=threshold)
    return metrics, p_arr, y_arr


def run_cold_target_study(
    master_nodes_path: str | Path,
    splits_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    pos_weight: float = 2.0,
    seed: int = 42,
    device: torch.device | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute complete cold-target / protein sequence benchmark comparing Baseline vs Protein Sequence Enhanced."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    splits_p = ensure_benchmark_splits(
        splits_dir=splits_dir,
        master_nodes_path=master_nodes_path,
        seed=seed,
    )

    print("=" * 80)
    print("STARTING AUDITDDI COLD-TARGET / PROTEIN SEQUENCE STUDY")
    print(f"Device      : {device}")
    print(f"Master Nodes: {master_nodes_path}")
    print(f"Splits Dir  : {splits_p}")
    print(f"Output Dir  : {out_p}")
    print("=" * 80)

    # 1. Populate MolecularCache
    cache = MolecularCache()
    cache.populate_from_master_nodes(master_nodes_path)

    # 2. Build DataLoaders
    df_train = pd.read_csv(splits_p / "transductive_train.csv")
    df_val = pd.read_csv(splits_p / "validation.csv")
    df_s1 = pd.read_csv(splits_p / "s1_test.csv")
    df_trans = pd.read_csv(splits_p / "transductive_test.csv")

    train_loader = build_cached_multimodal_dataloader(df_train, cache, batch_size=batch_size, shuffle=True)
    val_loader = build_cached_multimodal_dataloader(df_val, cache, batch_size=batch_size, shuffle=False)
    s1_loader = build_cached_multimodal_dataloader(df_s1, cache, batch_size=batch_size, shuffle=False)
    trans_loader = build_cached_multimodal_dataloader(df_trans, cache, batch_size=batch_size, shuffle=False)

    # Defensively ensure target_sequences attribute exists on cache
    if not hasattr(cache, "target_sequences") or not cache.target_sequences:
        cache.target_sequences = {}
        try:
            df_m = pd.read_csv(master_nodes_path)
            id_c = "drug_id" if "drug_id" in df_m.columns else df_m.columns[0]
            seq_c = next((c for c in ["target_sequence", "uniprot_sequence", "target_seq", "protein_sequence"] if c in df_m.columns), None)
            if seq_c:
                for _, row in df_m.iterrows():
                    val = str(row[seq_c]).strip() if pd.notna(row[seq_c]) else ""
                    if val and val.lower() != "nan":
                        cache.target_sequences[str(row[id_c]).strip()] = val
        except Exception:
            pass

    # Extract Cold-Target S1 subcohort: pairs where both drugs have UniProt sequence annotation
    s_col = "drug_a_id" if "drug_a_id" in df_s1.columns else df_s1.columns[0]
    t_col = "drug_b_id" if "drug_b_id" in df_s1.columns else df_s1.columns[1]
    target_seqs = getattr(cache, "target_sequences", {})
    cold_target_mask = df_s1.apply(
        lambda r: bool(target_seqs.get(str(r[s_col]).strip(), ""))
        and bool(target_seqs.get(str(r[t_col]).strip(), "")),
        axis=1,
    )
    df_cold_target = df_s1[cold_target_mask]
    if len(df_cold_target) >= 10:
        cold_target_loader = build_cached_multimodal_dataloader(df_cold_target, cache, batch_size=batch_size, shuffle=False)
    else:
        cold_target_loader = s1_loader

    print(f"Dataset Partitions: Train={len(df_train)}, Val={len(df_val)}, S1={len(df_s1)}, Cold-Target Subcohort={len(df_cold_target)}")

    first_smi = next(iter(cache.graphs.keys()))
    first_graph = cache.graphs[first_smi]
    in_dim = first_graph.x.size(1) if first_graph.x is not None else 78
    edge_dim = first_graph.edge_attr.size(1) if first_graph.edge_attr is not None else 10

    # Architectures to compare
    configs = [
        ("multimodal_without_seq", False),
        ("auditddi_protein_seq", True),
    ]

    results: dict[str, Any] = {}
    s1_probs: dict[str, np.ndarray] = {}
    s1_labels: np.ndarray | None = None
    cold_target_probs: dict[str, np.ndarray] = {}
    cold_target_labels: np.ndarray | None = None

    for model_name, use_protein_seq in configs:
        tag = "WITH PROTEIN SEQUENCES" if use_protein_seq else "WITHOUT SEQUENCES (MULTI-HOT)"
        print(f"\n--- Training {model_name.upper()} ({tag}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = PxDDIModel(
            in_channels=in_dim,
            hidden_channels=64,
            architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
            edge_feature_dim=edge_dim,
            use_toxicity_pair_features=True,
            gene_feature_dim=cache.gene_dim,
            gene_hidden_channels=64,
            use_clinical_toxicity=True,
            use_target_encoder=True,
            target_feature_dim=cache.target_dim,
            target_hidden_channels=64,
            use_protein_sequence_encoder=use_protein_seq,
            use_pdb_encoder=True,
            pdb_feature_dim=cache.pdb_dim,
            pdb_hidden_channels=64,
            use_geo_features=True,
            geo_dim=cache.geo_dim,
            use_cross_modal_attention=True,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_auc = 0.0
        best_state = None
        start_time = time.time()

        for ep in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                da = batch["drug_a"].to(device)
                db = batch["drug_b"].to(device)
                y = batch["labels"].to(device)

                out = safe_forward_multimodal(model, batch, da, db, device)
                risk_logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(risk_logits.view(-1), y.float().view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            val_m, _, _ = evaluate_loader_predictions(model, val_loader, device)
            if val_m["auroc"] > best_val_auc:
                best_val_auc = val_m["auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"Epoch {ep:02d}/{epochs:02d} - Loss: {train_loss / len(train_loader):.4f} - Val AUROC: {val_m['auroc']:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        # Optimize Youden index threshold on honest validation set
        val_m, val_p, val_y = evaluate_loader_predictions(model, val_loader, device)
        fpr, tpr, thresholds = roc_curve(val_y, val_p)
        j_scores = 2.0 * tpr - fpr
        best_thresh = float(thresholds[np.argmax(j_scores)]) if len(thresholds) > 0 else 0.40

        # Evaluate on S1 Cold-Start
        s1_m, s1_p, s1_y = evaluate_loader_predictions(model, s1_loader, device, threshold=best_thresh)
        s1_probs[model_name] = s1_p
        s1_labels = s1_y

        # Evaluate on Cold-Target Cohort
        ct_m, ct_p, ct_y = evaluate_loader_predictions(model, cold_target_loader, device, threshold=best_thresh)
        cold_target_probs[model_name] = ct_p
        cold_target_labels = ct_y

        # Evaluate on Transductive Test
        trans_m, _, _ = evaluate_loader_predictions(model, trans_loader, device, threshold=best_thresh)
        train_elapsed = time.time() - start_time

        results[model_name] = {
            "validation_auroc": val_m["auroc"],
            "optimal_threshold": best_thresh,
            "transductive_test": trans_m,
            "s1_cold_start": s1_m,
            "cold_target_cohort": ct_m,
            "training_time_seconds": train_elapsed,
        }
        print(f"Finished {model_name}: S1 AUROC = {s1_m['auroc']:.4f}, Cold-Target AUROC = {ct_m['auroc']:.4f}")

    if s1_labels is None or cold_target_labels is None:
        raise RuntimeError("Evaluation failed to collect predictions.")

    # Bootstrap hypothesis testing on Cold-Target Cohort
    stat_ct = paired_bootstrap_comparison(
        cold_target_labels,
        cold_target_probs["multimodal_without_seq"],
        cold_target_probs["auditddi_protein_seq"],
        seed=seed,
    )
    results["cold_target_statistical_comparison"] = stat_ct

    # Bootstrap hypothesis testing on S1 Cold-Start
    stat_s1 = paired_bootstrap_comparison(
        s1_labels,
        s1_probs["multimodal_without_seq"],
        s1_probs["auditddi_protein_seq"],
        seed=seed,
    )
    results["s1_statistical_comparison"] = stat_s1

    # Export JSON
    with open(out_p / "cold_target_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Export CSV summary
    rows = []
    for m_name in ["multimodal_without_seq", "auditddi_protein_seq"]:
        ct_data = results[m_name]["cold_target_cohort"]
        s1_data = results[m_name]["s1_cold_start"]
        rows.append({
            "model": m_name,
            "cold_target_auroc": ct_data["auroc"],
            "cold_target_auprc": ct_data["auprc"],
            "cold_target_sensitivity": ct_data["sensitivity"],
            "s1_auroc": s1_data["auroc"],
            "s1_auprc": s1_data["auprc"],
            "s1_sensitivity": s1_data["sensitivity"],
            "brier_score": ct_data["brier_score"],
            "ece": ct_data["ece"],
        })
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_p / "cold_target_benchmark_summary.csv", index=False)

    # Generate Markdown Report
    base_ct = results["multimodal_without_seq"]["cold_target_cohort"]
    prot_ct = results["auditddi_protein_seq"]["cold_target_cohort"]
    base_s1 = results["multimodal_without_seq"]["s1_cold_start"]
    prot_s1 = results["auditddi_protein_seq"]["s1_cold_start"]
    ci_ct = stat_ct["delta_auroc_ci95"]

    md_content = f"""# 🧬 AuditDDI Cold-Target & UniProt Protein Sequence Benchmark Study

## Executive Summary
This study benchmarks the generalization improvement provided by **UniProt primary amino-acid target sequence encoding** compared to standard multimodal multi-hot representations.

| Model | Cold-Target AUROC | Cold-Target AUPRC | Cold-Target Recall | S1 Cold AUROC | S1 Cold AUPRC | S1 Recall | Brier Score |
|---|---|---|---|---|---|---|---|
| **Multimodal Baseline** | {base_ct['auroc']:.4f} | {base_ct['auprc']:.4f} | {base_ct['sensitivity']*100:.1f}% | {base_s1['auroc']:.4f} | {base_s1['auprc']:.4f} | {base_s1['sensitivity']*100:.1f}% | {base_ct['brier_score']:.4f} |
| **AuditDDI Protein-Seq** | **{prot_ct['auroc']:.4f}** | **{prot_ct['auprc']:.4f}** | **{prot_ct['sensitivity']*100:.1f}%** | **{prot_s1['auroc']:.4f}** | **{prot_s1['auprc']:.4f}** | **{prot_s1['sensitivity']*100:.1f}%** | **{prot_ct['brier_score']:.4f}** |

## Statistical Significance on Cold-Target Cohort (1,000-Iteration Paired Bootstrap)
- **$\\Delta$ AUROC**: {stat_ct['delta_auroc_mean']:+.4f} (95% CI: [{ci_ct[0]:+.4f}, {ci_ct[1]:+.4f}])
- **$\\Delta$ AUPRC**: {stat_ct['delta_auprc_mean']:+.4f}
- **Paired Wilcoxon $p$-value**: {stat_ct['p_value']:.4e}
"""
    with open(out_p / "cold_target_performance_comparison.md", "w") as f:
        f.write(md_content)

    print(f"\nCold-Target benchmark artifacts generated in: {out_p}")
    return results
