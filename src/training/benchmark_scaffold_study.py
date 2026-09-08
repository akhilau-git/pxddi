"""Murcko Scaffold-Disjoint Benchmark Suite for AuditDDI.

Evaluates true inductive chemical generalization across structurally disjoint
Bemis-Murcko molecular scaffolds.

Models Evaluated:
1. Baseline Molecular Graph GNN (edge_aware_gat_v2).
2. AuditDDI Multimodal Network (auditddi_multimodal_v1) integrating:
   - Edge-aware molecular graphs + 1024-bit Morgan ECFP fingerprints
   - PharmGKB pharmacogenomics, FAERS clinical toxicity, BindingDB target affinities
   - PDB macromolecular structures, GEO transcriptomics, and UniProt sequences

Metrics & Validation:
- AUROC, AUPRC, F1, Balanced Accuracy, Recall (Sensitivity), Specificity
- Brier Score & Expected Calibration Error (ECE)
- 1,000-iteration Paired Bootstrap 95% Confidence Intervals
- Markdown report & CSV summary export
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data_prep.cached_graph_loader import (
    MolecularCache,
    build_cached_multimodal_dataloader,
)
from src.data_prep.scaffold_splits import (
    create_scaffold_disjoint_splits,
)
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_MULTIMODAL,
    PxDDIModel,
)


REQUIRED_SCAFFOLD_FILES = [
    "scaffold_train.csv",
    "scaffold_validation.csv",
    "scaffold_test.csv",
]


def ensure_scaffold_splits(
    splits_dir: str | Path | None = None,
    master_nodes_path: str | Path | None = None,
    master_edges_path: str | Path | None = None,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.20,
    seed: int = 42,
    **kwargs: Any,
) -> Path:
    """Ensure scaffold-disjoint splits exist; generate them automatically if missing."""
    if splits_dir is not None and str(splits_dir).strip().lower() not in ("none", ""):
        splits_path = Path(splits_dir)
    elif master_nodes_path is not None:
        nodes_p = Path(master_nodes_path).resolve()
        splits_path = nodes_p.parent.parent / "scaffold_splits"
    else:
        splits_path = Path("scaffold_splits")

    splits_path.mkdir(parents=True, exist_ok=True)

    all_exist = all(
        (splits_path / f).is_file() and (splits_path / f).stat().st_size > 0
        for f in REQUIRED_SCAFFOLD_FILES
    )
    if all_exist:
        print(f"Using existing scaffold-disjoint splits from: {splits_path}")
        return splits_path

    print(f"Generating scaffold-disjoint splits into: {splits_path}...")
    edge_candidates: list[Path] = []
    if master_edges_path is not None:
        edge_candidates.append(Path(master_edges_path))
    if master_nodes_path is not None:
        nodes_p = Path(master_nodes_path).resolve()
        edge_candidates.extend([
            nodes_p.parent / "master_ddi_edges.csv",
            nodes_p.parent.parent / "unified_graph" / "master_ddi_edges.csv",
            nodes_p.parent.parent / "splits" / "transductive_train.csv",
            nodes_p.parent.parent / "twosides" / "drug_drug_edges.csv",
            nodes_p.parent / "drug_drug_edges.csv",
        ])
    edge_candidates.extend([
        Path("unified_graph/master_ddi_edges.csv"),
        Path("splits/transductive_train.csv"),
        Path("twosides/drug_drug_edges.csv"),
        Path("master_ddi_edges.csv"),
    ])

    resolved_edges = next((p for p in edge_candidates if p.is_file()), None)
    if resolved_edges is None:
        raise FileNotFoundError(
            f"Cannot generate scaffold splits: interaction edges table not found. Looked in: {[str(p) for p in edge_candidates[:4]]}."
        )

    print(f"Loading interaction pairs from: {resolved_edges}")
    df_raw = pd.read_csv(resolved_edges, low_memory=False)

    src_col = "drug_a_id" if "drug_a_id" in df_raw.columns else ("source" if "source" in df_raw.columns else ("drug_a" if "drug_a" in df_raw.columns else df_raw.columns[0]))
    dst_col = "drug_b_id" if "drug_b_id" in df_raw.columns else ("target" if "target" in df_raw.columns else ("drug_b" if "drug_b" in df_raw.columns else df_raw.columns[1]))
    lbl_col = "label" if "label" in df_raw.columns else None

    # Load master nodes to map ID -> SMILES
    id_to_smiles: dict[str, str] = {}
    if master_nodes_path is not None and Path(master_nodes_path).is_file():
        df_nodes = pd.read_csv(master_nodes_path)
        node_id_col = "drug_id" if "drug_id" in df_nodes.columns else ("canonical_smiles" if "canonical_smiles" in df_nodes.columns else df_nodes.columns[0])
        smi_col = "canonical_smiles" if "canonical_smiles" in df_nodes.columns else node_id_col
        for _, r in df_nodes.iterrows():
            did = str(r[node_id_col]).strip()
            smi = str(r[smi_col]).strip()
            if did and smi:
                id_to_smiles[did] = smi

    df_pairs = df_raw[[src_col, dst_col]].copy().drop_duplicates()
    if lbl_col is not None:
        df_pairs["label"] = df_raw[lbl_col]
    else:
        df_pairs["label"] = 1.0

    def get_smiles(val: Any) -> str:
        s = str(val).strip()
        return id_to_smiles.get(s, s)

    df_pairs["_smiles_a"] = df_pairs[src_col].map(get_smiles)
    df_pairs["_smiles_b"] = df_pairs[dst_col].map(get_smiles)

    # Keep only valid SMILES
    valid_mask = (df_pairs["_smiles_a"].str.len() > 1) & (df_pairs["_smiles_b"].str.len() > 1)
    df_pairs = df_pairs[valid_mask].copy()

    split_tables = None
    audit_meta = None
    last_err = None
    for cand_seed in range(seed, seed + 30):
        try:
            split_tables, audit_meta = create_scaffold_disjoint_splits(
                df_pairs,
                drug_a_col="_smiles_a",
                drug_b_col="_smiles_b",
                label_col="label",
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                seed=cand_seed,
            )
            break
        except ValueError as err:
            last_err = err
            continue

    if split_tables is None:
        raise last_err or RuntimeError("Could not find a valid scaffold partition across candidate seeds.")

    for name in ["scaffold_train", "scaffold_validation", "scaffold_test"]:
        tbl = split_tables[name].copy()
        tbl = tbl.rename(columns={src_col: "drug_a_id", dst_col: "drug_b_id"})

        # Sample negatives strictly within this scaffold partition if only positives exist
        labels_present = set(tbl["label"].unique())
        if len(labels_present) < 2:
            partition_drugs = list(set(tbl["drug_a_id"].unique()) | set(tbl["drug_b_id"].unique()))
            known_pos = set(zip(tbl["drug_a_id"], tbl["drug_b_id"])) | set(zip(tbl["drug_b_id"], tbl["drug_a_id"]))

            rng = np.random.default_rng(seed)
            neg_a, neg_b = [], []
            n_pos = len(tbl)
            attempts = 0
            while len(neg_a) < n_pos and attempts < n_pos * 20 and len(partition_drugs) >= 2:
                attempts += 1
                da, db = rng.choice(partition_drugs, size=2, replace=False)
                if (da, db) not in known_pos and (db, da) not in known_pos:
                    neg_a.append(da)
                    neg_b.append(db)
                    known_pos.add((da, db))

            if neg_a:
                neg_df = pd.DataFrame({"drug_a_id": neg_a, "drug_b_id": neg_b, "label": 0.0})
                tbl = pd.concat([tbl[["drug_a_id", "drug_b_id", "label"]], neg_df], ignore_index=True)

        cols_to_save = [c for c in ["drug_a_id", "drug_b_id", "label"] if c in tbl.columns]
        tbl[cols_to_save].to_csv(splits_path / f"{name}.csv", index=False)

    with open(splits_path / "scaffold_split_audit.json", "w") as f:
        json.dump(audit_meta, f, indent=2, default=str)

    print(f"Scaffold splits successfully created in: {splits_path}")
    return splits_path


def compute_comprehensive_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Compute AUROC, AUPRC, F1, Balanced Acc, Sensitivity, Specificity, Brier, and ECE."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
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
    """Calculate 95% Confidence Intervals for AUROC / AUPRC difference."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    delta_aurocs = []
    delta_auprcs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot_y = labels[idx]
        if len(np.unique(boot_y)) < 2:
            continue
        auc_b = roc_auc_score(boot_y, probs_baseline[idx])
        auc_m = roc_auc_score(boot_y, probs_multimodal[idx])
        ap_b = average_precision_score(boot_y, probs_baseline[idx])
        ap_m = average_precision_score(boot_y, probs_multimodal[idx])
        delta_aurocs.append(auc_m - auc_b)
        delta_auprcs.append(ap_m - ap_b)

    delta_aurocs = np.array(delta_aurocs)
    delta_auprcs = np.array(delta_auprcs)

    # Paired Wilcoxon signed-rank test on absolute error
    errors_baseline = np.abs(labels - probs_baseline)
    errors_multimodal = np.abs(labels - probs_multimodal)
    try:
        w_stat, p_val = stats.wilcoxon(errors_baseline, errors_multimodal)
    except Exception:
        w_stat, p_val = 0.0, 1.0

    return {
        "delta_auroc_mean": float(np.mean(delta_aurocs)),
        "delta_auroc_ci95": [float(np.percentile(delta_aurocs, 2.5)), float(np.percentile(delta_aurocs, 97.5))],
        "delta_auprc_mean": float(np.mean(delta_auprcs)),
        "delta_auprc_ci95": [float(np.percentile(delta_auprcs, 2.5)), float(np.percentile(delta_auprcs, 97.5))],
        "wilcoxon_stat": float(w_stat),
        "p_value": float(p_val),
    }


def evaluate_split(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    threshold: float = 0.50,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Run model evaluation over a dataloader and compute metrics."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            da = batch["drug_a"].to(device)
            db = batch["drug_b"].to(device)
            fpa = batch["fp_a"].to(device)
            fpb = batch["fp_b"].to(device)
            ga = batch["gene_a"].to(device)
            gb = batch["gene_b"].to(device)
            gma = batch["gene_mask_a"].to(device)
            gmb = batch["gene_mask_b"].to(device)
            ta = batch["tox_a"].to(device)
            tb = batch["tox_b"].to(device)
            tma = batch["tox_mask_a"].to(device)
            tmb = batch["tox_mask_b"].to(device)
            tgta = batch["target_a"].to(device)
            tgtb = batch["target_b"].to(device)
            tgtma = batch["target_mask_a"].to(device)
            tgtmb = batch["target_mask_b"].to(device)
            geoa = batch["geo_a"].to(device)
            geob = batch["geo_b"].to(device)
            geoma = batch["geo_mask_a"].to(device)
            geomb = batch["geo_mask_b"].to(device)
            pdba = batch["pdb_a"].to(device)
            pdbb = batch["pdb_b"].to(device)
            pdbma = batch["pdb_mask_a"].to(device)
            pdbmb = batch["pdb_mask_b"].to(device)
            mem = batch.get("memory_features", None)
            if mem is not None:
                mem = mem.to(device)

            out = model(
                drug_a=da,
                drug_b=db,
                fp_a=fpa,
                fp_b=fpb,
                gene_a=ga,
                gene_b=gb,
                gene_mask_a=gma,
                gene_mask_b=gmb,
                clinical_tox_a=ta,
                clinical_tox_b=tb,
                clinical_tox_mask_a=tma,
                clinical_tox_mask_b=tmb,
                target_a=tgta,
                target_b=tgtb,
                target_mask_a=tgtma,
                target_mask_b=tgtmb,
                geo_a=geoa,
                geo_b=geob,
                geo_mask_a=geoma,
                geo_mask_b=geomb,
                pdb_a=pdba,
                pdb_b=pdbb,
                pdb_mask_a=pdbma,
                pdb_mask_b=pdbmb,
                memory_features=mem,
            )
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
            labels = batch["labels"].numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    metrics = compute_comprehensive_metrics(all_labels, all_probs, threshold=threshold)
    return metrics, all_probs, all_labels


def run_scaffold_disjoint_study(
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
    """Execute complete scaffold-disjoint benchmark study comparing Baseline vs Multimodal."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    splits_p = ensure_scaffold_splits(
        splits_dir=splits_dir,
        master_nodes_path=master_nodes_path,
        master_edges_path=kwargs.get("master_edges_path"),
        seed=seed,
    )

    print("=" * 80)
    print("STARTING AUDITDDI MURCKO SCAFFOLD-DISJOINT STUDY")
    print(f"Device      : {device}")
    print(f"Master Nodes: {master_nodes_path}")
    print(f"Splits Dir  : {splits_p}")
    print(f"Output Dir  : {out_p}")
    print("=" * 80)

    # 1. Populate MolecularCache
    cache = MolecularCache()
    cache.populate_from_master_nodes(master_nodes_path)

    # 2. Build DataLoaders
    df_train = pd.read_csv(splits_p / "scaffold_train.csv")
    df_val = pd.read_csv(splits_p / "scaffold_validation.csv")
    df_test = pd.read_csv(splits_p / "scaffold_test.csv")

    train_loader = build_cached_multimodal_dataloader(df_train, cache, batch_size=batch_size, shuffle=True)
    val_loader = build_cached_multimodal_dataloader(df_val, cache, batch_size=batch_size, shuffle=False)
    test_loader = build_cached_multimodal_dataloader(df_test, cache, batch_size=batch_size, shuffle=False)

    print(f"Dataset Partitions: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    results = {}
    test_probs = {}
    test_labels = None

    # Compare architectures
    configs = [
        ("baseline_gnn", MODEL_ARCHITECTURE_EDGE_AWARE, False),
        ("auditddi_multimodal", MODEL_ARCHITECTURE_MULTIMODAL, True),
    ]

    for model_name, arch_version, use_multi in configs:
        print(f"\n--- Training {model_name.upper()} ({arch_version}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Retrieve sample graph for channels
        first_smi = next(iter(cache.graphs.keys()))
        first_graph = cache.graphs[first_smi]
        in_dim = first_graph.x.size(1) if first_graph.x is not None else 78
        edge_dim = first_graph.edge_attr.size(1) if first_graph.edge_attr is not None else 10

        model = PxDDIModel(
            in_channels=in_dim,
            hidden_channels=64,
            architecture_version=arch_version,
            edge_feature_dim=edge_dim,
            use_toxicity_pair_features=True,
            gene_feature_dim=cache.gene_dim,
            gene_hidden_channels=64,
            use_clinical_toxicity=use_multi,
            use_target_encoder=use_multi,
            target_feature_dim=cache.target_dim,
            target_hidden_channels=64,
            use_pdb_encoder=use_multi,
            pdb_feature_dim=cache.pdb_dim,
            pdb_hidden_channels=64,
            use_geo_features=use_multi,
            geo_dim=cache.geo_dim,
            use_cross_modal_attention=use_multi,
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
                fpa = batch["fp_a"].to(device)
                fpb = batch["fp_b"].to(device)
                ga = batch["gene_a"].to(device)
                gb = batch["gene_b"].to(device)
                gma = batch["gene_mask_a"].to(device)
                gmb = batch["gene_mask_b"].to(device)
                ta = batch["tox_a"].to(device)
                tb = batch["tox_b"].to(device)
                tma = batch["tox_mask_a"].to(device)
                tmb = batch["tox_mask_b"].to(device)
                tgta = batch["target_a"].to(device)
                tgtb = batch["target_b"].to(device)
                tgtma = batch["target_mask_a"].to(device)
                tgtmb = batch["target_mask_b"].to(device)
                geoa = batch["geo_a"].to(device)
                geob = batch["geo_b"].to(device)
                geoma = batch["geo_mask_a"].to(device)
                geomb = batch["geo_mask_b"].to(device)
                pdba = batch["pdb_a"].to(device)
                pdbb = batch["pdb_b"].to(device)
                pdbma = batch["pdb_mask_a"].to(device)
                pdbmb = batch["pdb_mask_b"].to(device)
                y = batch["labels"].to(device)

                out = model(
                    drug_a=da,
                    drug_b=db,
                    fp_a=fpa,
                    fp_b=fpb,
                    gene_a=ga,
                    gene_b=gb,
                    gene_mask_a=gma,
                    gene_mask_b=gmb,
                    clinical_tox_a=ta,
                    clinical_tox_b=tb,
                    clinical_tox_mask_a=tma,
                    clinical_tox_mask_b=tmb,
                    target_a=tgta,
                    target_b=tgtb,
                    target_mask_a=tgtma,
                    target_mask_b=tgtmb,
                    geo_a=geoa,
                    geo_b=geob,
                    geo_mask_a=geoma,
                    geo_mask_b=geomb,
                    pdb_a=pdba,
                    pdb_b=pdbb,
                    pdb_mask_a=pdbma,
                    pdb_mask_b=pdbmb,
                )
                risk_logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(risk_logits.view(-1), y.float().view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            val_metrics, _, _ = evaluate_split(model, val_loader, device)
            if val_metrics["auroc"] > best_val_auc:
                best_val_auc = val_metrics["auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"Epoch {ep:02d}/{epochs:02d} - Loss: {train_loss / len(train_loader):.4f} - Val AUROC: {val_metrics['auroc']:.4f}")

        # Load best validation model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Optimize Youden index threshold on validation set
        val_metrics, val_p, val_y = evaluate_split(model, val_loader, device)
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(val_y, val_p)
        j_scores = 2.0 * tpr - fpr  # Cost-sensitive matching pos_weight=2.0
        best_thresh = float(thresholds[np.argmax(j_scores)]) if len(thresholds) > 0 else 0.50

        # Evaluate on held-out scaffold test set
        test_m, test_p, test_y = evaluate_split(model, test_loader, device, threshold=best_thresh)
        train_elapsed = time.time() - start_time

        test_probs[model_name] = test_p
        test_labels = test_y

        results[model_name] = {
            "validation_auroc": val_metrics["auroc"],
            "optimal_threshold": best_thresh,
            "test_metrics": test_m,
            "training_time_seconds": train_elapsed,
        }
        print(f"Finished {model_name}: Scaffold Test AUROC = {test_m['auroc']:.4f}, AUPRC = {test_m['auprc']:.4f}")

    if test_labels is None:
        raise RuntimeError("No test labels were evaluated during scaffold study.")

    # Bootstrap hypothesis testing
    stat_comparison = paired_bootstrap_comparison(
        test_labels,
        test_probs["baseline_gnn"],
        test_probs["auditddi_multimodal"],
        seed=seed,
    )
    results["statistical_comparison"] = stat_comparison

    # Export artifacts
    with open(out_p / "scaffold_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Export CSV summary
    rows = []
    for m_name in ["baseline_gnn", "auditddi_multimodal"]:
        m_data = results[m_name]["test_metrics"]
        rows.append({
            "model": m_name,
            "test_auroc": m_data["auroc"],
            "test_auprc": m_data["auprc"],
            "balanced_accuracy": m_data["balanced_accuracy"],
            "sensitivity": m_data["sensitivity"],
            "specificity": m_data["specificity"],
            "brier_score": m_data["brier_score"],
            "ece": m_data["ece"],
        })
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_p / "scaffold_benchmark_summary.csv", index=False)

    # Generate Markdown Report
    base_m = results["baseline_gnn"]["test_metrics"]
    multi_m = results["auditddi_multimodal"]["test_metrics"]
    ci_auc = stat_comparison["delta_auroc_ci95"]
    ci_ap = stat_comparison["delta_auprc_ci95"]

    md_content = f"""# AuditDDI Murcko Scaffold-Disjoint Generalization Study

## Executive Summary
This study measures the true out-of-distribution chemical generalization of **AuditDDI Multimodal** compared to a **Baseline Molecular Graph GNN** across disjoint Bemis-Murcko molecular scaffolds.

| Model | Test AUROC | Test AUPRC | Balanced Acc | Sensitivity (Recall) | Specificity | Brier Score | ECE |
|---|---|---|---|---|---|---|---|
| **Baseline 2D GNN** | {base_m['auroc']:.4f} | {base_m['auprc']:.4f} | {base_m['balanced_accuracy']:.4f} | {base_m['sensitivity']:.4f} | {base_m['specificity']:.4f} | {base_m['brier_score']:.4f} | {base_m['ece']:.4f} |
| **AuditDDI Multimodal** | **{multi_m['auroc']:.4f}** | **{multi_m['auprc']:.4f}** | **{multi_m['balanced_accuracy']:.4f}** | **{multi_m['sensitivity']:.4f}** | **{multi_m['specificity']:.4f}** | **{multi_m['brier_score']:.4f}** | **{multi_m['ece']:.4f}** |

## Statistical Significance (1,000-Iteration Paired Bootstrap)
- **$\\Delta$ AUROC**: {stat_comparison['delta_auroc_mean']:+.4f} (95% CI: [{ci_auc[0]:+.4f}, {ci_auc[1]:+.4f}])
- **$\\Delta$ AUPRC**: {stat_comparison['delta_auprc_mean']:+.4f} (95% CI: [{ci_ap[0]:+.4f}, {ci_ap[1]:+.4f}])
- **Paired Wilcoxon $p$-value**: {stat_comparison['p_value']:.4e}
"""
    with open(out_p / "scaffold_performance_comparison.md", "w") as f:
        f.write(md_content)

    print(f"\nBenchmark artifacts generated in: {out_p}")
    return results
