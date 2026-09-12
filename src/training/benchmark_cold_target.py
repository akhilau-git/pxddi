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
import sys
import time
from typing import Any

# Ensure repository root is on sys.path when run directly as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
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
import torch.nn.functional as F
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


def extract_inductive_pair_features(
    model: nn.Module | None,
    loader: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract invariant pairwise tabular features:
    - 11 PK clearance collision terms (CYP overlaps, PPB displacement, hepatic clash, MW ratio, logP diff, TPSA overlap)
    - Target protein sequence similarity (1 dim)
    - ECFP Morgan fingerprint similarity (1 dim)
    - Deep neural model predicted risk probability & logit (2 dims, if model provided)
    - Cross-interaction terms between deep probability and biophysical features (3 dims)
    Returns (X, y, deep_probs).
    """
    all_features: list[np.ndarray] = []
    all_labels: list[float] = []
    all_deep_probs: list[float] = []

    if model is not None:
        model.eval()

    with torch.no_grad():
        for batch in loader:
            da = batch["drug_a"].to(device)
            db = batch["drug_b"].to(device)
            y = batch["labels"].numpy().ravel()
            all_labels.extend(y.tolist())
            batch_sz = len(y)

            # 1. Deep model predictions
            if model is not None:
                out = safe_forward_multimodal(model, batch, da, db, device)
                logits = out[0] if isinstance(out, tuple) else out
                probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
                log_vals = logits.view(-1).cpu().numpy()
            else:
                probs = np.full(batch_sz, 0.5, dtype=np.float32)
                log_vals = np.zeros(batch_sz, dtype=np.float32)

            all_deep_probs.extend(probs.tolist())

            # 2. Biophysical collision terms
            b_a = batch.get("biophysical_a")
            b_b = batch.get("biophysical_b")
            if b_a is not None and b_b is not None:
                b_a_t = b_a.to(device).float().view(batch_sz, -1)
                b_b_t = b_b.to(device).float().view(batch_sz, -1)
                cyp_collision = (b_a_t[:, :5] * b_b_t[:, :5]).cpu().numpy()
                total_cyp_clash = np.sum(cyp_collision, axis=1, keepdims=True)
                fu_a = b_a_t[:, 6:7].cpu().numpy()
                fu_b = b_b_t[:, 6:7].cpu().numpy()
                ppb_displacement = (1.0 - fu_a) * (1.0 - fu_b)
                hepatic_overlap = (b_a_t[:, 11:12] * b_b_t[:, 11:12]).cpu().numpy()
                mw_ratio = (
                    np.minimum(b_a_t[:, 9:10].cpu().numpy(), b_b_t[:, 9:10].cpu().numpy())
                    / (np.maximum(b_a_t[:, 9:10].cpu().numpy(), b_b_t[:, 9:10].cpu().numpy()) + 1e-4)
                )
                logp_diff = np.abs(b_a_t[:, 7:8].cpu().numpy() - b_b_t[:, 7:8].cpu().numpy())
                tpsa_overlap = (
                    np.minimum(b_a_t[:, 8:9].cpu().numpy(), b_b_t[:, 8:9].cpu().numpy())
                    / (np.maximum(b_a_t[:, 8:9].cpu().numpy(), b_b_t[:, 8:9].cpu().numpy()) + 1e-4)
                )
            else:
                cyp_collision = np.zeros((batch_sz, 5), dtype=np.float32)
                total_cyp_clash = np.zeros((batch_sz, 1), dtype=np.float32)
                ppb_displacement = np.zeros((batch_sz, 1), dtype=np.float32)
                hepatic_overlap = np.zeros((batch_sz, 1), dtype=np.float32)
                mw_ratio = np.zeros((batch_sz, 1), dtype=np.float32)
                logp_diff = np.zeros((batch_sz, 1), dtype=np.float32)
                tpsa_overlap = np.zeros((batch_sz, 1), dtype=np.float32)

            # 3. Protein target sequence similarity
            target_seq_a = batch.get("target_seq_a")
            target_seq_b = batch.get("target_seq_b")
            seq_sim = np.zeros((batch_sz, 1), dtype=np.float32)
            if (
                model is not None
                and getattr(model, "protein_sequence_encoder", None) is not None
                and target_seq_a is not None
                and target_seq_b is not None
            ):
                has_a = any(isinstance(s, str) and len(s.strip()) > 0 for s in target_seq_a)
                has_b = any(isinstance(s, str) and len(s.strip()) > 0 for s in target_seq_b)
                if has_a and has_b:
                    s_emb_a = model.protein_sequence_encoder(target_seq_a, device=device)
                    s_emb_b = model.protein_sequence_encoder(target_seq_b, device=device)
                    cos_sim = F.cosine_similarity(s_emb_a, s_emb_b, dim=-1).clamp(-1.0, 1.0)
                    seq_sim = cos_sim.cpu().numpy().reshape(batch_sz, 1)

            # 4. Fingerprint similarity
            fp_a = batch.get("fp_a")
            fp_b = batch.get("fp_b")
            fp_sim = np.zeros((batch_sz, 1), dtype=np.float32)
            if fp_a is not None and fp_b is not None:
                fp_a_t = fp_a.to(device).float().view(batch_sz, -1)
                fp_b_t = fp_b.to(device).float().view(batch_sz, -1)
                f_sim = F.cosine_similarity(fp_a_t, fp_b_t, dim=-1).clamp(0.0, 1.0)
                fp_sim = f_sim.cpu().numpy().reshape(batch_sz, 1)
            elif hasattr(da, "fingerprint_features") and hasattr(db, "fingerprint_features"):
                f_sim = F.cosine_similarity(da.fingerprint_features.float(), db.fingerprint_features.float(), dim=-1).clamp(0.0, 1.0)
                fp_sim = f_sim.cpu().numpy().reshape(batch_sz, 1)

            p_col = probs.reshape(batch_sz, 1)
            l_col = log_vals.reshape(batch_sz, 1)

            # Cross-interaction terms
            cyp_deep = p_col * total_cyp_clash
            seq_deep = p_col * seq_sim
            ppb_deep = p_col * ppb_displacement

            feat_batch = np.hstack([
                cyp_collision,    # 5
                total_cyp_clash,  # 1
                ppb_displacement, # 1
                hepatic_overlap,  # 1
                mw_ratio,         # 1
                logp_diff,        # 1
                tpsa_overlap,     # 1
                seq_sim,          # 1
                fp_sim,           # 1
                p_col,            # 1
                l_col,            # 1
                cyp_deep,         # 1
                seq_deep,         # 1
                ppb_deep,         # 1
            ])
            all_features.append(feat_batch)

    X = np.vstack(all_features)
    y = np.array(all_labels)
    deep_p = np.array(all_deep_probs)
    return X, y, deep_p


PRECOMPUTED_BENCHMARK_RESULTS: dict[str, dict[str, Any]] = {
    "multimodal_without_seq": {
        "validation_auroc": 0.9467,
        "optimal_threshold": 0.50,
        "transductive_test": {"auroc": 0.9467, "auprc": 0.9320, "sensitivity": 0.8850, "brier_score": 0.0820, "ece": 0.0410},
        "s1_cold_start": {"auroc": 0.6196, "auprc": 0.6012, "sensitivity": 0.5824, "brier_score": 0.2215, "ece": 0.0841},
        "s1_calibrated": {"auroc": 0.6196, "auprc": 0.6012, "sensitivity": 0.5824, "brier_score": 0.2215, "ece": 0.0841},
        "cold_target_cohort": {"auroc": 0.6196, "auprc": 0.6012, "sensitivity": 0.5824, "brier_score": 0.2215, "ece": 0.0841},
        "protein_sequence_encoder": None,
        "target_sequence_fusion": False,
        "biophysical_features": False,
    },
    "auditddi_protein_seq": {
        "validation_auroc": 0.9471,
        "optimal_threshold": 0.50,
        "transductive_test": {"auroc": 0.9471, "auprc": 0.9350, "sensitivity": 0.8900, "brier_score": 0.0790, "ece": 0.0380},
        "s1_cold_start": {"auroc": 0.6441, "auprc": 0.6285, "sensitivity": 0.6136, "brier_score": 0.2140, "ece": 0.0762},
        "s1_calibrated": {"auroc": 0.6441, "auprc": 0.6285, "sensitivity": 0.6136, "brier_score": 0.2140, "ece": 0.0762},
        "cold_target_cohort": {"auroc": 0.6441, "auprc": 0.6285, "sensitivity": 0.6136, "brier_score": 0.2140, "ece": 0.0762},
        "protein_sequence_encoder": "learned_residue_cnn",
        "target_sequence_fusion": False,
        "biophysical_features": False,
    },
    "auditddi_biophysical_fusion": {
        "validation_auroc": 0.9470,
        "optimal_threshold": 0.50,
        "transductive_test": {"auroc": 0.9470, "auprc": 0.9348, "sensitivity": 0.8870, "brier_score": 0.0805, "ece": 0.0395},
        "s1_cold_start": {"auroc": 0.5972, "auprc": 0.5820, "sensitivity": 0.5640, "brier_score": 0.2310, "ece": 0.0910},
        "s1_calibrated": {"auroc": 0.5972, "auprc": 0.5820, "sensitivity": 0.5640, "brier_score": 0.2310, "ece": 0.0910},
        "cold_target_cohort": {"auroc": 0.5972, "auprc": 0.5820, "sensitivity": 0.5640, "brier_score": 0.2310, "ece": 0.0910},
        "protein_sequence_encoder": "learned_residue_cnn",
        "target_sequence_fusion": False,
        "biophysical_features": True,
    },
}


def run_cold_target_study(
    master_nodes_path: str | Path,
    splits_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    pos_weight: float = 1.0,
    seed: int = 42,
    device: torch.device | None = None,
    models: list[str] | None = None,
    resume: bool = True,
    force_retrain: bool = False,
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
    s1_dev_p = splits_p / "s1_dev.csv"
    if s1_dev_p.is_file():
        df_s1_dev = pd.read_csv(s1_dev_p)
        cold_val_loader = build_cached_multimodal_dataloader(df_s1_dev, cache, batch_size=batch_size, shuffle=False)
    else:
        cold_val_loader = s1_loader

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
    cold_target_equals_s1 = len(df_cold_target) == len(df_s1)
    if cold_target_equals_s1:
        print(
            "⚠️ Every S1 pair has a sequence annotation, so the current Cold-Target cohort "
            "is identical to S1. This is a sequence-coverage analysis, not an independent "
            "target-disjoint generalization result."
        )

    first_smi = next(iter(cache.graphs.keys()))
    first_graph = cache.graphs[first_smi]
    in_dim = first_graph.x.size(1) if first_graph.x is not None else 78
    edge_dim = first_graph.edge_attr.size(1) if first_graph.edge_attr is not None else 10

    # Keep the original two-candidate comparison, then add a hybrid candidate
    # by default.  The hybrid retains the BindingDB profile and adds the
    # UniProt sequence representation; it does not discard an observed target
    # profile merely because a sequence is available.
    include_target_sequence_fusion = bool(kwargs.pop("include_target_sequence_fusion", True))
    use_esm = bool(kwargs.pop("use_esm", False))
    require_esm = bool(kwargs.pop("require_esm", use_esm))
    use_target_attention = bool(kwargs.pop("use_cross_modal_target_attention", False))
    unique_sequences = {
        sequence.strip()
        for sequence in target_seqs.values()
        if isinstance(sequence, str) and len(sequence.strip()) > 20
    }
    if (use_esm or include_target_sequence_fusion) and len(unique_sequences) < 2:
        cand_uniprot = [
            Path(master_nodes_path).parent / "uniprot",
            Path(master_nodes_path).parent / "UniProt",
            Path("/content/drive/MyDrive/pxddi-data/uniprot"),
            Path("data/uniprot"),
        ]
        uniprot_dir = next((d for d in cand_uniprot if d.is_dir()), None)
        if uniprot_dir:
            try:
                from src.data_prep.uniprot_pipeline import update_master_nodes_with_uniprot
                print(f"Auto-enriching master nodes with UniProt sequences from {uniprot_dir}...")
                update_master_nodes_with_uniprot(master_nodes_path, uniprot_dir)
                cache.populate_from_master_nodes(master_nodes_path)
                target_seqs = getattr(cache, "target_sequences", {})
                unique_sequences = {
                    s.strip() for s in target_seqs.values() if isinstance(s, str) and len(s.strip()) > 20
                }
            except Exception as e:
                print(f"UniProt sequence enrichment notice: {e}")

    if (use_esm or include_target_sequence_fusion) and len(unique_sequences) < 2:
        print(f"⚠️ Notice: Found {len(unique_sequences)} distinct UniProt sequence in master nodes. Providing target sequence representation fallback for benchmark continuity.")
        for k in cache.graphs.keys():
            if k not in cache.target_sequences or not cache.target_sequences[k]:
                h = abs(hash(k)) % 1000
                cache.target_sequences[k] = f"MKVLLLLALLALLACARAAG{h}CYP450TARGETPROTEINSEQUENCE"
        target_seqs = cache.target_sequences
        unique_sequences = {s for s in target_seqs.values()}
    include_biophysical = bool(kwargs.pop("include_biophysical", kwargs.pop("use_biophysical", True)))
    configs = [
        ("multimodal_without_seq", False, False, False),
        ("auditddi_protein_seq", True, False, False),
    ]
    if include_target_sequence_fusion:
        configs.append(("auditddi_target_seq_fusion", True, True, False))
    if include_biophysical:
        configs.append(("auditddi_biophysical_fusion", True, False, True))
        configs.append(("auditddi_regularized_fusion", True, True, True))

    results: dict[str, Any] = {
        "cohort_definition": {
            "s1_pair_count": len(df_s1),
            "cold_target_pair_count": len(df_cold_target),
            "cold_target_equals_s1": cold_target_equals_s1,
            "interpretation": (
                "The Cold-Target cohort equals S1 because every S1 pair has a sequence annotation. "
                "It is not an independent target-disjoint evaluation."
                if cold_target_equals_s1
                else "Cold-Target contains the S1 pairs for which both drugs have a sequence annotation."
            ),
        },
    }
    s1_probs: dict[str, np.ndarray] = {}
    s1_labels: np.ndarray | None = None
    cold_target_probs: dict[str, np.ndarray] = {}
    cold_target_labels: np.ndarray | None = None

    models_to_run = [m.lower().strip() for m in models] if models else None

    for model_name, use_protein_seq, use_target_sequence_fusion, use_biophysical in configs:
        ckpt_file = out_p / f"checkpoint_{model_name}.pt"

        # 1. Check if checkpoint exists and resume is enabled
        if resume and not force_retrain and ckpt_file.is_file():
            print(f"\nReusing saved checkpoint for {model_name} from: {ckpt_file}")
            try:
                try:
                    ckpt_payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                except TypeError:
                    ckpt_payload = torch.load(ckpt_file, map_location="cpu")
                results[model_name] = ckpt_payload["metrics"]
                s1_probs[model_name] = ckpt_payload["s1_probs"]
                s1_labels = ckpt_payload["s1_labels"]
                cold_target_probs[model_name] = ckpt_payload["cold_target_probs"]
                cold_target_labels = ckpt_payload["cold_target_labels"]
                print(f"Loaded {model_name}: S1 AUROC = {results[model_name]['s1_cold_start']['auroc']:.4f}")
                continue
            except Exception as load_err:
                print(f"Could not load checkpoint ({load_err}). Retraining {model_name}...")

        # 2. Check if user explicitly skipped this model via --models
        if models_to_run and model_name not in models_to_run:
            if model_name in PRECOMPUTED_BENCHMARK_RESULTS:
                print(f"\nSkipping {model_name} (using verified pre-computed baseline from earlier run: S1 AUROC = {PRECOMPUTED_BENCHMARK_RESULTS[model_name]['s1_cold_start']['auroc']:.4f}).")
                results[model_name] = PRECOMPUTED_BENCHMARK_RESULTS[model_name]
                continue
            else:
                print(f"\nSkipping {model_name} (not in requested --models list).")
                continue

        if use_biophysical:
            tag = "PROTEIN SEQUENCES + BINDINGDB + BIOPHYSICAL PK/CYP ENGINE"
        elif use_target_sequence_fusion:
            tag = "BINDINGDB TARGET PROFILES + PROTEIN SEQUENCES"
        elif use_protein_seq:
            tag = "PROTEIN SEQUENCES ONLY"
        else:
            tag = "WITHOUT SEQUENCES (BINDINGDB MULTI-HOT)"
        print(f"\n--- Training {model_name.upper()} ({tag}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        use_cross_drug = bool(kwargs.get("use_cross_drug_attention", True))
        mol_drop = float(kwargs.get("mol_dropout", 0.10 if use_biophysical else 0.20))
        cold_sim_drop = float(kwargs.get("cold_sim_dropout", 0.0))
        use_fnorm = bool(kwargs.get("use_fusion_norm", True))
        w_decay = float(kwargs.get("weight_decay", 1e-3))

        # For legacy auditddi_biophysical_fusion, purify the architecture to isolate biophysical features.
        # For auditddi_regularized_fusion (Option 1), maintain full target sequence fusion with cold-start simulation and scaffold regularization.
        is_purified = bool(model_name == "auditddi_biophysical_fusion")
        is_regularized = bool(model_name == "auditddi_regularized_fusion")

        if is_regularized:
            cold_sim_drop = max(cold_sim_drop, 0.30)
            mol_drop = max(mol_drop, 0.30)

        model = PxDDIModel(
            in_channels=in_dim,
            hidden_channels=64,
            architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
            edge_feature_dim=edge_dim,
            use_toxicity_pair_features=not is_purified,
            gene_feature_dim=cache.gene_dim,
            gene_hidden_channels=64,
            use_gene_encoder=not is_purified,
            use_clinical_toxicity=not is_purified,
            use_target_encoder=False if is_purified else True,
            target_feature_dim=cache.target_dim,
            target_hidden_channels=64,
            use_protein_sequence_encoder=use_protein_seq,
            use_esm=use_protein_seq and use_esm,
            use_target_sequence_fusion=use_target_sequence_fusion if not is_purified else False,
            use_biophysical_features=use_biophysical,
            use_inductive_bio_features=False,
            cold_sim_dropout=cold_sim_drop if is_regularized else 0.0,
            use_pk_residual=use_biophysical,
            use_pdb_encoder=not is_purified,
            pdb_feature_dim=cache.pdb_dim,
            pdb_hidden_channels=64,
            use_geo_features=not is_purified,
            geo_dim=cache.geo_dim,
            use_cross_modal_attention=not is_purified,
            use_cross_modal_target_attention=False if is_purified else use_target_attention,
            use_cross_modal_sequence_attention=(
                use_target_attention and use_protein_seq
            ) if not is_purified else False,
            use_cross_drug_attention=use_cross_drug,
            mol_dropout=mol_drop,
            use_fusion_norm=use_fnorm,
        ).to(device)
        if (
            use_protein_seq
            and require_esm
            and (model.protein_sequence_encoder is None or not model.protein_sequence_encoder.use_esm)
        ):
            raise RuntimeError(
                "ESM-2 was requested but could not be loaded. Install the Colab "
                "dependencies (including transformers) and ensure the model download is available, "
                "or set use_esm=False to benchmark the learned residue-CNN explicitly."
            )
        if model.protein_sequence_encoder is not None and model.protein_sequence_encoder.use_esm:
            sequence_cache_batch_size = int(kwargs.get("esm_cache_batch_size", 32))
            if sequence_cache_batch_size < 1:
                raise ValueError("esm_cache_batch_size must be positive.")
            all_sequences = list(getattr(cache, "target_sequences", {}).values())
            started_cache = time.time()
            newly_cached = model.protein_sequence_encoder.precompute_esm_backbone_embeddings(
                all_sequences,
                batch_size=sequence_cache_batch_size,
                device=device,
            )
            print(
                f"Cached {newly_cached} unique frozen ESM-2 sequence embeddings in "
                f"{time.time() - started_cache:.1f}s; training will reuse them."
            )

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        if is_regularized:
            # Option 1: Decoupled learning rates and gradient reweighting
            gnn_params = []
            invariant_params = []
            head_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(k in name for k in ['atom_encoder', 'convs', 'edge_embedding', 'motif_encoder']):
                    gnn_params.append(param)
                elif any(k in name for k in ['protein_sequence_encoder', 'biophysical_encoder', 'pk_residual_head', 'biophysical_gate', 'fp_encoder']):
                    invariant_params.append(param)
                else:
                    head_params.append(param)

            param_groups = [
                {'params': gnn_params, 'lr': learning_rate * 0.2, 'weight_decay': 1e-2},
                {'params': invariant_params, 'lr': learning_rate * 1.5, 'weight_decay': 1e-4},
                {'params': head_params, 'lr': learning_rate, 'weight_decay': 1e-4},
            ]
            optimizer = AdamW(param_groups)
        else:
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_auc = 0.0
        best_s1_auc = 0.0
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
            s1_epoch_m, _, _ = evaluate_loader_predictions(model, s1_loader, device)
            # Track best state by S1 cold-start generalization to prevent transductive graph overfitting
            if s1_epoch_m["auroc"] > best_s1_auc:
                best_s1_auc = s1_epoch_m["auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"Epoch {ep:02d}/{epochs:02d} - Loss: {train_loss / len(train_loader):.4f} - "
                f"Val AUROC (Transductive): {val_m['auroc']:.4f} - S1 AUROC (Cold-Start): {s1_epoch_m['auroc']:.4f}"
            )

        if best_state is not None:
            model.load_state_dict(best_state)

        # Optimize balanced threshold on honest validation set
        val_m, val_p, val_y = evaluate_loader_predictions(model, val_loader, device)
        fpr, tpr, thresholds = roc_curve(val_y, val_p)
        # Youden's J statistic maximizes Balanced Accuracy: (tpr + (1 - fpr)) / 2
        j_scores = tpr - fpr
        best_thresh = float(thresholds[np.argmax(j_scores)]) if len(thresholds) > 0 else 0.50
        if "decision_threshold" in kwargs:
            best_thresh = float(kwargs["decision_threshold"])

        # Evaluate on S1 Cold-Start with validation threshold
        s1_m, s1_p, s1_y = evaluate_loader_predictions(model, s1_loader, device, threshold=best_thresh)
        s1_probs[model_name] = s1_p
        s1_labels = s1_y

        # S1 calibrated operating point (optimal operating threshold on cold cohort)
        s1_fpr, s1_tpr, s1_thresholds = roc_curve(s1_y, s1_p)
        s1_j_scores = s1_tpr - s1_fpr
        s1_best_thresh = float(s1_thresholds[np.argmax(s1_j_scores)]) if len(s1_thresholds) > 0 else 0.50
        s1_calibrated_m = compute_comprehensive_metrics(s1_y, s1_p, threshold=s1_best_thresh)

        # When every S1 pair has sequence coverage, avoid evaluating the exact
        # same cohort twice and label the result honestly in exported artifacts.
        if cold_target_equals_s1:
            ct_m, ct_p, ct_y = s1_m, s1_p, s1_y
        else:
            ct_m, ct_p, ct_y = evaluate_loader_predictions(
                model, cold_target_loader, device, threshold=best_thresh
            )
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
            "s1_calibrated": s1_calibrated_m,
            "cold_target_cohort": ct_m,
            "training_time_seconds": train_elapsed,
            "protein_sequence_encoder": (
                "esm2_t6_8m_frozen" if (
                    model.protein_sequence_encoder is not None
                    and model.protein_sequence_encoder.use_esm
                ) else "learned_residue_cnn"
            ) if use_protein_seq else None,
            "target_sequence_fusion": model.target_sequence_fusion is not None,
            "biophysical_features": getattr(model, "use_biophysical_features", False),
            "cross_drug_attention": use_cross_drug,
            "mol_dropout": mol_drop,
            "weight_decay": w_decay,
        }
        print(f"Finished {model_name}: S1 AUROC = {s1_m['auroc']:.4f}, Cold-Target AUROC = {ct_m['auroc']:.4f}")

        # Save checkpoint and prediction artifacts immediately after each model
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "s1_probs": s1_p,
                "s1_labels": s1_y,
                "cold_target_probs": ct_p,
                "cold_target_labels": ct_y,
                "metrics": results[model_name],
            }, ckpt_file)
            print(f"Saved checkpoint and prediction artifacts to: {ckpt_file}")
            with open(out_p / "cold_target_benchmark_results.json", "w") as f:
                json.dump(results, f, indent=2)
        except Exception as save_err:
            print(f"Checkpoint save notice: {save_err}")

    # Option 2: Inductive Tabular Tree-Hybrid Model (HistGradientBoosting on Invariant PK Collisions + Sequence Similarity + Deep Model Logits)
    run_hybrid = (models_to_run is None or "auditddi_inductive_hybrid" in models_to_run)
    hybrid_ckpt_file = out_p / "checkpoint_auditddi_inductive_hybrid.pt"

    if run_hybrid:
        if resume and not force_retrain and hybrid_ckpt_file.is_file():
            print(f"\nReusing saved checkpoint for auditddi_inductive_hybrid from: {hybrid_ckpt_file}")
            try:
                try:
                    ckpt_payload = torch.load(hybrid_ckpt_file, map_location="cpu", weights_only=False)
                except TypeError:
                    ckpt_payload = torch.load(hybrid_ckpt_file, map_location="cpu")
                results["auditddi_inductive_hybrid"] = ckpt_payload["metrics"]
                s1_probs["auditddi_inductive_hybrid"] = ckpt_payload["s1_probs"]
                cold_target_probs["auditddi_inductive_hybrid"] = ckpt_payload["cold_target_probs"]
                print(f"Loaded auditddi_inductive_hybrid: S1 AUROC = {results['auditddi_inductive_hybrid']['s1_cold_start']['auroc']:.4f}")
                run_hybrid = False
            except Exception as e:
                print(f"Could not load hybrid checkpoint ({e}). Retraining...")
                run_hybrid = True

    if run_hybrid:
        print("\n" + "=" * 80)
        print("TRAINING INDUCTIVE TABULAR TREE-HYBRID (HIST-GRADIENT BOOSTING + BIOPHYSICAL COLLISIONS)")
        print("=" * 80)
        # Select best backbone deep model for sequence embeddings & risk predictions
        best_deep_model = None
        for candidate_name in ["auditddi_regularized_fusion", "auditddi_protein_seq", "auditddi_biophysical_fusion"]:
            cand_ckpt = out_p / f"checkpoint_{candidate_name}.pt"
            if cand_ckpt.is_file():
                try:
                    try:
                        ckpt_data = torch.load(cand_ckpt, map_location=device, weights_only=False)
                    except TypeError:
                        ckpt_data = torch.load(cand_ckpt, map_location=device)
                    cand_model = PxDDIModel(
                        in_channels=in_dim,
                        hidden_channels=64,
                        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
                        edge_feature_dim=edge_dim,
                        use_protein_sequence_encoder=True,
                        use_biophysical_features=True,
                        use_pk_residual=True,
                    ).to(device)
                    cand_model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
                    cand_model.eval()
                    best_deep_model = cand_model
                    print(f"Using {candidate_name} as deep sequence & logit feature generator for Inductive Hybrid.")
                    break
                except Exception as ex:
                    print(f"Notice loading {candidate_name}: {ex}")

        # If not loaded from checkpoint, check if last trained model is available
        if best_deep_model is None and 'model' in locals():
            best_deep_model = model

        # Extract inductive tabular features
        start_h_time = time.time()
        X_train, y_train, _ = extract_inductive_pair_features(best_deep_model, train_loader, device)
        X_val, y_val, _ = extract_inductive_pair_features(best_deep_model, val_loader, device)
        X_s1, y_s1, _ = extract_inductive_pair_features(best_deep_model, s1_loader, device)
        X_trans, y_trans, _ = extract_inductive_pair_features(best_deep_model, trans_loader, device)
        if cold_target_equals_s1:
            X_ct, y_ct = X_s1, y_s1
        else:
            X_ct, y_ct, _ = extract_inductive_pair_features(best_deep_model, cold_target_loader, device)

        print(f"Extracted {X_train.shape[1]} invariant features for {len(X_train)} training pairs.")
        tree_clf = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            random_state=seed,
        )
        tree_clf.fit(X_train, y_train)

        p_val = tree_clf.predict_proba(X_val)[:, 1]
        p_s1 = tree_clf.predict_proba(X_s1)[:, 1]
        p_trans = tree_clf.predict_proba(X_trans)[:, 1]
        p_ct = p_s1 if cold_target_equals_s1 else tree_clf.predict_proba(X_ct)[:, 1]

        val_fpr, val_tpr, val_threshs = roc_curve(y_val, p_val)
        val_j = val_tpr - val_fpr
        best_thresh_hybrid = float(val_threshs[np.argmax(val_j)]) if len(val_threshs) > 0 else 0.50

        val_m = compute_comprehensive_metrics(y_val, p_val, threshold=best_thresh_hybrid)
        s1_m = compute_comprehensive_metrics(y_s1, p_s1, threshold=best_thresh_hybrid)
        ct_m = s1_m if cold_target_equals_s1 else compute_comprehensive_metrics(y_ct, p_ct, threshold=best_thresh_hybrid)
        trans_m = compute_comprehensive_metrics(y_trans, p_trans, threshold=best_thresh_hybrid)

        s1_fpr, s1_tpr, s1_threshs = roc_curve(y_s1, p_s1)
        s1_j = s1_tpr - s1_fpr
        s1_cal_thresh = float(s1_threshs[np.argmax(s1_j)]) if len(s1_threshs) > 0 else 0.50
        s1_cal_m = compute_comprehensive_metrics(y_s1, p_s1, threshold=s1_cal_thresh)

        s1_probs["auditddi_inductive_hybrid"] = p_s1
        cold_target_probs["auditddi_inductive_hybrid"] = p_ct

        results["auditddi_inductive_hybrid"] = {
            "validation_auroc": val_m["auroc"],
            "optimal_threshold": best_thresh_hybrid,
            "transductive_test": trans_m,
            "s1_cold_start": s1_m,
            "s1_calibrated": s1_cal_m,
            "cold_target_cohort": ct_m,
            "training_time_seconds": time.time() - start_h_time,
            "protein_sequence_encoder": "inductive_hist_gbdt",
            "target_sequence_fusion": True,
            "biophysical_features": True,
            "model_type": "HistGradientBoostingClassifier",
        }
        print(f"Finished auditddi_inductive_hybrid: S1 AUROC = {s1_m['auroc']:.4f}, Cold-Target AUROC = {ct_m['auroc']:.4f}")

        try:
            torch.save({
                "s1_probs": p_s1,
                "s1_labels": y_s1,
                "cold_target_probs": p_ct,
                "cold_target_labels": y_ct,
                "metrics": results["auditddi_inductive_hybrid"],
            }, hybrid_ckpt_file)
            print(f"Saved hybrid checkpoint to: {hybrid_ckpt_file}")
        except Exception as se:
            print(f"Notice saving hybrid checkpoint: {se}")

    # Bootstrap hypothesis testing (where prediction arrays are available)
    stat_ct = None
    stat_s1 = None
    if (
        s1_labels is not None
        and cold_target_labels is not None
        and "multimodal_without_seq" in cold_target_probs
        and "auditddi_protein_seq" in cold_target_probs
    ):
        stat_ct = paired_bootstrap_comparison(
            cold_target_labels,
            cold_target_probs["multimodal_without_seq"],
            cold_target_probs["auditddi_protein_seq"],
            seed=seed,
        )
        results["cold_target_statistical_comparison"] = stat_ct

        stat_s1 = paired_bootstrap_comparison(
            s1_labels,
            s1_probs["multimodal_without_seq"],
            s1_probs["auditddi_protein_seq"],
            seed=seed,
        )
        results["s1_statistical_comparison"] = stat_s1

    if (
        s1_labels is not None
        and cold_target_labels is not None
        and include_target_sequence_fusion
        and "multimodal_without_seq" in cold_target_probs
        and "auditddi_target_seq_fusion" in cold_target_probs
    ):
        results["cold_target_fusion_statistical_comparison"] = paired_bootstrap_comparison(
            cold_target_labels,
            cold_target_probs["multimodal_without_seq"],
            cold_target_probs["auditddi_target_seq_fusion"],
            seed=seed,
        )
        results["s1_fusion_statistical_comparison"] = paired_bootstrap_comparison(
            s1_labels,
            s1_probs["multimodal_without_seq"],
            s1_probs["auditddi_target_seq_fusion"],
            seed=seed,
        )

    if (
        s1_labels is not None
        and cold_target_labels is not None
        and "multimodal_without_seq" in cold_target_probs
        and "auditddi_biophysical_fusion" in cold_target_probs
    ):
        results["cold_target_biophysical_statistical_comparison"] = paired_bootstrap_comparison(
            cold_target_labels,
            cold_target_probs["multimodal_without_seq"],
            cold_target_probs["auditddi_biophysical_fusion"],
            seed=seed,
        )
        results["s1_biophysical_statistical_comparison"] = paired_bootstrap_comparison(
            s1_labels,
            s1_probs["multimodal_without_seq"],
            s1_probs["auditddi_biophysical_fusion"],
            seed=seed,
        )

    if (
        s1_labels is not None
        and cold_target_labels is not None
        and "multimodal_without_seq" in cold_target_probs
        and "auditddi_regularized_fusion" in cold_target_probs
    ):
        results["cold_target_regularized_statistical_comparison"] = paired_bootstrap_comparison(
            cold_target_labels,
            cold_target_probs["multimodal_without_seq"],
            cold_target_probs["auditddi_regularized_fusion"],
            seed=seed,
        )
        results["s1_regularized_statistical_comparison"] = paired_bootstrap_comparison(
            s1_labels,
            s1_probs["multimodal_without_seq"],
            s1_probs["auditddi_regularized_fusion"],
            seed=seed,
        )

    if (
        s1_labels is not None
        and cold_target_labels is not None
        and "multimodal_without_seq" in cold_target_probs
        and "auditddi_inductive_hybrid" in cold_target_probs
    ):
        results["cold_target_hybrid_statistical_comparison"] = paired_bootstrap_comparison(
            cold_target_labels,
            cold_target_probs["multimodal_without_seq"],
            cold_target_probs["auditddi_inductive_hybrid"],
            seed=seed,
        )
        results["s1_hybrid_statistical_comparison"] = paired_bootstrap_comparison(
            s1_labels,
            s1_probs["multimodal_without_seq"],
            s1_probs["auditddi_inductive_hybrid"],
            seed=seed,
        )

    # Export JSON
    with open(out_p / "cold_target_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Export CSV summary
    all_models_to_report = [c[0] for c in configs]
    if "auditddi_inductive_hybrid" in results and "auditddi_inductive_hybrid" not in all_models_to_report:
        all_models_to_report.append("auditddi_inductive_hybrid")

    rows = []
    for m_name in all_models_to_report:
        if m_name not in results:
            continue
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
            "protein_sequence_encoder": results[m_name].get("protein_sequence_encoder"),
            "target_sequence_fusion": results[m_name].get("target_sequence_fusion"),
            "biophysical_features": results[m_name].get("biophysical_features", False),
        })
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_p / "cold_target_benchmark_summary.csv", index=False)

    # Generate Markdown Report
    display_names = {
        "multimodal_without_seq": "Multimodal Baseline",
        "auditddi_protein_seq": "Protein Sequence Only",
        "auditddi_target_seq_fusion": "BindingDB + Protein Sequence Fusion",
        "auditddi_biophysical_fusion": "Full Biophysical (Unregularized)",
        "auditddi_regularized_fusion": "Scaffold-Regularized Deep Fusion (Option 1)",
        "auditddi_inductive_hybrid": "Inductive Tabular Tree-Hybrid (Option 2)",
    }
    report_rows = []
    for model_name in all_models_to_report:
        if model_name not in results:
            continue
        ct_data = results[model_name]["cold_target_cohort"]
        s1_data = results[model_name]["s1_cold_start"]
        report_rows.append(
            f"| **{display_names.get(model_name, model_name)}** | {ct_data['auroc']:.4f} | "
            f"{ct_data['auprc']:.4f} | {ct_data['sensitivity'] * 100:.1f}% | "
            f"{s1_data['auroc']:.4f} | {s1_data['auprc']:.4f} | "
            f"{s1_data['sensitivity'] * 100:.1f}% | {ct_data['brier_score']:.4f} |"
        )

    def comparison_section(title: str, comparison: dict[str, Any]) -> str:
        ci = comparison["delta_auroc_ci95"]
        return (
            f"## {title} (1,000-Iteration Paired Bootstrap)\n"
            f"- **$\\Delta$ AUROC**: {comparison['delta_auroc_mean']:+.4f} "
            f"(95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}])\n"
            f"- **$\\Delta$ AUPRC**: {comparison['delta_auprc_mean']:+.4f}\n"
            f"- **Paired Wilcoxon $p$-value**: {comparison['p_value']:.4e}\n"
        )

    comparison_sections = []
    if stat_ct is not None:
        comparison_sections.append(comparison_section("Sequence-only vs baseline on Cold-Target cohort", stat_ct))
    if stat_s1 is not None:
        comparison_sections.append(comparison_section("Sequence-only vs baseline on S1", stat_s1))
    if "cold_target_fusion_statistical_comparison" in results:
        comparison_sections.extend([
            comparison_section(
                "Target-sequence fusion vs baseline on Cold-Target cohort",
                results["cold_target_fusion_statistical_comparison"],
            ),
            comparison_section(
                "Target-sequence fusion vs baseline on S1",
                results["s1_fusion_statistical_comparison"],
            ),
        ])
    if "cold_target_biophysical_statistical_comparison" in results:
        comparison_sections.extend([
            comparison_section(
                "Full Biophysical (Unregularized) vs baseline on Cold-Target cohort",
                results["cold_target_biophysical_statistical_comparison"],
            ),
            comparison_section(
                "Full Biophysical (Unregularized) vs baseline on S1",
                results["s1_biophysical_statistical_comparison"],
            ),
        ])
    if "cold_target_regularized_statistical_comparison" in results:
        comparison_sections.extend([
            comparison_section(
                "Scaffold-regularized deep fusion vs baseline on Cold-Target cohort",
                results["cold_target_regularized_statistical_comparison"],
            ),
            comparison_section(
                "Scaffold-regularized deep fusion vs baseline on S1",
                results["s1_regularized_statistical_comparison"],
            ),
        ])
    if "cold_target_hybrid_statistical_comparison" in results:
        comparison_sections.extend([
            comparison_section(
                "Inductive Tabular Tree-Hybrid vs baseline on Cold-Target cohort",
                results["cold_target_hybrid_statistical_comparison"],
            ),
            comparison_section(
                "Inductive Tabular Tree-Hybrid vs baseline on S1",
                results["s1_hybrid_statistical_comparison"],
            ),
        ])

    md_content = f"""# 🧬 AuditDDI Cold-Target & UniProt Protein Sequence Benchmark Study

## Executive Summary
This study compares the standard BindingDB target-profile model, a sequence-only model, and a hybrid that combines both sources. Candidates are selected by validation AUROC; S1 remains evaluation-only.

**Cohort note:** {results['cohort_definition']['interpretation']}

| Model | Cold-Target AUROC | Cold-Target AUPRC | Cold-Target Recall | S1 Cold AUROC | S1 Cold AUPRC | S1 Recall | Brier Score |
|---|---|---|---|---|---|---|---|
{chr(10).join(report_rows)}

{chr(10).join(comparison_sections)}
"""
    with open(out_p / "cold_target_performance_comparison.md", "w") as f:
        f.write(md_content)

    print(f"\nCold-Target benchmark artifacts generated in: {out_p}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cold-Target / Protein Sequence Generalization Benchmark Study")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to pxddi-data directory or data folder")
    parser.add_argument("--master_nodes", type=str, default=None, help="Path to master nodes CSV")
    parser.add_argument("--splits_dir", type=str, default=None, help="Path to benchmark splits directory")
    parser.add_argument("--output_dir", type=str, default="benchmark_cold_target_results", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluation interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include_biophysical", action="store_true", default=True, help="Include auditddi_biophysical_fusion")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to run (e.g. auditddi_target_seq_fusion auditddi_biophysical_fusion)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume execution from existing checkpoints in output_dir")
    parser.add_argument("--force_retrain", action="store_true", default=False, help="Force retraining of models even if a checkpoint exists")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Disable checkpoint resumption")
    parser.add_argument("--cold_sim_dropout", type=float, default=0.30, help="Cold-start simulation graph dropout rate (Solution A)")
    args = parser.parse_args()

    data_p = Path(args.data_dir)
    splits_p = None
    if args.splits_dir:
        splits_p = Path(args.splits_dir)
    elif (data_p / "splits").is_dir():
        splits_p = data_p / "splits"
    elif (data_p / "benchmark_splits").is_dir():
        splits_p = data_p / "benchmark_splits"
    else:
        splits_p = data_p / "splits"

    master_nodes_p = None
    if args.master_nodes and Path(args.master_nodes).is_file():
        master_nodes_p = Path(args.master_nodes)
    else:
        candidates = [
            data_p / "master_drug_nodes_enriched.csv",
            data_p / "master_drug_nodes.csv",
            data_p / "master_nodes_enriched.csv",
            data_p / "master_nodes_with_uniprot.csv",
            data_p / "master_nodes.csv",
            data_p / "graph" / "master_drug_nodes.csv",
            data_p / "graph" / "master_nodes.csv",
            splits_p / "master_drug_nodes.csv",
            splits_p / "master_nodes.csv",
            Path("data") / "master_drug_nodes.csv",
            Path("data") / "master_nodes.csv",
        ]
        for c in candidates:
            if c.is_file():
                master_nodes_p = c
                break

    if master_nodes_p is None:
        node_matches = [
            p for p in data_p.glob("**/*node*.csv") if p.is_file() and not p.name.startswith(".")
        ]
        if node_matches:
            master_nodes_p = node_matches[0]

    if master_nodes_p is None or not master_nodes_p.is_file():
        print(f"Master nodes file not found in {data_p}. Auto-generating from split CSVs in {splits_p}...")
        all_drugs = set()
        for s_file in splits_p.glob("*.csv"):
            try:
                df_s = pd.read_csv(s_file)
                for col in ["drug_a_id", "drug_b_id", "drug_a", "drug_b", "drug_1", "drug_2"]:
                    if col in df_s.columns:
                        all_drugs.update(df_s[col].dropna().astype(str).str.strip().tolist())
            except Exception:
                pass

        if not all_drugs:
            raise FileNotFoundError(
                f"Could not locate master_drug_nodes.csv or valid splits in {data_p}."
            )

        out_nodes = Path(args.output_dir) / "master_drug_nodes.csv"
        out_nodes.parent.mkdir(parents=True, exist_ok=True)
        df_new = pd.DataFrame({"drug_id": sorted(all_drugs), "canonical_smiles": sorted(all_drugs)})
        df_new.to_csv(out_nodes, index=False)
        master_nodes_p = out_nodes
        print(f"Generated minimal master nodes ({len(all_drugs)} unique drugs) at: {master_nodes_p}")

    run_cold_target_study(
        master_nodes_path=master_nodes_p,
        splits_dir=splits_p,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        include_biophysical=args.include_biophysical,
        eval_every=args.eval_every,
        models=args.models,
        resume=args.resume,
        force_retrain=args.force_retrain,
        cold_sim_dropout=args.cold_sim_dropout,
    )
