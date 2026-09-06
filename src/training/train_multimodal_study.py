"""Unified Multimodal Study Suite for AuditDDI.

Includes:
1. Extended Training with Validation Checkpointing and Convergence Curve Tracking.
2. Modality Ablation Studies (Molecular Only vs +Genes vs +FAERS vs Full Multimodal).
3. S1 Cold-Start Error Analysis stratified by PharmGKB/FAERS Coverage Tiers.
4. Model Calibration Analysis (ECE, Brier score, Platt scaling).
5. Production Checkpoint Export.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
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
from src.models.calibration import (
    apply_calibrator,
    expected_calibration_error,
    fit_platt_calibrator,
)
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_ABLATION_FAERS,
    MODEL_ARCHITECTURE_ABLATION_GENES,
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_MULTIMODAL,
    PxDDIModel,
)
from src.training.benchmark_cold_start import (
    ensure_benchmark_splits,
    evaluate_loader,
)


def predict_loader(
    model: PxDDIModel,
    loader: Any,
    device: torch.device,
    is_multimodal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw probabilities and ground truth targets from a DataLoader."""
    model.eval()
    all_scores: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for batch in loader:
            da = batch['drug_a'].to(device)
            db = batch['drug_b'].to(device)
            lbls = batch['labels'].cpu().numpy().ravel()

            if is_multimodal:
                risk_logits, _, _ = model(
                    drug_a=da,
                    drug_b=db,
                    fp_a=batch['fp_a'].to(device),
                    fp_b=batch['fp_b'].to(device),
                    gene_a=batch['gene_a'].to(device),
                    gene_b=batch['gene_b'].to(device),
                    gene_mask_a=batch['gene_mask_a'].to(device),
                    gene_mask_b=batch['gene_mask_b'].to(device),
                    clinical_tox_a=batch['tox_a'].to(device),
                    clinical_tox_b=batch['tox_b'].to(device),
                )
            else:
                risk_logits, _, _ = model(drug_a=da, drug_b=db)

            probs = torch.sigmoid(risk_logits).cpu().numpy().ravel()
            all_scores.extend(probs.tolist())
            all_targets.extend(lbls.tolist())

    return np.array(all_scores, dtype=float), np.array(all_targets, dtype=float)


def train_extended_multimodal(
    cache: MolecularCache,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-5,
    architecture_version: str = MODEL_ARCHITECTURE_MULTIMODAL,
    device: torch.device | None = None,
    chembl_pretrained_path: str | Path | None = None,
    use_cross_modal_attention: bool = True,
) -> tuple[PxDDIModel, pd.DataFrame, dict[str, Any]]:
    """Train the multimodal model across extended epochs with checkpointing."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    is_multimodal = (architecture_version != MODEL_ARCHITECTURE_EDGE_AWARE)

    train_loader = build_cached_multimodal_dataloader(train_df, cache, batch_size=batch_size, shuffle=True)
    val_loader = build_cached_multimodal_dataloader(val_df, cache, batch_size=batch_size, shuffle=False)

    test_loaders = {
        name: build_cached_multimodal_dataloader(df, cache, batch_size=batch_size, shuffle=False)
        for name, df in test_splits.items()
    }

    sample_batch = next(iter(train_loader))
    in_channels = sample_batch['drug_a'].x.size(1)
    edge_dim = sample_batch['drug_a'].edge_attr.size(1)

    hidden_dim = 64
    if chembl_pretrained_path and Path(chembl_pretrained_path).is_file():
        try:
            bundle_meta = torch.load(chembl_pretrained_path, map_location='cpu', weights_only=False)
            if isinstance(bundle_meta, dict) and 'encoder_configuration' in bundle_meta:
                hidden_dim = int(bundle_meta['encoder_configuration'].get('hidden_channels', 64))
        except Exception:
            pass

    model = PxDDIModel(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        edge_feature_dim=edge_dim,
        architecture_version=architecture_version,
        gene_feature_dim=cache.gene_dim,
        gene_hidden_channels=64,
        use_clinical_toxicity=is_multimodal,
        use_cross_modal_attention=use_cross_modal_attention if is_multimodal else False,
    )

    if chembl_pretrained_path and Path(chembl_pretrained_path).is_file():
        from src.models.encoder_pretraining import load_pretrained_edge_aware_encoder
        print(f"Loading ChEMBL pre-trained encoder weights from: {chembl_pretrained_path} (hidden_dim={hidden_dim})")
        try:
            load_pretrained_edge_aware_encoder(
                encoder=model.encoder,
                path=chembl_pretrained_path,
                expected_in_channels=in_channels,
                expected_edge_feature_dim=edge_dim,
                expected_hidden_channels=hidden_dim,
                map_location=device,
            )
            print("Successfully initialized molecular encoder with ChEMBL representations.")
        except Exception as exc:
            print(f"Warning: could not load ChEMBL weights ({exc}), proceeding with random initialization.")

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    history_records: list[dict[str, Any]] = []
    best_val_auroc = -1.0
    best_weights_path = out_p / f'{architecture_version}_best.pt'

    print(f"\n{'=' * 80}")
    print(f"STARTING EXTENDED TRAINING: {architecture_version} ({epochs} epochs on {device})")
    print(f"{'=' * 80}")

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            da = batch['drug_a'].to(device)
            db = batch['drug_b'].to(device)
            y = batch['labels'].to(device)

            if is_multimodal:
                risk_logits, _, _ = model(
                    drug_a=da,
                    drug_b=db,
                    fp_a=batch['fp_a'].to(device),
                    fp_b=batch['fp_b'].to(device),
                    gene_a=batch['gene_a'].to(device),
                    gene_b=batch['gene_b'].to(device),
                    gene_mask_a=batch['gene_mask_a'].to(device),
                    gene_mask_b=batch['gene_mask_b'].to(device),
                    clinical_tox_a=batch['tox_a'].to(device),
                    clinical_tox_b=batch['tox_b'].to(device),
                )
            else:
                risk_logits, _, _ = model(drug_a=da, drug_b=db)

            loss = criterion(risk_logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        ep_sec = time.perf_counter() - ep_start
        avg_loss = total_loss / max(n_batches, 1)

        val_metrics = evaluate_loader(model, val_loader, device, is_multimodal=is_multimodal)
        s1_metrics = evaluate_loader(model, test_loaders['s1_cold'], device, is_multimodal=is_multimodal)

        record = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'epoch_sec': ep_sec,
            'val_auroc': val_metrics['auroc'],
            'val_auprc': val_metrics['auprc'],
            's1_cold_auroc': s1_metrics['auroc'],
            's1_cold_auprc': s1_metrics['auprc'],
        }
        history_records.append(record)

        is_best = val_metrics['auroc'] > best_val_auroc
        if is_best:
            best_val_auroc = val_metrics['auroc']
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auroc': best_val_auroc,
                    'in_channels': in_channels,
                    'hidden_channels': 64,
                    'edge_feature_dim': edge_dim,
                    'architecture_version': architecture_version,
                    'gene_feature_dim': cache.gene_dim,
                    'gene_hidden_channels': 64,
                    'use_clinical_toxicity': is_multimodal,
                },
                best_weights_path,
            )

        best_mark = " (★ Best)" if is_best else ""
        print(f"  Epoch {epoch:02d}/{epochs:02d} ({ep_sec:.1f}s) - Loss: {avg_loss:.4f} | "
              f"Val AUROC: {val_metrics['auroc']:.4f} | S1 AUROC: {s1_metrics['auroc']:.4f}{best_mark}")

    # Load best checkpoint for final evaluation
    if best_weights_path.is_file():
        ckpt = torch.load(best_weights_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\nLoaded best model from epoch {ckpt['epoch']} (Val AUROC: {ckpt['val_auroc']:.4f})")

    history_df = pd.DataFrame(history_records)
    history_df.to_csv(out_p / f'{architecture_version}_training_history.csv', index=False)

    # Final split evaluation
    final_results: dict[str, Any] = {'architecture': architecture_version, 'best_val_auroc': best_val_auroc}
    for name, loader in test_loaders.items():
        m = evaluate_loader(model, loader, device, is_multimodal=is_multimodal)
        for k, v in m.items():
            final_results[f'{name}_{k}'] = v

    return model, history_df, final_results


def run_modality_ablation_study(
    cache: MolecularCache,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Systematically run all 4 modality ablation variants and report deltas."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    ablation_variants = [
        ('Molecular Only (Baseline)', MODEL_ARCHITECTURE_EDGE_AWARE),
        ('Molecular + PharmGKB Genes', MODEL_ARCHITECTURE_ABLATION_GENES),
        ('Molecular + FAERS Toxicity', MODEL_ARCHITECTURE_ABLATION_FAERS),
        ('Full Multimodal (AuditDDI)', MODEL_ARCHITECTURE_MULTIMODAL),
    ]

    all_ablation_results: list[dict[str, Any]] = []

    print(f"\n{'=' * 80}")
    print("STARTING SYSTEMATIC MODALITY ABLATION STUDY")
    print(f"{'=' * 80}")

    for display_name, arch in ablation_variants:
        print(f"\n--> Training Variant: {display_name} ({arch})...")
        _, _, results = train_extended_multimodal(
            cache=cache,
            train_df=train_df,
            val_df=val_df,
            test_splits=test_splits,
            output_dir=out_p / 'ablation_checkpoints',
            epochs=epochs,
            batch_size=batch_size,
            architecture_version=arch,
            device=device,
        )
        results['variant_name'] = display_name
        all_ablation_results.append(results)

    ablation_df = pd.DataFrame(all_ablation_results)

    # Compute deltas relative to baseline
    baseline_s1 = ablation_df.loc[ablation_df['architecture'] == MODEL_ARCHITECTURE_EDGE_AWARE, 's1_cold_auroc'].values[0]
    baseline_trans = ablation_df.loc[ablation_df['architecture'] == MODEL_ARCHITECTURE_EDGE_AWARE, 'transductive_auroc'].values[0]

    ablation_df['delta_s1_auroc'] = ablation_df['s1_cold_auroc'] - baseline_s1
    ablation_df['delta_transductive_auroc'] = ablation_df['transductive_auroc'] - baseline_trans

    csv_path = out_p / 'ablation_study_results.csv'
    ablation_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 80}")
    print("ABLATION STUDY SUMMARY (QUANTIFIED MODALITY CONTRIBUTIONS):")
    print(f"{'=' * 80}")
    cols = ['variant_name', 's1_cold_auroc', 'delta_s1_auroc', 'transductive_auroc', 'delta_transductive_auroc']
    print(ablation_df[[c for c in cols if c in ablation_df.columns]].to_string(index=False))
    return ablation_df


def analyze_cold_start_coverage_errors(
    model: PxDDIModel,
    cache: MolecularCache,
    s1_test_df: pd.DataFrame,
    output_dir: str | Path,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect misclassifications on unseen cold-start pairs stratified by external coverage."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if s1_test_df.empty:
        empty_annotated = pd.DataFrame(columns=pd.Index([
            'drug_a', 'drug_b', 'true_label', 'pred_prob', 'binary_pred',
            'is_correct', 'error_type', 'coverage_tier', 'gene_a', 'gene_b', 'faers_a', 'faers_b'
        ]))
        empty_summary = pd.DataFrame(columns=pd.Index([
            'coverage_tier', 'pair_count', 'accuracy', 'auroc', 'fpr', 'fnr', 'false_positives', 'false_negatives'
        ]))
        empty_annotated.to_csv(out_p / 'cold_start_error_analysis.csv', index=False)
        empty_summary.to_csv(out_p / 'cold_start_coverage_summary.csv', index=False)
        return empty_annotated, empty_summary

    loader = build_cached_multimodal_dataloader(s1_test_df, cache, batch_size=64, shuffle=False)
    scores, targets = predict_loader(model, loader, device, is_multimodal=True)
    preds = (scores >= 0.5).astype(int)

    # Annotate coverage tier per pair
    # Resolve columns
    src_col = 'drug_a_id' if 'drug_a_id' in s1_test_df.columns else 'source'
    dst_col = 'drug_b_id' if 'drug_b_id' in s1_test_df.columns else 'target'

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(s1_test_df.itertuples(index=False)):
        sa = getattr(row, src_col)
        sb = getattr(row, dst_col)
        lbl = float(targets[idx])
        prob = float(scores[idx])
        pred = int(preds[idx])

        has_gene_a = bool(cache.gene_masks.get(sa, torch.tensor(0.0)).item() > 0.5)
        has_gene_b = bool(cache.gene_masks.get(sb, torch.tensor(0.0)).item() > 0.5)
        has_tox_a = bool(cache.toxicity_masks.get(sa, torch.tensor(0.0)).item() > 0.5)
        has_tox_b = bool(cache.toxicity_masks.get(sb, torch.tensor(0.0)).item() > 0.5)

        has_any_ext_a = has_gene_a or has_tox_a
        has_any_ext_b = has_gene_b or has_tox_b

        if has_any_ext_a and has_any_ext_b:
            tier = 'Both Drugs Profiled'
        elif has_any_ext_a or has_any_ext_b:
            tier = 'One Drug Profiled'
        else:
            tier = 'Zero External Coverage'

        err_type = 'Correct'
        if pred == 1 and lbl == 0:
            err_type = 'False Positive'
        elif pred == 0 and lbl == 1:
            err_type = 'False Negative'

        rows.append({
            'drug_a': sa,
            'drug_b': sb,
            'true_label': lbl,
            'pred_prob': prob,
            'binary_pred': pred,
            'is_correct': (pred == lbl),
            'error_type': err_type,
            'coverage_tier': tier,
            'gene_a': has_gene_a,
            'gene_b': has_gene_b,
            'faers_a': has_tox_a,
            'faers_b': has_tox_b,
        })

    annotated_df = pd.DataFrame(rows)
    annotated_df.to_csv(out_p / 'cold_start_error_analysis.csv', index=False)

    # Compute stratified summary metrics per coverage tier
    tier_summary: list[dict[str, Any]] = []
    for tier_name, group in annotated_df.groupby('coverage_tier'):
        y_true = group['true_label'].to_numpy()
        y_prob = group['pred_prob'].to_numpy()
        y_pred = group['binary_pred'].to_numpy()

        acc = accuracy_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

        # False positive rate and false negative rate
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)

        tier_summary.append({
            'coverage_tier': tier_name,
            'pair_count': len(group),
            'accuracy': float(acc),
            'auroc': float(auroc),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        })

    summary_df = pd.DataFrame(tier_summary)
    summary_df.to_csv(out_p / 'cold_start_coverage_summary.csv', index=False)

    print(f"\n{'=' * 80}")
    print("S1 COLD-START PERFORMANCE STRATIFIED BY EXTERNAL COVERAGE TIER:")
    print(f"{'=' * 80}")
    print(summary_df.to_string(index=False))
    return annotated_df, summary_df


def evaluate_multimodal_calibration(
    model: PxDDIModel,
    cache: MolecularCache,
    val_df: pd.DataFrame,
    test_splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Calculate ECE, Brier score, and Platt scaling calibration metrics."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Validation predictions for fitting Platt calibrator
    val_loader = build_cached_multimodal_dataloader(val_df, cache, batch_size=64, shuffle=False)
    val_scores, val_targets = predict_loader(model, val_loader, device, is_multimodal=True)

    calibrator = fit_platt_calibrator(val_targets, val_scores)

    calibration_report: dict[str, Any] = {}

    for split_name, df in test_splits.items():
        loader = build_cached_multimodal_dataloader(df, cache, batch_size=64, shuffle=False)
        scores, targets = predict_loader(model, loader, device, is_multimodal=True)

        if len(scores) == 0 or len(targets) == 0:
            calibration_report[split_name] = {
                'n_pairs': 0,
                'raw_ece': 0.0,
                'raw_brier_score': 0.0,
                'calibrated_ece': 0.0,
                'calibrated_brier_score': 0.0,
                'ece_reduction_pct': 0.0,
            }
            continue

        # Raw calibration
        raw_ece = expected_calibration_error(targets, scores, bins=10) or 0.0
        raw_brier = float(brier_score_loss(targets, scores))

        # Calibrated predictions
        cal_scores = apply_calibrator(scores, calibrator)
        cal_ece = expected_calibration_error(targets, cal_scores, bins=10) or 0.0
        cal_brier = float(brier_score_loss(targets, cal_scores))

        calibration_report[split_name] = {
            'n_pairs': len(scores),
            'raw_ece': raw_ece,
            'raw_brier_score': raw_brier,
            'calibrated_ece': cal_ece,
            'calibrated_brier_score': cal_brier,
            'ece_reduction_pct': ((raw_ece - cal_ece) / raw_ece * 100.0) if raw_ece else 0.0,
        }

    out_json = out_p / 'calibration_metrics.json'
    out_json.write_text(json.dumps(calibration_report, indent=2), encoding='utf-8')

    print(f"\n{'=' * 80}")
    print("MODEL CALIBRATION RESULTS (ECE & BRIER SCORE):")
    print(f"{'=' * 80}")
    for split_name, d in calibration_report.items():
        print(f"  [{split_name.upper():<14}] Raw ECE: {d['raw_ece']:.4f} -> Calibrated: {d['calibrated_ece']:.4f} "
              f"({d['ece_reduction_pct']:+.1f}% error reduction) | Brier: {d['calibrated_brier_score']:.4f}")

    return calibration_report


def run_full_multimodal_study(
    master_nodes_path: str | Path,
    splits_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    master_edges_path: str | Path | None = None,
    extended_epochs: int = 15,
    ablation_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    device: torch.device | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute complete extended training, ablation study, error analysis, and calibration."""
    # Support flexible parameter aliases
    if 'epochs' in kwargs:
        extended_epochs = int(kwargs.pop('epochs'))
    if 'lr' in kwargs:
        learning_rate = float(kwargs.pop('lr'))
    run_ablation: bool = kwargs.pop('run_ablation', True)
    run_error_analysis: bool = kwargs.pop('run_error_analysis', True)
    calibrate: bool = kwargs.pop('calibrate', True)

    if output_dir is None:
        out_p = Path(master_nodes_path).resolve().parent.parent / 'multimodal_study_results'
    else:
        out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    splits_p = ensure_benchmark_splits(
        splits_dir=splits_dir,
        master_nodes_path=master_nodes_path,
        master_edges_path=master_edges_path,
        **kwargs,
    )

    print("=" * 80)
    print("STARTING AUDITDDI MULTIMODAL COMPREHENSIVE STUDY")
    print(f"Master Nodes : {master_nodes_path}")
    print(f"Splits Dir   : {splits_p}")
    print(f"Output Dir   : {out_p}")
    print("=" * 80)

    # 1. Populate Cache
    cache = MolecularCache(gene_dim=50)
    cache.populate_from_master_nodes(master_nodes_path)

    # 2. Load Splits
    train_df = pd.read_csv(splits_p / 'transductive_train.csv')
    val_df = pd.read_csv(splits_p / 'validation.csv')
    test_splits = {
        'transductive': pd.read_csv(splits_p / 'transductive_test.csv'),
        's1_cold': pd.read_csv(splits_p / 's1_test.csv'),
        's2_semi': pd.read_csv(splits_p / 's2_test.csv'),
    }

    chembl_pretrained_path: str | Path | None = kwargs.pop('chembl_pretrained_path', None)
    use_cross_modal_attention: bool = kwargs.pop('use_cross_modal_attention', True)

    # 3. Extended Training (Full Multimodal Model)
    best_model, history_df, extended_metrics = train_extended_multimodal(
        cache=cache,
        train_df=train_df,
        val_df=val_df,
        test_splits=test_splits,
        output_dir=out_p,
        epochs=extended_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        chembl_pretrained_path=chembl_pretrained_path,
        use_cross_modal_attention=use_cross_modal_attention,
    )

    # 4. Modality Ablation Study
    ablation_dict: list[dict[str, Any]] = []
    if run_ablation:
        ablation_df = run_modality_ablation_study(
            cache=cache,
            train_df=train_df,
            val_df=val_df,
            test_splits=test_splits,
            output_dir=out_p / 'ablation',
            epochs=ablation_epochs,
            batch_size=batch_size,
            device=device,
        )
        ablation_dict = ablation_df.to_dict(orient='records')

    # 5. Cold-Start Error Analysis Stratified by External Coverage
    tier_dict: list[dict[str, Any]] = []
    if run_error_analysis:
        err_df, tier_summary_df = analyze_cold_start_coverage_errors(
            model=best_model,
            cache=cache,
            s1_test_df=test_splits['s1_cold'],
            output_dir=out_p / 'error_analysis',
            device=device,
        )
        tier_dict = tier_summary_df.to_dict(orient='records')

    # 6. Model Calibration (ECE & Reliability)
    calibration_report: dict[str, Any] = {}
    if calibrate:
        calibration_report = evaluate_multimodal_calibration(
            model=best_model,
            cache=cache,
            val_df=val_df,
            test_splits=test_splits,
            output_dir=out_p / 'calibration',
            device=device,
        )

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MULTIMODAL STUDY COMPLETE!")
    print(f"All models, ablation reports, and error analysis saved to: {out_p}")
    print("=" * 80)

    return {
        'extended_metrics': extended_metrics,
        'transductive_and_cold_metrics': extended_metrics,
        'ablation': ablation_dict,
        'ablation_results': ablation_dict,
        'tier_summary': tier_dict,
        'calibration_report': calibration_report,
    }


# Convenience alias matching external call conventions
run_full_study = run_full_multimodal_study

