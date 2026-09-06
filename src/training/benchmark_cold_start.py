"""Cold-Start (S1) Benchmarking Suite for AuditDDI.

Compares:
1. Baseline Model: Molecular Graph GNN (edge_aware_gat_v2).
2. Multimodal Model: Molecular Graph + ECFP + PharmGKB Genes + FAERS Toxicity (auditddi_multimodal_v1).

Tracks:
- Transductive Test AUROC / AUPRC / F1 / MCC
- S2 Semi-Inductive Test AUROC / AUPRC
- S1 Cold-Start (Unseen Drugs) AUROC / AUPRC
- Training Runtime per Epoch (seconds)
- Peak GPU/CPU Memory Footprint (MB)
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
    average_precision_score,
    brier_score_loss,
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
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_MULTIMODAL,
    PxDDIModel,
)


def evaluate_loader(
    model: PxDDIModel,
    loader: Any,
    device: torch.device,
    is_multimodal: bool = False,
) -> dict[str, float]:
    """Compute comprehensive performance metrics on an evaluation split."""
    model.eval()
    all_targets: list[float] = []
    all_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            da = batch['drug_a'].to(device)
            db = batch['drug_b'].to(device)
            lbls = batch['labels'].cpu().numpy()

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
            all_targets.extend(lbls.ravel().tolist())

    targets = np.array(all_targets)
    scores = np.array(all_scores)

    if len(np.unique(targets)) < 2:
        return {'auroc': 0.5, 'auprc': float(np.mean(targets)), 'f1': 0.0, 'mcc': 0.0, 'brier': 0.25}

    preds = (scores >= 0.5).astype(int)
    return {
        'auroc': float(roc_auc_score(targets, scores)),
        'auprc': float(average_precision_score(targets, scores)),
        'f1': float(f1_score(targets, preds, zero_division=0)),
        'mcc': matthews_corrcoef(targets, preds),
        'brier': float(brier_score_loss(targets, scores)),
    }


def train_benchmark_model(
    architecture_version: str,
    cache: MolecularCache,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_splits: dict[str, pd.DataFrame],
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train and evaluate one model architecture on the benchmark splits."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_multimodal = (architecture_version == MODEL_ARCHITECTURE_MULTIMODAL)
    print(f"\n--- Training: {architecture_version} (Multimodal={is_multimodal}) on {device} ---")

    train_loader = build_cached_multimodal_dataloader(train_df, cache, batch_size=batch_size, shuffle=True)
    val_loader = build_cached_multimodal_dataloader(val_df, cache, batch_size=batch_size, shuffle=False)

    sample_batch = next(iter(train_loader))
    in_channels = sample_batch['drug_a'].x.size(1)
    edge_dim = sample_batch['drug_a'].edge_attr.size(1)

    model = PxDDIModel(
        in_channels=in_channels,
        hidden_channels=64,
        edge_feature_dim=edge_dim,
        architecture_version=architecture_version,
        gene_feature_dim=cache.gene_dim,
        gene_hidden_channels=64,
        use_clinical_toxicity=is_multimodal,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    epoch_times = []
    start_time = time.perf_counter()

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
        ep_duration = time.perf_counter() - ep_start
        epoch_times.append(ep_duration)
        val_metrics = evaluate_loader(model, val_loader, device, is_multimodal=is_multimodal)
        print(f"  Epoch {epoch}/{epochs} ({ep_duration:.2f}s) - Loss: {total_loss/max(n_batches,1):.4f} - Val AUROC: {val_metrics['auroc']:.4f}")

    total_training_time = time.perf_counter() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0.0

    # Evaluate across all test splits
    results = {
        'architecture': architecture_version,
        'avg_epoch_time_sec': float(np.mean(epoch_times)),
        'total_training_time_sec': total_training_time,
        'peak_memory_mb': peak_memory_mb,
    }

    for split_name, split_df in test_splits.items():
        loader = build_cached_multimodal_dataloader(split_df, cache, batch_size=batch_size, shuffle=False)
        metrics = evaluate_loader(model, loader, device, is_multimodal=is_multimodal)
        for k, v in metrics.items():
            results[f'{split_name}_{k}'] = v

    return results


def run_benchmark(
    master_nodes_path: str | Path,
    splits_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Execute complete cold-start benchmark comparing baseline vs multimodal GNN."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = Path(splits_dir)

    print("=" * 80)
    print("STARTING AUDITDDI COLD-START BENCHMARK")
    print("=" * 80)

    # 1. Populate Cache
    cache = MolecularCache(gene_dim=50)
    cache.populate_from_master_nodes(master_nodes_path)

    # 2. Load Splits
    train_df = pd.read_csv(splits_path / 'transductive_train.csv')
    val_df = pd.read_csv(splits_path / 'validation.csv')

    test_splits = {
        'transductive': pd.read_csv(splits_path / 'transductive_test.csv'),
        's1_cold': pd.read_csv(splits_path / 's1_test.csv'),
        's2_semi': pd.read_csv(splits_path / 's2_test.csv'),
    }

    print(f"Loaded splits: Train={len(train_df):,}, Val={len(val_df):,}, "
          f"Transductive={len(test_splits['transductive']):,}, S1_Cold={len(test_splits['s1_cold']):,}, S2_Semi={len(test_splits['s2_semi']):,}")

    # 3. Train Baseline
    baseline_res = train_benchmark_model(
        architecture_version=MODEL_ARCHITECTURE_EDGE_AWARE,
        cache=cache,
        train_df=train_df,
        val_df=val_df,
        test_splits=test_splits,
        epochs=epochs,
        batch_size=batch_size,
    )

    # 4. Train Multi-Modal
    multimodal_res = train_benchmark_model(
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        cache=cache,
        train_df=train_df,
        val_df=val_df,
        test_splits=test_splits,
        epochs=epochs,
        batch_size=batch_size,
    )

    # 5. Export Summary
    comparison_df = pd.DataFrame([baseline_res, multimodal_res])
    csv_out = out_dir / 'benchmark_results.csv'
    json_out = out_dir / 'benchmark_summary.json'

    comparison_df.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps([baseline_res, multimodal_res], indent=2), encoding='utf-8')

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS COMPARISON:")
    print("=" * 80)
    cols = ['architecture', 's1_cold_auroc', 's1_cold_auprc', 's2_semi_auroc', 'transductive_auroc', 'avg_epoch_time_sec', 'peak_memory_mb']
    print(comparison_df[[c for c in cols if c in comparison_df.columns]].to_string(index=False))
    print(f"\nSaved benchmark outputs to: {out_dir}")
    return comparison_df
