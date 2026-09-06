"""Multi-Task Polypharmacy Side-Effect Prediction Module for AuditDDI.

Extends binary DDI prediction to multi-label prediction across TWOSIDES adverse event types
(e.g., hypotension, hyperkalemia, QT prolongation, gastrointestinal hemorrhage).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from src.data_prep.cached_graph_loader import MolecularCache
from src.models.ddi_model import MODEL_ARCHITECTURE_MULTIMODAL, PxDDIModel


def extract_top_side_effects(
    master_edges_path: str | Path,
    top_k: int = 100,
) -> list[str]:
    """Return the top-K most prevalent side-effect interaction types in TWOSIDES."""
    df_edges = pd.read_csv(master_edges_path, usecols=['interaction_type'], low_memory=False)
    counts = df_edges['interaction_type'].value_counts()
    return counts.head(top_k).index.tolist()


def prepare_multitask_pairs(
    master_edges_path: str | Path,
    top_side_effects: list[str],
    valid_drugs: set[str] | list[str] | None = None,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Aggregate pairwise TWOSIDES edges into multi-hot binary label vectors."""
    effect_to_idx = {name: idx for idx, name in enumerate(top_side_effects)}
    n_effects = len(top_side_effects)

    df_edges = pd.read_csv(
        master_edges_path,
        usecols=['drug_a_id', 'drug_b_id', 'interaction_type'],
        low_memory=False,
    )
    # Filter to top side effects
    filtered = df_edges[df_edges['interaction_type'].isin(effect_to_idx)].copy()

    # Filter out single ions or invalid molecules that could not form molecular graphs
    if valid_drugs is not None:
        valid_set = set(str(d).strip() for d in valid_drugs)
        filtered['drug_a_id'] = filtered['drug_a_id'].astype(str).str.strip()
        filtered['drug_b_id'] = filtered['drug_b_id'].astype(str).str.strip()
        filtered = filtered[
            filtered['drug_a_id'].isin(valid_set) & filtered['drug_b_id'].isin(valid_set)
        ].copy()

    # Canonical order for undirected pairs
    filtered['pair_a'] = filtered[['drug_a_id', 'drug_b_id']].min(axis=1)
    filtered['pair_b'] = filtered[['drug_a_id', 'drug_b_id']].max(axis=1)

    grouped = filtered.groupby(['pair_a', 'pair_b'])['interaction_type'].apply(list).reset_index()
    if max_pairs is not None and len(grouped) > max_pairs:
        grouped = grouped.sample(n=max_pairs, random_state=42).reset_index(drop=True)

    multitask_labels = []
    for side_effects in grouped['interaction_type']:
        vec = np.zeros(n_effects, dtype=np.float32)
        for se in side_effects:
            if se in effect_to_idx:
                vec[effect_to_idx[se]] = 1.0
        multitask_labels.append(vec.tolist())

    return pd.DataFrame({
        'drug_a_id': grouped['pair_a'],
        'drug_b_id': grouped['pair_b'],
        'multitask_labels': multitask_labels,
    })


class MultitaskCachedDataset(Dataset):
    """Dataset serving multi-modal cached drug graphs with multi-hot side-effect targets."""

    def __init__(self, pairs_df: pd.DataFrame, cache: MolecularCache):
        self.cache = cache
        valid_set = set(cache.graphs.keys())
        p_df = pairs_df.copy()
        p_df['drug_a_id'] = p_df['drug_a_id'].astype(str).str.strip()
        p_df['drug_b_id'] = p_df['drug_b_id'].astype(str).str.strip()
        self.pairs = p_df[
            p_df['drug_a_id'].isin(valid_set) & p_df['drug_b_id'].isin(valid_set)
        ].reset_index(drop=True)
        self._fallback_graph = next(iter(cache.graphs.values())) if len(cache.graphs) > 0 else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.pairs.iloc[idx]
        drug_a = str(row['drug_a_id']).strip()
        drug_b = str(row['drug_b_id']).strip()
        labels = torch.tensor(row['multitask_labels'], dtype=torch.float32)

        g_a = self.cache.graphs.get(drug_a, self._fallback_graph)
        g_b = self.cache.graphs.get(drug_b, self._fallback_graph)

        return {
            'drug_a': g_a,
            'drug_b': g_b,
            'fp_a': self.cache.fingerprints.get(drug_a, torch.zeros(1024)),
            'fp_b': self.cache.fingerprints.get(drug_b, torch.zeros(1024)),
            'gene_a': self.cache.gene_vectors.get(drug_a, torch.zeros(self.cache.gene_dim)),
            'gene_b': self.cache.gene_vectors.get(drug_b, torch.zeros(self.cache.gene_dim)),
            'gene_mask_a': self.cache.gene_masks.get(drug_a, torch.tensor(0.0)),
            'gene_mask_b': self.cache.gene_masks.get(drug_b, torch.tensor(0.0)),
            'tox_a': self.cache.toxicity_scalars.get(drug_a, torch.tensor(0.0)),
            'tox_b': self.cache.toxicity_scalars.get(drug_b, torch.tensor(0.0)),
            'labels': labels,
        }


def multitask_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'drug_a': Batch.from_data_list([item['drug_a'] for item in batch]),
        'drug_b': Batch.from_data_list([item['drug_b'] for item in batch]),
        'fp_a': torch.stack([item['fp_a'] for item in batch]),
        'fp_b': torch.stack([item['fp_b'] for item in batch]),
        'gene_a': torch.stack([item['gene_a'] for item in batch]),
        'gene_b': torch.stack([item['gene_b'] for item in batch]),
        'gene_mask_a': torch.stack([item['gene_mask_a'] for item in batch]),
        'gene_mask_b': torch.stack([item['gene_mask_b'] for item in batch]),
        'tox_a': torch.stack([item['tox_a'] for item in batch]),
        'tox_b': torch.stack([item['tox_b'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }


def evaluate_multitask_loader(
    model: PxDDIModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Calculate Micro and Macro AUROC/AUPRC across all adverse event tasks."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            da = batch['drug_a'].to(device)
            db = batch['drug_b'].to(device)
            logits, _, _ = model(
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
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(batch['labels'].cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # Micro AUROC and AUPRC (flattened across all tasks)
    micro_auroc = roc_auc_score(targets.ravel(), preds.ravel()) if len(np.unique(targets)) > 1 else 0.5
    micro_auprc = average_precision_score(targets.ravel(), preds.ravel())

    # Macro AUROC and AUPRC (averaged across evaluable tasks)
    task_aurocs = []
    task_auprcs = []
    for k in range(targets.shape[1]):
        y_t = targets[:, k]
        y_p = preds[:, k]
        if len(np.unique(y_t)) > 1:
            task_aurocs.append(roc_auc_score(y_t, y_p))
            task_auprcs.append(average_precision_score(y_t, y_p))

    macro_auroc = float(np.mean(task_aurocs)) if task_aurocs else 0.5
    macro_auprc = float(np.mean(task_auprcs)) if task_auprcs else 0.0

    return {
        'micro_auroc': float(micro_auroc),
        'micro_auprc': float(micro_auprc),
        'macro_auroc': macro_auroc,
        'macro_auprc': macro_auprc,
        'evaluable_tasks': len(task_aurocs),
        'total_tasks': targets.shape[1],
    }


def run_multitask_side_effect_study(
    master_nodes_path: str | Path,
    master_edges_path: str | Path,
    output_dir: str | Path,
    top_k_side_effects: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    device: torch.device | None = None,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Execute end-to-end multi-task adverse event prediction on TWOSIDES."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print(f'STARTING MULTI-TASK POLYPHARMACY SIDE-EFFECT STUDY (Top {top_k_side_effects} Adverse Events)')
    print('=' * 80)

    # 1. Extract Top Side-Effect Vocabulary
    top_effects = extract_top_side_effects(master_edges_path, top_k=top_k_side_effects)
    print(f'Selected Top {len(top_effects)} Side Effects: {top_effects[:5]}...')
    (out_p / 'side_effects_vocabulary.json').write_text(json.dumps(top_effects, indent=2), encoding='utf-8')

    # 2. Populate Cache
    cache = MolecularCache(gene_dim=50)
    cache.populate_from_master_nodes(master_nodes_path)

    # 3. Build Multi-Task Pairs (excluding inorganic single ions without graphs)
    valid_drugs = set(cache.graphs.keys())
    pairs_df = prepare_multitask_pairs(
        master_edges_path,
        top_effects,
        valid_drugs=valid_drugs,
        max_pairs=max_pairs,
    )
    print(f'Built Multi-Task Dataset: {len(pairs_df)} unique drug pairs across {len(top_effects)} targets.')

    # Train/Validation/Test split (80/10/10)
    n = len(pairs_df)
    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:int(0.8 * n)]
    val_idx = indices[int(0.8 * n):int(0.9 * n)]
    test_idx = indices[int(0.9 * n):]

    train_ds = MultitaskCachedDataset(pairs_df.iloc[train_idx], cache)
    val_ds = MultitaskCachedDataset(pairs_df.iloc[val_idx], cache)
    test_ds = MultitaskCachedDataset(pairs_df.iloc[test_idx], cache)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=multitask_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=multitask_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=multitask_collate_fn)

    sample_batch = next(iter(train_loader))
    in_channels = sample_batch['drug_a'].x.size(1)
    edge_dim = sample_batch['drug_a'].edge_attr.size(1)

    # 4. Multi-Task Model Instantiation
    model = PxDDIModel(
        in_channels=in_channels,
        hidden_channels=64,
        edge_feature_dim=edge_dim,
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        gene_feature_dim=cache.gene_dim,
        gene_hidden_channels=64,
        use_clinical_toxicity=True,
        num_side_effects=len(top_effects),
        use_cross_modal_attention=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            da = batch['drug_a'].to(device)
            db = batch['drug_b'].to(device)
            targets = batch['labels'].to(device)

            logits, _, _ = model(
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
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()
        val_metrics = evaluate_multitask_loader(model, val_loader, device)
        elapsed = time.perf_counter() - t0

        print(f"Epoch {ep:2d}/{epochs:2d} ({elapsed:4.1f}s) | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Micro AUROC: {val_metrics['micro_auroc']:.4f} | Macro AUROC: {val_metrics['macro_auroc']:.4f}")

        history.append({
            'epoch': ep,
            'train_loss': total_loss / len(train_loader),
            'val_micro_auroc': val_metrics['micro_auroc'],
            'val_macro_auroc': val_metrics['macro_auroc'],
            'val_micro_auprc': val_metrics['micro_auprc'],
            'val_macro_auprc': val_metrics['macro_auprc'],
        })

        if val_metrics['micro_auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['micro_auroc']
            torch.save(model.state_dict(), out_p / 'auditddi_multitask_best.pt')

    # 5. Final Test Evaluation
    if (out_p / 'auditddi_multitask_best.pt').is_file():
        model.load_state_dict(torch.load(out_p / 'auditddi_multitask_best.pt', map_location=device))

    test_metrics = evaluate_multitask_loader(model, test_loader, device)
    print('=' * 80)
    print('FINAL MULTI-TASK TEST RESULTS:')
    print(f"  Test Micro AUROC: {test_metrics['micro_auroc']:.4f}")
    print(f"  Test Macro AUROC: {test_metrics['macro_auroc']:.4f}")
    print(f"  Test Micro AUPRC: {test_metrics['micro_auprc']:.4f}")
    print(f"  Test Macro AUPRC: {test_metrics['macro_auprc']:.4f}")
    print('=' * 80)

    summary_results = {
        'top_k_side_effects': len(top_effects),
        'test_metrics': test_metrics,
        'history': history,
        'model_path': str(out_p / 'auditddi_multitask_best.pt'),
    }
    (out_p / 'multitask_results.json').write_text(json.dumps(summary_results, indent=2), encoding='utf-8')
    pd.DataFrame(history).to_csv(out_p / 'multitask_training_history.csv', index=False)

    return summary_results
