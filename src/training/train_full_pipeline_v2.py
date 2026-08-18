"""
train_full_pipeline_v2.py — Final consolidated pipeline
"""

import torch
import time
import os
import sys
import json
import hashlib
import random
import numpy as np
from datetime import datetime, timezone

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DRIVE_BASE = os.environ.get('PXDDI_DATA_BASE', '/content/drive/MyDrive/pxddi-data/')

sys.path.append(os.path.join(DRIVE_BASE, 'pxddi/src'))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, matthews_corrcoef
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader

from models.ddi_model import PxDDIModel
from data_prep.prepare_twosides import smiles_to_graph, NUM_ATOM_FEATURES
from data_prep.splits import build_binary_pair_dataset, create_splits
from data_prep.pubchem_bridge import canonicalize, resolve_toxicity_bridge

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {DEVICE}")
if DEVICE.type != 'cuda':
    print("WARNING: Not using CUDA!")

scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

TWOSIDES_EDGES = DRIVE_BASE + 'twosides/drug_drug_edges.csv'
TOXICITY_BRIDGE = DRIVE_BASE + 'checkpoints/toxicity_smiles_bridge.csv'
USE_CHEMBERTA = False
CHECKPOINT_PATH = DRIVE_BASE + ('checkpoints/pxddi_model_chemberta.pt' if USE_CHEMBERTA else 'checkpoints/pxddi_model.pt')
RUN_ARTIFACTS_DIR = DRIVE_BASE + 'artifacts/run_' + datetime.now().strftime("%Y%m%d_%H%M%S")

DATA_CAP = 200000
EPOCHS = 200
HIDDEN_CHANNELS = 128

history = {'epoch': [], 'loss': [], 'auroc': []}

def save_checkpoint_safe(state_dict_bundle, path):
    tmp_path = path + '.tmp'
    torch.save(state_dict_bundle, tmp_path)
    os.replace(tmp_path, path)  # atomic — no corrupted partial file on interrupt
    with open(path, 'rb') as f:
        checkpoint_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    print(f"Checkpoint saved. SHA256 (short): {checkpoint_hash}")
    return checkpoint_hash

def get_preds_labels(model, loader):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for da, db, _, _, _, _, rl in loader:
            da, db = da.to(DEVICE), db.to(DEVICE)
            rp, _, _ = model(da, db)
            preds.extend(torch.sigmoid(rp).cpu().numpy())
            labels.extend(rl.numpy())
    return np.array(labels), np.array(preds)

def plot_roc_pr_confusion(name, labels, preds, threshold, plots_dir):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(rec, prec)
    pred_labels = (preds >= threshold).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    mcc = matthews_corrcoef(labels, pred_labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(fpr, tpr, label=f'AUC={roc_auc:.3f}'); axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[0].set_title(f'{name} — ROC (n={len(labels)})'); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].legend()

    axes[1].plot(rec, prec, label=f'PR-AUC={pr_auc:.3f}')
    axes[1].set_title(f'{name} — Precision-Recall'); axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].legend()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['No Interaction','Interaction'], yticklabels=['No Interaction','Interaction'])
    axes[2].set_title(f'{name} — Confusion Matrix (threshold={threshold:.3f}, MCC={mcc:.3f})')

    plt.tight_layout()
    fname = plots_dir + f'{name.lower().replace(" ", "_")}_full_eval.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    return {'auroc': roc_auc, 'pr_auc': pr_auc, 'mcc': mcc, 'confusion_matrix': cm.tolist()}

def plot_toxicity_bridge_coverage(total, matched, unique, conflicts, plots_dir):
    stages = ['Source rows', 'PubChem matched', 'Unique structures', 'Conflicts resolved']
    values = [total, matched, unique, conflicts]
    plt.figure(figsize=(7,5))
    plt.bar(stages, values, color=['#4C72B0','#55A868','#C44E52','#8172B2'])
    plt.title('Toxicity Bridge Coverage Funnel')
    plt.ylabel('Count')
    for i, v in enumerate(values): plt.text(i, v+2, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(plots_dir + 'toxicity_bridge_coverage.png', dpi=150)
    print(f"Saved {plots_dir}toxicity_bridge_coverage.png")

def get_toxicity(smiles, lookup, default=None):
    canon = canonicalize(smiles)
    return lookup.get(canon, default)

def load_toxicity_lookup():
    bridge = pd.read_csv(TOXICITY_BRIDGE)
    resolved, summary, conflicts = resolve_toxicity_bridge(bridge)
    
    audit_dir = os.path.join(RUN_ARTIFACTS_DIR, 'audits')
    os.makedirs(audit_dir, exist_ok=True)
    conflicts.to_csv(os.path.join(audit_dir, 'toxicity_bridge_conflicts.csv'), index=False)
    with open(os.path.join(audit_dir, 'toxicity_bridge_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    global toxicity_summary
    toxicity_summary = summary
    return dict(zip(resolved['canonical_smiles'], resolved['toxicity_score']))

def remove_reversed_duplicates(df, source_col='source', target_col='target'):
    """Prevents leakage from A-B and B-A both appearing (possibly in
    different splits)."""
    df = df.copy()
    before = len(df)
    
    # Vectorized sorting of pairs is vastly faster than apply() on millions of rows
    min_col = np.minimum(df[source_col].astype(str), df[target_col].astype(str))
    max_col = np.maximum(df[source_col].astype(str), df[target_col].astype(str))
    
    df = df.assign(__min=min_col, __max=max_col)
    df = df.drop_duplicates(subset=['__min', '__max'])
    df = df.drop(columns=['__min', '__max'])
    
    print(f"Removed {before - len(df)} reversed/duplicate pairs")
    return df

def add_negative_samples(df, source_col='source', target_col='target', neg_ratio=1.0, seed=42):
    combined = build_binary_pair_dataset(
        df,
        source_col=source_col,
        target_col=target_col,
        neg_ratio=neg_ratio,
        seed=seed,
    )
    positives = int((combined['label'] == 1).sum())
    negatives = int((combined['label'] == 0).sum())
    print(f'Final dataset: {len(combined)} unique unordered pairs ({positives} pos, {negatives} neg)')
    return combined

class PxDDIDataset(Dataset):
    def __init__(self, df, tox_lookup, source_col='source', target_col='target', label_col='label'):
        super().__init__()
        records = []
        skipped = 0
        total = len(df)
        for i, row in enumerate(df.itertuples(), 1):
            source, target, label = getattr(row, source_col), getattr(row, target_col), getattr(row, label_col)
            ga = smiles_to_graph(source)
            gb = smiles_to_graph(target)
            if ga is None or gb is None:
                skipped += 1
                continue
            tox_a = get_toxicity(source, tox_lookup)
            tox_b = get_toxicity(target, tox_lookup)
            tox_a_known = float(tox_a is not None)
            tox_b_known = float(tox_b is not None)
            tox_a = tox_a if tox_a is not None else 0.0
            tox_b = tox_b if tox_b is not None else 0.0
            records.append((ga, gb, tox_a, tox_b, tox_a_known, tox_b_known, label))
            if i % 5000 == 0:
                print(f"    ...processed {i}/{total} rows")
        print(f"Built dataset: {len(records)} valid pairs, {skipped} skipped")
        self.records = records

    def len(self): return len(self.records)
    def get(self, idx): return self.records[idx]

def collate_fn(batch):
    ga = [b[0] for b in batch]
    gb = [b[1] for b in batch]
    ta = torch.tensor([b[2] for b in batch], dtype=torch.float)
    tb = torch.tensor([b[3] for b in batch], dtype=torch.float)
    tak = torch.tensor([b[4] for b in batch], dtype=torch.float)
    tbk = torch.tensor([b[5] for b in batch], dtype=torch.float)
    lb = torch.tensor([b[6] for b in batch], dtype=torch.float)
    return Batch.from_data_list(ga), Batch.from_data_list(gb), ta, tb, tak, tbk, lb

def build_loader(df, tox_lookup, batch_size=32, shuffle=True):
    ds = PxDDIDataset(df, tox_lookup)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def multi_task_loss(rp, tap, tbp, rl, tal, tbl, tak, tbk):
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    ddi_loss = bce(rp, rl).mean()
    tox_a_loss = (bce(tap, tal) * tak).sum() / (tak.sum() + 1e-8)
    tox_b_loss = (bce(tbp, tbl) * tbk).sum() / (tbk.sum() + 1e-8)
    return ddi_loss + 0.3 * (tox_a_loss + tox_b_loss)

def train_one_epoch(model, loader, opt):
    model.train(); total = 0
    for da, db, tal, tbl, tak, tbk, rl in loader:
        da, db = da.to(DEVICE), db.to(DEVICE)
        tal, tbl, tak, tbk, rl = tal.to(DEVICE), tbl.to(DEVICE), tak.to(DEVICE), tbk.to(DEVICE), rl.to(DEVICE)
        opt.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                rp, tap, tbp = model(da, db)
                loss = multi_task_loss(rp, tap, tbp, rl, tal, tbl, tak, tbk)
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(opt); scaler.update()
        else:
            rp, tap, tbp = model(da, db)
            loss = multi_task_loss(rp, tap, tbp, rl, tal, tbl, tak, tbk)
            loss.backward()
            opt.step()
        total += loss.item()
    return total / len(loader)

if __name__ == "__main__":
    os.makedirs(RUN_ARTIFACTS_DIR, exist_ok=True)
    
    print("STEP 1: Loading toxicity lookup...")
    tox_lookup = load_toxicity_lookup()

    print(f"\nSTEP 2: Loading TWOSIDES edges (capped at {DATA_CAP})...")
    edges = pd.read_csv(TWOSIDES_EDGES)
    edges = remove_reversed_duplicates(edges) # Prevent leakage
    edges = edges.sample(n=min(DATA_CAP, len(edges)), random_state=SEED)
    full_df = add_negative_samples(edges, neg_ratio=1.0, seed=SEED)

    print("\nSTEP 3: Creating cold-start splits...")
    splits = create_splits(full_df, drug_a_col='source', drug_b_col='target', seed=SEED)

    print("\nSTEP 4: Building DataLoaders...")
    train_loader = build_loader(splits['transductive_train'], tox_lookup, batch_size=128)
    val_loader = build_loader(splits['validation'], tox_lookup, batch_size=128, shuffle=False)
    test_loader = build_loader(splits['transductive_test'], tox_lookup, batch_size=128, shuffle=False)
    s1_loader = build_loader(splits['s1_test'], tox_lookup, batch_size=128, shuffle=False)
    s2_loader = build_loader(splits['s2_test'], tox_lookup, batch_size=128, shuffle=False)

    print("\nSTEP 5: Training...")
    model = PxDDIModel(in_channels=NUM_ATOM_FEATURES, hidden_channels=HIDDEN_CHANNELS, use_chemberta=USE_CHEMBERTA).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_auroc = 0
    for epoch in range(EPOCHS):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer)
        lr_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | LR: {current_lr:.6f} | Time: {time.time()-t0:.1f}s")
        labels, preds = get_preds_labels(model, val_loader)
        
        if len(set(labels)) > 1:
            from sklearn.metrics import roc_auc_score
            val_auroc = roc_auc_score(labels, preds)
            history['epoch'].append(epoch + 1)
            history['loss'].append(loss)
            history['auroc'].append(val_auroc)
            
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                save_checkpoint_safe({
                    'model_state_dict': model.state_dict(),
                    'hidden_channels': HIDDEN_CHANNELS,
                    'in_channels': NUM_ATOM_FEATURES,
                    'use_chemberta': USE_CHEMBERTA,
                    'auroc': float(val_auroc),
                    'epoch': epoch + 1,
                    'data_cap': DATA_CAP,
                }, CHECKPOINT_PATH)
                print(f"  -> New best model saved (VALIDATION AUROC {best_auroc:.4f})")

    print("\n=== FINAL BENCHMARK ===")
    best_checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint: epoch={best_checkpoint['epoch']}")

    # Find threshold on validation
    val_labels, val_preds = get_preds_labels(model, val_loader)
    fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
    FROZEN_THRESHOLD = thresholds[(tpr - fpr).argmax()]
    print(f"Threshold selected on VALIDATION set: {FROZEN_THRESHOLD:.4f}")
    
    # Add to checkpoint
    best_checkpoint['threshold'] = float(FROZEN_THRESHOLD)
    save_checkpoint_safe(best_checkpoint, CHECKPOINT_PATH)
    
    # Generate final plots using the requested suite
    PLOTS_DIR = DRIVE_BASE + 'plots/'
    os.makedirs(PLOTS_DIR, exist_ok=True)

    results_summary = {}
    for name, loader in [("Transductive", test_loader), ("S1", s1_loader), ("S2", s2_loader)]:
        labels, preds = get_preds_labels(model, loader)
        results_summary[name] = plot_roc_pr_confusion(name, labels, preds, FROZEN_THRESHOLD, PLOTS_DIR)

    plot_toxicity_bridge_coverage(
        total=toxicity_summary['source_rows'], 
        matched=toxicity_summary['rows_with_canonical_smiles'], 
        unique=toxicity_summary['unique_canonical_structures'], 
        conflicts=toxicity_summary['conflicting_canonical_structures'], 
        plots_dir=PLOTS_DIR
    )

    with open(PLOTS_DIR + 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("\nFull results summary saved as JSON — use directly in your paper's benchmark table.")
