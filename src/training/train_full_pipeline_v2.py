"""
train_full_pipeline_v2.py — v2, with 4 fixes:
1. Data cap raised (50k this run, easy to bump to 200k after)
2. 100 epochs + learning rate scheduler
3. Toxicity bridge coverage diagnostic (shows why 402 -> 339)
4. S1 class balance diagnostic (checks if 0.4973 is a real signal or noise)
"""

import torch
import time
import os
import sys
sys.path.append('/content/drive/MyDrive/pxddi-data/pxddi/src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader

from models.ddi_model import PxDDIModel
from data_prep.prepare_twosides import smiles_to_graph, NUM_ATOM_FEATURES
from data_prep.splits import create_splits
from data_prep.pubchem_bridge import canonicalize

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {DEVICE}")
assert DEVICE.type == 'cuda', "Set Runtime > GPU first!"
scaler = torch.amp.GradScaler('cuda')

DRIVE_BASE = '/content/drive/MyDrive/pxddi-data/'
TWOSIDES_EDGES = DRIVE_BASE + 'twosides/drug_drug_edges.csv'
TOXICITY_BRIDGE = DRIVE_BASE + 'checkpoints/toxicity_smiles_bridge.csv'
USE_CHEMBERTA = False
CHECKPOINT_PATH = DRIVE_BASE + ('checkpoints/pxddi_model_chemberta.pt' if USE_CHEMBERTA else 'checkpoints/pxddi_model.pt')

DATA_CAP = 200000
EPOCHS = 200
HIDDEN_CHANNELS = 128

history = {'epoch': [], 'loss': [], 'auroc': [], 'f1': []}

def plot_training_curves(history, save_path):
    """Saves loss and AUROC curves as PNG — use these directly in your paper."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['epoch'], history['loss'], color='crimson')
    axes[0].set_title('Training Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['epoch'], history['auroc'], label='AUROC', color='navy')
    axes[1].plot(history['epoch'], history['f1'], label='F1', color='seagreen')
    axes[1].set_title('Validation Performance Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curves to {save_path}")
    plt.close()

def plot_benchmark_comparison(results_dict, save_path):
    """
    results_dict example:
    {'Transductive': (0.9735, 0.9170), 'S1 (unseen)': (0.5021, 0.3530), 'S2 (one unseen)': (0.7474, 0.6387)}
    This is your headline comparison chart — put this directly in your paper.
    """
    labels = list(results_dict.keys())
    aurocs = [v[0] for v in results_dict.values()]
    f1s = [v[1] for v in results_dict.values()]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar([i - width/2 for i in x], aurocs, width, label='AUROC', color='navy')
    ax.bar([i + width/2 for i in x], f1s, width, label='F1', color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline (0.5)')
    ax.set_title('PxDDI Performance: Transductive vs. Cold-Start Splits')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved benchmark comparison to {save_path}")
    plt.close()

def plot_roc_curve(labels, preds, save_path, title="ROC Curve"):
    """A real ROC curve — standard in every ML paper's results section."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=150)
    print(f"Saved ROC curve to {save_path}")
    plt.close()


def load_toxicity_lookup():
    """FIX #3: diagnose why 402 matched drugs -> only 339 used in training."""
    bridge = pd.read_csv(TOXICITY_BRIDGE)
    print(f"Bridge file has {len(bridge)} total rows")
    matched = bridge.dropna(subset=['canonical_smiles'])
    print(f"  {len(matched)} rows have a non-null canonical_smiles")
    lookup = dict(zip(matched['canonical_smiles'], matched['toxicity_score']))
    print(f"  {len(lookup)} unique canonical SMILES in final lookup dict "
          f"(duplicates collapse to one entry — this explains 402 -> fewer if drugs share structures)")
    return lookup


def get_toxicity(smiles, lookup, default=None):
    canon = canonicalize(smiles)
    return lookup.get(canon, default)


def add_negative_samples(df, source_col='source', target_col='target', neg_ratio=1.0, seed=42):
    rng = np.random.default_rng(seed)
    all_drugs = pd.unique(df[[source_col, target_col]].values.ravel())
    real_pairs = set(zip(df[source_col], df[target_col]))

    n_negatives = int(len(df) * neg_ratio)
    negatives = []
    attempts = 0
    max_attempts = n_negatives * 20

    while len(negatives) < n_negatives and attempts < max_attempts:
        a, b = rng.choice(all_drugs, size=2, replace=False)
        if (a, b) not in real_pairs and (b, a) not in real_pairs:
            negatives.append({source_col: a, target_col: b, 'label': 0.0})
        attempts += 1

    print(f"Generated {len(negatives)} negative samples (target was {n_negatives})")

    positives = df[[source_col, target_col]].copy()
    positives['label'] = 1.0
    negatives_df = pd.DataFrame(negatives).drop_duplicates(subset=[source_col, target_col])
    print(f"After deduplication: {len(negatives_df)} unique 'no interaction reported in TWOSIDES' samples")

    combined = pd.concat([positives, negatives_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Final dataset: {len(combined)} rows ({positives.shape[0]} pos, {len(negatives_df)} 'no interaction reported in TWOSIDES')")
    return combined


class PxDDIDataset(Dataset):
    def __init__(self, df, tox_lookup, source_col='source', target_col='target', label_col='label'):
        super().__init__()
        records = []
        skipped = 0
        skipped_examples = []
        total = len(df)
        for i, row in enumerate(df.itertuples(), 1):
            source, target, label = getattr(row, source_col), getattr(row, target_col), getattr(row, label_col)
            ga = smiles_to_graph(source)
            gb = smiles_to_graph(target)
            if ga is None or gb is None:
                skipped += 1
                skipped_examples.append((source, target))
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
        print(f"Built dataset: {len(records)} valid pairs, {skipped} skipped (bad SMILES)")
        if skipped_examples:
            print(f"Sample of skipped SMILES (first 5): {skipped_examples[:5]}")
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
        with torch.amp.autocast('cuda'):
            rp, tap, tbp = model(da, db)
            loss = multi_task_loss(rp, tap, tbp, rl, tal, tbl, tak, tbk)
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()  # type: ignore
        scaler.step(opt); scaler.update()
        total += loss.item()
    return total / len(loader)


def find_best_threshold(labels, preds):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    j_scores = tpr - fpr  # Youden's J statistic — standard threshold selection method
    best_idx = j_scores.argmax()
    return thresholds[best_idx]


def evaluate(model, loader, name, threshold=None):
    """FIX #4: also prints class balance so we can tell if a low AUROC
    is a real modeling issue or just noise from a small/unbalanced set."""
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for da, db, _, _, _, _, rl in loader:
            da, db = da.to(DEVICE), db.to(DEVICE)
            rp, _, _ = model(da, db)
            preds.extend(torch.sigmoid(rp).cpu().numpy())
            labels.extend(rl.numpy())

    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    print(f"  [{name}] set size: {len(labels)} (pos={n_pos}, neg={n_neg})")

    if len(set(labels)) < 2:
        print(f"  [{name}] SKIPPED — only one class present")
        return None, None
    if len(labels) < 100:
        print(f"  [{name}] WARNING: small test set ({len(labels)} samples) — AUROC may be noisy")

    auroc = roc_auc_score(labels, preds)
    if threshold is None:
        best_thresh = find_best_threshold(labels, preds)
        f1 = f1_score(labels, [1 if p > 0.5 else 0 for p in preds])
        best_f1 = f1_score(labels, [1 if p > best_thresh else 0 for p in preds])
        print(f"  [{name}] AUROC: {auroc:.4f} | F1 (0.5 threshold): {f1:.4f} | F1 (optimal {best_thresh:.3f}): {best_f1:.4f}")
        return auroc, best_f1
    else:
        f1 = f1_score(labels, [1 if p > threshold else 0 for p in preds])
        print(f"  [{name}] AUROC: {auroc:.4f} | F1 (threshold {threshold:.3f}): {f1:.4f}")
        return auroc, f1


if __name__ == "__main__":
    print("STEP 1: Loading real toxicity lookup...")
    tox_lookup = load_toxicity_lookup()

    print(f"\nSTEP 2: Loading real TWOSIDES edges (capped at {DATA_CAP}) + generating negatives...")
    edges = pd.read_csv(TWOSIDES_EDGES)
    print(f"Raw positive edges (before cap): {len(edges)}")
    edges = edges.sample(n=min(DATA_CAP, len(edges)), random_state=42)
    print(f"Raw positive edges (after cap): {len(edges)}")
    full_df = add_negative_samples(edges, neg_ratio=1.0)

    print("\nSTEP 3: Creating cold-start-aware splits...")
    splits = create_splits(full_df, drug_a_col='source', drug_b_col='target')
    for name, split_df in splits.items():
        print(f"  {name}: {len(split_df)} rows")

    print("\nSTEP 4: Building real DataLoaders...")
    train_loader = build_loader(splits['transductive_train'], tox_lookup, batch_size=128)
    val_loader = build_loader(splits['validation'], tox_lookup, batch_size=128, shuffle=False)
    test_loader = build_loader(splits['transductive_test'], tox_lookup, batch_size=128, shuffle=False)
    s1_loader = build_loader(splits['s1_test'], tox_lookup, batch_size=128, shuffle=False)
    s2_loader = build_loader(splits['s2_test'], tox_lookup, batch_size=128, shuffle=False)

    print("\nSTEP 5: Training...")
    model = PxDDIModel(in_channels=NUM_ATOM_FEATURES, hidden_channels=HIDDEN_CHANNELS, use_chemberta=USE_CHEMBERTA).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # FIX #2: decaying learning rate schedule, matching FG-DDI's approach
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_auroc = 0
    for epoch in range(EPOCHS):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer)
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | LR: {current_lr:.6f} | Time: {time.time()-t0:.1f}s")
        val_auroc, val_f1 = evaluate(model, val_loader, "validation")
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(loss)
        history['auroc'].append(val_auroc if val_auroc else 0)
        history['f1'].append(val_f1 if val_f1 else 0)
        
        if val_auroc and val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'model_state_dict': model.state_dict(),
                'hidden_channels': HIDDEN_CHANNELS,
                'in_channels': NUM_ATOM_FEATURES,
                'use_chemberta': USE_CHEMBERTA,
                'auroc': val_auroc,
                'epoch': epoch + 1,
                'data_cap': DATA_CAP,
            }, CHECKPOINT_PATH)
            print(f"  -> New best model saved (VALIDATION AUROC {val_auroc:.4f})")

    print("\nReloading BEST checkpoint (not final epoch) for final reporting...")
    best_checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint: AUROC={best_checkpoint['auroc']:.4f}, epoch={best_checkpoint['epoch']}")

    # Find threshold on validation set
    val_preds, val_labels = [], []
    model.eval()
    with torch.no_grad():
        for da, db, _, _, _, _, rl in val_loader:
            da, db = da.to(DEVICE), db.to(DEVICE)
            rp, _, _ = model(da, db)
            val_preds.extend(torch.sigmoid(rp).cpu().numpy())
            val_labels.extend(rl.numpy())
    FROZEN_THRESHOLD = find_best_threshold(val_labels, val_preds)
    print(f"Threshold selected on VALIDATION set (not test): {FROZEN_THRESHOLD:.4f}")
    
    # Save threshold into the checkpoint dict
    best_checkpoint['threshold'] = float(FROZEN_THRESHOLD)
    torch.save(best_checkpoint, CHECKPOINT_PATH)
    print("Saved optimal threshold to checkpoint.")

    print("\n=== FINAL BENCHMARK (test touched ONCE, model selected via validation) ===")
    print(f"Threshold (from validation): {FROZEN_THRESHOLD:.4f}")
    
    PLOTS_DIR = DRIVE_BASE + 'plots/'
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_training_curves(history, PLOTS_DIR + 'training_curves.png')

    final_results = {
        'Transductive': evaluate(model, test_loader, "Transductive TEST", threshold=FROZEN_THRESHOLD),
        'S1 (unseen)': evaluate(model, s1_loader, "S1 (both unseen) TEST", threshold=FROZEN_THRESHOLD),
        'S2 (one unseen)': evaluate(model, s2_loader, "S2 (one unseen) TEST", threshold=FROZEN_THRESHOLD),
    }
    plot_benchmark_comparison(final_results, PLOTS_DIR + 'benchmark_comparison.png')
    
    # Get predictions for transductive set for ROC curve
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for da, db, _, _, _, _, rl in test_loader:
            da, db = da.to(DEVICE), db.to(DEVICE)
            rp, _, _ = model(da, db)
            preds.extend(torch.sigmoid(rp).cpu().numpy())
            labels.extend(rl.numpy())
    plot_roc_curve(labels, preds, PLOTS_DIR + 'roc_curve.png')
