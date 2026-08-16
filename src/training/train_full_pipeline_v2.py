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
CHECKPOINT_PATH = DRIVE_BASE + 'checkpoints/pxddi_model.pt'

DATA_CAP = 200000  # bump to 200000 once this run confirms everything works
EPOCHS = 200
HIDDEN_CHANNELS = 128


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


def get_toxicity(smiles, lookup, default=0.0):
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
            records.append((ga, gb, tox_a, tox_b, label))
            if i % 5000 == 0:
                print(f"    ...processed {i}/{total} rows")
        print(f"Built dataset: {len(records)} valid pairs, {skipped} skipped (bad SMILES)")
        self.records = records

    def len(self): return len(self.records)
    def get(self, idx): return self.records[idx]


def collate_fn(batch):
    ga = [b[0] for b in batch]
    gb = [b[1] for b in batch]
    ta = torch.tensor([b[2] for b in batch], dtype=torch.float)
    tb = torch.tensor([b[3] for b in batch], dtype=torch.float)
    lb = torch.tensor([b[4] for b in batch], dtype=torch.float)
    return Batch.from_data_list(ga), Batch.from_data_list(gb), ta, tb, lb


def build_loader(df, tox_lookup, batch_size=32, shuffle=True):
    ds = PxDDIDataset(df, tox_lookup)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def multi_task_loss(rp, tap, tbp, rl, tal, tbl):
    bce = torch.nn.BCEWithLogitsLoss()
    return bce(rp, rl) + 0.3 * (bce(tap, tal) + bce(tbp, tbl))


def train_one_epoch(model, loader, opt):
    model.train(); total = 0
    for da, db, tal, tbl, rl in loader:
        da, db = da.to(DEVICE), db.to(DEVICE)
        tal, tbl, rl = tal.to(DEVICE), tbl.to(DEVICE), rl.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            rp, tap, tbp = model(da, db)
            loss = multi_task_loss(rp, tap, tbp, rl, tal, tbl)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total += loss.item()
    return total / len(loader)


def find_best_threshold(labels, preds):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    j_scores = tpr - fpr  # Youden's J statistic — standard threshold selection method
    best_idx = j_scores.argmax()
    return thresholds[best_idx]


def evaluate(model, loader, name):
    """FIX #4: also prints class balance so we can tell if a low AUROC
    is a real modeling issue or just noise from a small/unbalanced set."""
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for da, db, _, _, rl in loader:
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
    best_thresh = find_best_threshold(labels, preds)
    f1 = f1_score(labels, [1 if p > 0.5 else 0 for p in preds])
    best_f1 = f1_score(labels, [1 if p > best_thresh else 0 for p in preds])
    print(f"  [{name}] AUROC: {auroc:.4f} | F1 (0.5 threshold): {f1:.4f} | F1 (optimal {best_thresh:.3f}): {best_f1:.4f}")
    return auroc, best_f1


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
    test_loader = build_loader(splits['transductive_test'], tox_lookup, batch_size=128, shuffle=False)
    s1_loader = build_loader(splits['s1_test'], tox_lookup, batch_size=128, shuffle=False)
    s2_loader = build_loader(splits['s2_test'], tox_lookup, batch_size=128, shuffle=False)

    print("\nSTEP 5: Training...")
    model = PxDDIModel(in_channels=NUM_ATOM_FEATURES, hidden_channels=HIDDEN_CHANNELS).to(DEVICE)
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
        auroc, f1 = evaluate(model, test_loader, "transductive_test")
        if auroc and auroc > best_auroc:
            best_auroc = auroc
            torch.save({
                'model_state_dict': model.state_dict(),
                'hidden_channels': HIDDEN_CHANNELS,
                'in_channels': NUM_ATOM_FEATURES,
                'auroc': auroc,
                'epoch': epoch + 1,
                'data_cap': DATA_CAP,
            }, CHECKPOINT_PATH)
            print(f"  -> New best model saved (AUROC {auroc:.4f})")

    print("\n=== FINAL BENCHMARK TABLE (real data, all modules) ===")
    evaluate(model, test_loader, "Transductive")
    evaluate(model, s1_loader, "S1 (both drugs unseen)")
    evaluate(model, s2_loader, "S2 (one drug unseen)")
