import torch, time, os, sys
sys.path.append('src')
from models.ddi_model import PxDDIModel
from data_prep.universal_loader import load_all_files_in_folder
from data_prep.prepare_twosides import NUM_ATOM_FEATURES
from data_prep.splits import create_splits
from data_prep.build_dataloader import build_dataloader
from sklearn.metrics import roc_auc_score, f1_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {DEVICE}")
assert DEVICE.type == 'cuda', "Set Runtime > GPU first!"
scaler = torch.cuda.amp.GradScaler()

def multi_task_loss(rp, tap, tbp, rl, tal, tbl):
    bce = torch.nn.BCEWithLogitsLoss()
    return bce(rp, rl) + 0.3*(bce(tap, tal) + bce(tbp, tbl))

def train_one_epoch(model, loader, opt):
    model.train(); total = 0
    for da, db, tal, tbl, rl in loader:
        da, db = da.to(DEVICE), db.to(DEVICE)
        tal, tbl, rl = tal.to(DEVICE), tbl.to(DEVICE), rl.to(DEVICE)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            rp, tap, tbp = model(da, db)
            loss = multi_task_loss(rp, tap, tbp, rl, tal, tbl)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total += loss.item()
    return total/len(loader)

def evaluate(model, loader, name):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for da, db, _, _, rl in loader:
            da, db = da.to(DEVICE), db.to(DEVICE)
            rp, _, _ = model(da, db)
            preds.extend(torch.sigmoid(rp).cpu().numpy()); labels.extend(rl.numpy())
    if len(set(labels)) < 2: print(f"  [{name}] SKIPPED"); return None, None
    auroc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, [1 if p>0.5 else 0 for p in preds])
    print(f"  [{name}] AUROC: {auroc:.4f} | F1: {f1:.4f}")
    return auroc, f1

if __name__ == "__main__":
    BASE = "/content/drive/MyDrive/pxddi-data/"
    CKPT = BASE + "checkpoints/pxddi_model.pt"
    print("Loading TWOSIDES..."); df = load_all_files_in_folder(BASE+"twosides/", row_cap_per_file=20000)
    print("Splitting..."); splits = create_splits(df)
    print("Building loaders...")
    train_l = build_dataloader(splits['transductive_train'], 32)
    test_l = build_dataloader(splits['transductive_test'], 32, False)
    s1_l = build_dataloader(splits['s1_test'], 32, False)
    s2_l = build_dataloader(splits['s2_test'], 32, False)

    model = PxDDIModel(in_channels=NUM_ATOM_FEATURES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best = 0
    for epoch in range(30):
        t0 = time.time()
        loss = train_one_epoch(model, train_l, opt)
        print(f"\nEpoch {epoch+1}/30 | Loss {loss:.4f} | {time.time()-t0:.1f}s")
        auroc, f1 = evaluate(model, test_l, "transductive_test")
        if auroc and auroc > best:
            best = auroc
            os.makedirs(BASE+"checkpoints/", exist_ok=True)
            torch.save(model.state_dict(), CKPT)
            print(f"  -> saved (AUROC {auroc:.4f})")
    print("\n=== FINAL BENCHMARK ===")
    evaluate(model, test_l, "Transductive"); evaluate(model, s1_l, "S1"); evaluate(model, s2_l, "S2")
