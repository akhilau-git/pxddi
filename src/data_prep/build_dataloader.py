import torch, pandas as pd
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from .prepare_twosides import smiles_to_graph

COLUMN_OVERRIDE = {}  # fix here if auto-detect guesses wrong
SMILES_A_CANDIDATES = ['drug1_smiles','smiles_1','smiles_a','drug_a_smiles','SMILES_1']
SMILES_B_CANDIDATES = ['drug2_smiles','smiles_2','smiles_b','drug_b_smiles','SMILES_2']
LABEL_CANDIDATES = ['label','interaction','y']

def _detect(df, cands, key, name):
    if key in COLUMN_OVERRIDE: print(f"  [{name}] OVERRIDE: {COLUMN_OVERRIDE[key]}"); return COLUMN_OVERRIDE[key]
    for c in cands:
        if c in df.columns: print(f"  [{name}] auto-detected: {c}"); return c
    raise ValueError(f"Could not detect {name}. Columns: {df.columns.tolist()}")

def detect_columns(df):
    return (_detect(df, SMILES_A_CANDIDATES, "smiles_a", "Drug A"),
            _detect(df, SMILES_B_CANDIDATES, "smiles_b", "Drug B"),
            _detect(df, LABEL_CANDIDATES, "label", "Label"))


def parse_binary_label(value, column_name):
    """Return a verified binary label without turning every non-null value positive."""
    if pd.isna(value):
        raise ValueError(f"Missing binary label in column '{column_name}'")

    if isinstance(value, str):
        normalized = value.strip().lower()
        string_values = {
            '0': 0.0, 'false': 0.0, 'no': 0.0, 'negative': 0.0,
            '1': 1.0, 'true': 1.0, 'yes': 1.0, 'positive': 1.0,
        }
        if normalized in string_values:
            return string_values[normalized]

    try:
        label = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Column '{column_name}' must contain binary 0/1 labels; got {value!r}"
        ) from exc

    if label not in (0.0, 1.0):
        raise ValueError(
            f"Column '{column_name}' must contain binary 0/1 labels; got {value!r}"
        )
    return label

class DDIPairDataset(Dataset):
    def __init__(self, df, sa, sb, lc, ta=None, tb=None):
        super().__init__()
        recs, skip = [], 0
        for _, row in df.iterrows():
            ga, gb = smiles_to_graph(row[sa]), smiles_to_graph(row[sb])
            if ga is None or gb is None: skip += 1; continue
            label = parse_binary_label(row[lc], lc)
            recs.append((ga, gb, float(row[ta]) if ta else 0.0, float(row[tb]) if tb else 0.0, label))
        print(f"Built dataset: {len(recs)} valid, {skip} skipped")
        self.records = recs
    def len(self): return len(self.records)
    def get(self, i): return self.records[i]

def collate_fn(batch):
    ga = [b[0] for b in batch]; gb = [b[1] for b in batch]
    ta = torch.tensor([b[2] for b in batch], dtype=torch.float)
    tb = torch.tensor([b[3] for b in batch], dtype=torch.float)
    lb = torch.tensor([b[4] for b in batch], dtype=torch.float)
    return Batch.from_data_list(ga), Batch.from_data_list(gb), ta, tb, lb

def build_dataloader(df, batch_size=32, shuffle=True):
    sa, sb, lc = detect_columns(df)
    ds = DDIPairDataset(df, sa, sb, lc)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
