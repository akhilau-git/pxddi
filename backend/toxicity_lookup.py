"""
toxicity_lookup.py
Loads the real toxicity bridge (built during training) so the backend
can honestly report whether a drug's toxicity score is REAL data or
an unset default — never silently presenting 'unknown' as 'safe'.
"""
from pathlib import Path
import sys
import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))
from data_prep.pubchem_bridge import canonicalize

def load_known_toxicity_smiles(bridge_csv_path: str):
    try:
        bridge = pd.read_csv(bridge_csv_path)
        bridge = bridge.dropna(subset=['canonical_smiles'])
        return set(bridge['canonical_smiles'])
    except Exception as e:
        print(f"Warning: could not load toxicity lookup: {e}")
        return set()

CHECKPOINTS_DIR = Path(__file__).resolve().parent / 'checkpoints'
KNOWN_TOXICITY_SMILES = load_known_toxicity_smiles(
    CHECKPOINTS_DIR / 'toxicity_smiles_bridge.csv'
)


def is_toxicity_known(smiles: str, known_canonical_smiles=None) -> bool:
    """Check bridge coverage using canonical, rather than raw, SMILES."""
    canonical_smiles = canonicalize(smiles)
    known_smiles = (
        KNOWN_TOXICITY_SMILES
        if known_canonical_smiles is None
        else known_canonical_smiles
    )
    return canonical_smiles is not None and canonical_smiles in known_smiles
