"""
toxicity_lookup.py
Loads the real toxicity bridge (built during training) so the backend
can honestly report whether a drug's toxicity score is REAL data or
an unset default — never silently presenting 'unknown' as 'safe'.
"""
import pandas as pd
import sys
sys.path.append('/app/src')
from data_prep.pubchem_bridge import canonicalize

def load_known_toxicity_smiles(bridge_csv_path: str):
    try:
        bridge = pd.read_csv(bridge_csv_path)
        bridge = bridge.dropna(subset=['canonical_smiles'])
        return set(bridge['canonical_smiles'])
    except Exception as e:
        print(f"Warning: could not load toxicity lookup: {e}")
        return set()

KNOWN_TOXICITY_SMILES = load_known_toxicity_smiles('checkpoints/toxicity_smiles_bridge.csv')
