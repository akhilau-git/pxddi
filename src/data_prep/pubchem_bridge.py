"""
pubchem_bridge.py
Bridges FAERS drug NAMES to real SMILES structures using PubChem's
free public REST API. No synthetic data — every SMILES returned here
is a real, verified molecular structure from PubChem's database.

Caches results to Drive so we never re-query the same drug twice.
"""

import requests
import pandas as pd
import time
import os
from rdkit import Chem

PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/TXT"

def fetch_smiles_from_pubchem(drug_name: str, max_retries: int = 2):
    """Single real API call to PubChem. Returns SMILES string or None."""
    clean_name = drug_name.strip()
    for attempt in range(max_retries):
        try:
            resp = requests.get(PUBCHEM_URL.format(clean_name), timeout=10)
            if resp.status_code == 200:
                return resp.text.strip()
            return None  # 404 etc — name not found, not a network error
        except requests.exceptions.RequestException:
            time.sleep(1)
    return None


def canonicalize(smiles: str):
    """Standardizes a SMILES string so different-looking-but-same
    molecules match. Essential for matching PubChem's SMILES against
    TWOSIDES's SMILES correctly."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def build_name_to_smiles_bridge(tox_labels_df: pd.DataFrame, cache_path: str,
                                 top_n: int = 500, delay: float = 0.25):
    """
    Queries PubChem for the top_n most-reported FAERS drugs (by n_reports),
    builds a REAL drugname -> canonical_smiles mapping.
    Caches to cache_path so re-running doesn't re-query PubChem.
    delay: seconds between requests (respects PubChem's rate limits).
    """
    if os.path.exists(cache_path):
        print(f"Loading cached bridge from {cache_path}")
        return pd.read_csv(cache_path)

    top_drugs = tox_labels_df.sort_values('n_reports', ascending=False).head(top_n)
    print(f"Querying PubChem for top {top_n} most-reported drugs...")

    results = []
    for i, row in enumerate(top_drugs.itertuples(), 1):
        smiles = fetch_smiles_from_pubchem(str(row.drugname))
        canonical = canonicalize(smiles) if smiles else None
        results.append({
            'drugname': row.drugname,
            'raw_smiles': smiles,
            'canonical_smiles': canonical,
            'toxicity_score': row.toxicity_score,
            'n_reports': row.n_reports
        })
        if i % 50 == 0:
            print(f"  Progress: {i}/{top_n} — {sum(1 for r in results if r['canonical_smiles'])} matched so far")
        time.sleep(delay)

    bridge_df = pd.DataFrame(results)
    matched = bridge_df['canonical_smiles'].notna().sum()
    print(f"\nBRIDGE COMPLETE: {matched}/{top_n} drug names successfully matched to real SMILES "
          f"({matched/top_n*100:.1f}% match rate)")

    bridge_df.to_csv(cache_path, index=False)
    print(f"Cached to {cache_path} — future runs will load instantly instead of re-querying")
    return bridge_df
