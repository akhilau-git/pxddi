"""Tests for canonical toxicity-coverage checks."""

import pandas as pd

from backend.toxicity_lookup import is_toxicity_known, load_known_toxicity_smiles
from src.data_prep.pubchem_bridge import canonicalize


def test_equivalent_smiles_match_canonical_toxicity_coverage():
    raw_aspirin = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    canonical_aspirin = canonicalize(raw_aspirin)

    assert canonical_aspirin is not None
    assert is_toxicity_known(raw_aspirin, {canonical_aspirin})


def test_invalid_smiles_is_not_known():
    assert not is_toxicity_known('not-a-smiles', set())


def test_conflicting_bridge_structure_is_excluded_from_clean_coverage(tmp_path):
    bridge_path = tmp_path / 'bridge.csv'
    pd.DataFrame({
        'drugname': ['Drug A', 'Drug A synonym', 'Drug B'],
        'raw_smiles': ['CCO', 'CCO', 'CCN'],
        'canonical_smiles': ['CCO', 'CCO', 'CCN'],
        'toxicity_score': [0.2, 0.8, 0.4],
        'n_reports': [10, 20, 30],
    }).to_csv(bridge_path, index=False)

    known_smiles, error, summary = load_known_toxicity_smiles(bridge_path)

    assert error is None
    assert known_smiles == {'CCN'}
    assert summary is not None
    assert summary['excluded_conflicting_structures'] == 1
