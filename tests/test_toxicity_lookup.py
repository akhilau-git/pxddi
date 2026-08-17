"""Tests for canonical toxicity-coverage checks."""

from backend.toxicity_lookup import is_toxicity_known
from src.data_prep.pubchem_bridge import canonicalize


def test_equivalent_smiles_match_canonical_toxicity_coverage():
    raw_aspirin = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    canonical_aspirin = canonicalize(raw_aspirin)

    assert canonical_aspirin is not None
    assert is_toxicity_known(raw_aspirin, {canonical_aspirin})


def test_invalid_smiles_is_not_known():
    assert not is_toxicity_known('not-a-smiles', set())
