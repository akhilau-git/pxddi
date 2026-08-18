import pytest
import pandas as pd
from src.data_prep.pubchem_bridge import resolve_toxicity_bridge

def test_resolve_toxicity_bridge_keeps_exact_duplicates_once():
    """Verify exact duplicate toxicity scores are retained once."""
    bridge = pd.DataFrame({
        'drugname': ['DrugA', 'DrugA_Synonym', 'DrugB'],
        'raw_smiles': ['CCO', 'CCO', 'CCN'],
        'canonical_smiles': ['CCO', 'CCO', 'CCN'],
        'toxicity_score': [0.5, 0.5, 1.0],  # DrugA and Synonym have exact same score
        'n_reports': [100, 50, 20]
    })
    
    resolved, summary, conflicts = resolve_toxicity_bridge(bridge)
    
    assert len(resolved) == 2
    assert summary['duplicate_canonical_structures'] == 1 # CCO is a duplicate
    assert summary['conflicting_canonical_structures'] == 0 # but no conflict
    assert summary['resolved_unique_canonical_structures'] == 2
    assert len(conflicts) == 0
    
    # Order is an implementation detail; each structure must occur once.
    assert set(resolved['canonical_smiles']) == {'CCO', 'CCN'}

def test_resolve_toxicity_bridge_excludes_conflicting_scores():
    """Verify conflicting toxicity scores are completely excluded and returned in the conflict report."""
    bridge = pd.DataFrame({
        'drugname': ['DrugA', 'DrugA_Synonym', 'DrugB'],
        'raw_smiles': ['CCO', 'CCO', 'CCN'],
        'canonical_smiles': ['CCO', 'CCO', 'CCN'],
        'toxicity_score': [0.5, 0.9, 1.0],  # DrugA and Synonym have DIFFERENT scores
        'n_reports': [100, 50, 20]
    })
    
    resolved, summary, conflicts = resolve_toxicity_bridge(bridge)
    
    # DrugA (CCO) should be excluded completely
    assert len(resolved) == 1 
    assert list(resolved['canonical_smiles']) == ['CCN']
    
    assert summary['duplicate_canonical_structures'] == 1
    assert summary['conflicting_canonical_structures'] == 1
    assert summary['excluded_conflicting_structures'] == 1
    
    # The conflict report should contain the conflicting SMILES
    assert len(conflicts) == 1
    assert conflicts.iloc[0]['canonical_smiles'] == 'CCO'
    assert conflicts.iloc[0]['unique_scores'] == 2
