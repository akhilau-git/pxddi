import pytest
import pandas as pd
from src.data_prep.faers_pipeline import aggregate_toxicity_labels

def test_aggregate_toxicity_labels_no_cartesian_explosion():
    """Verify multiple severe outcome rows for the same primaryid do not inflate n_reports counts."""
    
    drug = pd.DataFrame({
        'primaryid': [1, 1, 2, 3],
        'drugname': ['DrugA', 'DrugA', 'DrugB', 'DrugA']
    })
    
    # primaryid 1 has MULTIPLE severe outcomes. If not aggregated correctly, 
    # merging could result in 2 drug rows * 2 outc rows = 4 rows for primaryid 1!
    outc = pd.DataFrame({
        'primaryid': [1, 1, 2, 3],
        'outc_cod': ['DE', 'HO', 'OT', 'OT'] # DE and HO are severe
    })
    
    # min_reports=1 to allow small test datasets
    tox = aggregate_toxicity_labels(drug, outc, min_reports=1, missing_outcome_policy='exclude')
    
    # Expected: 
    # primaryid 1: max severe = 1 (DrugA)
    # primaryid 2: max severe = 0 (DrugB)
    # primaryid 3: max severe = 0 (DrugA)
    
    assert len(tox) == 2 # DrugA, DrugB
    
    drug_a_tox = tox[tox['drugname'] == 'DRUGA'].iloc[0]
    # DrugA has 2 unique reports: id 1 (severe) and id 3 (not severe)
    assert drug_a_tox['n_reports'] == 2
    assert drug_a_tox['toxicity_score'] == 0.5
    
    drug_b_tox = tox[tox['drugname'] == 'DRUGB'].iloc[0]
    assert drug_b_tox['n_reports'] == 1
    assert drug_b_tox['toxicity_score'] == 0.0

def test_aggregate_toxicity_labels_missing_outcome_policy():
    """Verify behavior of missing_outcome_policy ('exclude' vs 'non_severe')."""
    
    drug = pd.DataFrame({
        'primaryid': [1, 2, 3],
        'drugname': ['DrugA', 'DrugA', 'DrugB']
    })
    
    outc = pd.DataFrame({
        'primaryid': [1], # Only report 1 has an outcome
        'outc_cod': ['DE']
    })
    
    # 1. Exclude policy (reports without outcomes are completely dropped)
    tox_exclude = aggregate_toxicity_labels(drug, outc, min_reports=1, missing_outcome_policy='exclude')
    # primaryid 2 and 3 dropped. DrugB has 0 reports -> omitted. DrugA has 1 report.
    assert len(tox_exclude) == 1
    assert tox_exclude.iloc[0]['n_reports'] == 1
    assert tox_exclude.iloc[0]['toxicity_score'] == 1.0
    
    # 2. Non-severe policy (reports without outcomes are treated as non-severe = 0)
    tox_nonsevere = aggregate_toxicity_labels(drug, outc, min_reports=1, missing_outcome_policy='non_severe')
    # primaryid 2 and 3 retained.
    # DrugA: id 1 (severe), id 2 (non-severe). n_reports = 2. score = 0.5
    # DrugB: id 3 (non-severe). n_reports = 1. score = 0.0
    assert len(tox_nonsevere) == 2
    
    drug_a_tox = tox_nonsevere[tox_nonsevere['drugname'] == 'DRUGA'].iloc[0]
    assert drug_a_tox['n_reports'] == 2
    assert drug_a_tox['toxicity_score'] == 0.5
    
    drug_b_tox = tox_nonsevere[tox_nonsevere['drugname'] == 'DRUGB'].iloc[0]
    assert drug_b_tox['n_reports'] == 1
    assert drug_b_tox['toxicity_score'] == 0.0
