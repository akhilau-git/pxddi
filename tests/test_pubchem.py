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


def test_build_faers_bridge_import_and_execution(tmp_path, monkeypatch):
    """Verify build_faers_bridge executes properly with signals and master nodes."""
    from src.data_prep.pubchem_bridge import build_faers_bridge, PubChemLookupResult

    # Mock lookup so no internet call is needed
    monkeypatch.setattr(
        'src.data_prep.pubchem_bridge.lookup_pubchem_smiles',
        lambda drug_name, **kwargs: PubChemLookupResult('CCO', 'matched', 1, 'http://test'),
    )

    signals_csv = tmp_path / 'signals.csv'
    pd.DataFrame({
        'drugname': ['ASPIRIN', 'WARFARIN'],
        'toxicity_score': [0.2, 0.8],
        'n_reports': [100, 50],
    }).to_csv(signals_csv, index=False)

    nodes_csv = tmp_path / 'nodes.csv'
    pd.DataFrame({
        'drug_name': ['Aspirin', 'Warfarin'],
        'canonical_smiles': ['CC(=O)Oc1ccccc1C(=O)O', 'CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O'],
    }).to_csv(nodes_csv, index=False)

    out_csv = tmp_path / 'faers_bridge.csv'
    df = build_faers_bridge(
        faers_signals_path=signals_csv,
        master_nodes_path=nodes_csv,
        output_path=out_csv,
        top_n_faers=10,
        delay=0.0,
    )
    assert len(df) == 2
    assert out_csv.exists()
    assert 'canonical_smiles' in df.columns


def test_build_faers_bridge_with_ascii_dir(tmp_path, monkeypatch):
    """Verify build_faers_bridge discovers and builds from FAERS ascii directory."""
    from src.data_prep.pubchem_bridge import build_faers_bridge, PubChemLookupResult

    monkeypatch.setattr(
        'src.data_prep.pubchem_bridge.lookup_pubchem_smiles',
        lambda drug_name, **kwargs: PubChemLookupResult('CCO', 'matched', 1, 'http://test'),
    )

    ascii_dir = tmp_path / 'faers' / 'ASCII'
    ascii_dir.mkdir(parents=True)
    # Write minimal ASCII files
    (ascii_dir / 'DRUG20Q1.txt').write_text(
        'primaryid$drug_seq$role_cod$drugname\n1$1$PS$ASPIRIN\n2$1$PS$WARFARIN\n',
        encoding='utf-8',
    )
    (ascii_dir / 'OUTC20Q1.txt').write_text(
        'primaryid$outc_cod\n1$DE\n2$HO\n',
        encoding='utf-8',
    )

    out_csv = tmp_path / 'faers_bridge.csv'
    # Test auto-detection when faers_signals_path points to nonexistent signals in a sibling directory
    nonexistent_signals = tmp_path / 'clinical_evidence' / 'faers_drug_event_signals.csv'
    df = build_faers_bridge(
        faers_signals_path=nonexistent_signals,
        output_path=out_csv,
        top_n_faers=10,
        min_reports=1,
        delay=0.0,
    )
    assert len(df) >= 1
    assert out_csv.exists()


def test_pharmgkb_faers_provenance_tracking(tmp_path):
    from src.data_prep.expanded_pharmgkb_bridge import update_master_nodes_with_pharmgkb_faers_analogs

    # Drug A is profiled, Drug B is an analog of Drug A, Drug C has no analog
    master_df = pd.DataFrame([
        {
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "gene_symbols": "['CYP2C9']",
            "gene_vector_multihot": [1] + [0] * 49,
            "toxicity_score": 0.25,
            "n_faers_reports": 10,
        },
        {
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)OCC",  # Ethyl acetylsalicylate (analog)
            "gene_symbols": None,
            "gene_vector_multihot": None,
            "toxicity_score": None,
            "n_faers_reports": None,
        },
    ])
    in_csv = tmp_path / "master_nodes.csv"
    master_df.to_csv(in_csv, index=False)

    out_csv = tmp_path / "master_nodes_enriched.csv"
    res_df, summary = update_master_nodes_with_pharmgkb_faers_analogs(
        master_nodes_csv=in_csv,
        output_path=out_csv,
        similarity_threshold=0.60,
    )

    assert "pharmgkb_provenance" in res_df.columns
    assert "faers_provenance" in res_df.columns
    assert "is_imputed_pharmgkb" in res_df.columns
    assert "is_imputed_faers" in res_df.columns

    # Check that drug 0 was observed and drug 1 was imputed
    assert res_df.loc[0, "pharmgkb_provenance"] == "observed"
    assert res_df.loc[0, "is_imputed_pharmgkb"] is False or res_df.loc[0, "is_imputed_pharmgkb"] == 0
    assert res_df.loc[1, "pharmgkb_provenance"] == "imputed_analog"
    assert res_df.loc[1, "is_imputed_pharmgkb"] is True or res_df.loc[1, "is_imputed_pharmgkb"] == 1

    # Verify that input file was preserved
    orig = pd.read_csv(in_csv)
    assert "pharmgkb_provenance" not in orig.columns
    assert out_csv.is_file()



