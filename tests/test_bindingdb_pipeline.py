import json
from pathlib import Path
import tempfile
import pandas as pd
import pytest
import torch

from src.data_prep.bindingdb_pipeline import (
    parse_bindingdb_directory,
    update_master_nodes_with_bindingdb,
    extract_top_target_vocabulary,
    encode_multihot_target_vector,
)
from src.data_prep.build_unified_graph import build_unified_graph
from src.data_prep.cached_graph_loader import (
    MolecularCache,
    build_cached_multimodal_dataloader,
)


@pytest.fixture
def mock_bindingdb_dir(tmp_path):
    """Create a minimal synthetic BindingDB dataset directory."""
    bdir = tmp_path / "BindingDB"
    bdir.mkdir()

    # Drugs
    # Aspirin: CC(=O)Oc1ccccc1C(=O)O
    # Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
    df_drugs = pd.DataFrame({
        "bindingdb_drug_id": ["BDB_D1", "BDB_D2"],
        "drug_name": ["Aspirin", "Caffeine"],
        "canonical_smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
    })
    df_drugs.to_csv(bdir / "bindingdb_drugs.csv", index=False)

    # Targets
    df_targets = pd.DataFrame({
        "bindingdb_target_id": ["T_COX1", "T_A2A"],
        "target_name": ["PTGS1", "ADORA2A"],
        "uniprot_id": ["P23219", "P29274"],
    })
    df_targets.to_csv(bdir / "bindingdb_targets.csv", index=False)

    # Edges
    df_edges = pd.DataFrame({
        "bindingdb_drug_id": ["BDB_D1", "BDB_D2"],
        "bindingdb_target_id": ["T_COX1", "T_A2A"],
        "affinity_nm": [12.5, 45.0],
    })
    df_edges.to_csv(bdir / "drug_target_edges.csv", index=False)

    return bdir


def test_extract_and_encode_target_vocabulary():
    target_lists = [["PTGS1", "PTGS2"], ["ADORA2A"], ["PTGS1", "ADORA2A"]]
    vocab = extract_top_target_vocabulary(target_lists, top_k=2)
    assert len(vocab) == 2
    assert "PTGS1" in vocab
    assert "ADORA2A" in vocab

    vec = encode_multihot_target_vector(["PTGS1"], vocab)
    idx = vocab.index("PTGS1")
    assert vec[idx] == 1
    assert sum(vec) == 1


def test_parse_bindingdb_directory(mock_bindingdb_dir):
    df_profiles, summary = parse_bindingdb_directory(mock_bindingdb_dir, top_k_targets=10)
    assert summary["total_profiled_drugs"] == 2
    assert summary["target_vocabulary_size"] == 2
    assert "PTGS1" in summary["top_10_targets"]
    assert "ADORA2A" in summary["top_10_targets"]

    assert len(df_profiles) == 2
    asp_row = df_profiles[df_profiles["canonical_smiles"] == "CC(=O)Oc1ccccc1C(=O)O"].iloc[0]
    assert "PTGS1" in json.loads(asp_row["targets_list"])
    assert bool(asp_row["is_bindingdb_active"]) is True


def test_update_master_nodes_with_bindingdb(mock_bindingdb_dir, tmp_path):
    # Create minimal master_drug_nodes.csv
    nodes_csv = tmp_path / "master_drug_nodes.csv"
    df_master = pd.DataFrame({
        "drug_id": ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCO"],  # Ethanol unprofiled
        "inchikey": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        "display_name": ["Aspirin", "Caffeine", "Ethanol"],
    })
    df_master.to_csv(nodes_csv, index=False)

    updated_df, summary = update_master_nodes_with_bindingdb(
        master_nodes_path=nodes_csv,
        bindingdb_dir_or_profiles=mock_bindingdb_dir,
        output_path=nodes_csv,
        top_k_targets=5,
    )

    assert summary["nodes_with_bindingdb_targets"] == 2
    assert summary["total_nodes"] == 3
    assert summary["bindingdb_coverage_pct"] == pytest.approx(66.67, rel=1e-2)

    # Check updated columns
    assert "bindingdb_targets_json" in updated_df.columns
    assert "bindingdb_target_vector" in updated_df.columns
    assert "is_bindingdb_active" in updated_df.columns

    # Aspirin and Caffeine active, Ethanol inactive
    assert updated_df.loc[updated_df["drug_id"] == "CC(=O)Oc1ccccc1C(=O)O", "is_bindingdb_active"].values[0] == True
    assert updated_df.loc[updated_df["drug_id"] == "CCO", "is_bindingdb_active"].values[0] == False


def test_build_unified_graph_with_bindingdb(mock_bindingdb_dir, tmp_path):
    # Minimal TWOSIDES edges
    twosides_csv = tmp_path / "twosides_edges.csv"
    smi_a = "CC(=O)Oc1ccccc1C(=O)O"
    smi_b = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    df_edges = pd.DataFrame({
        "source": [smi_a],
        "target": [smi_b],
        "interaction_type": ["Arrhythmia"],
    })
    df_edges.to_csv(twosides_csv, index=False)

    out_dir = tmp_path / "unified_graph_out"
    catalog, summary = build_unified_graph(
        twosides_edges_path=twosides_csv,
        bindingdb_dir=mock_bindingdb_dir,
        output_dir=out_dir,
        top_k_targets=10,
    )

    assert summary["total_nodes"] == 2
    assert summary["total_edges"] == 1
    assert summary["bindingdb_module_status"] == "active"
    assert summary["nodes_with_bindingdb_targets"] == 2
    assert summary["bindingdb_coverage_pct"] == 100.0
    assert (out_dir / "target_vocabulary.json").is_file()


def test_molecular_cache_with_bindingdb_targets(tmp_path):
    cache = MolecularCache(gene_dim=50, target_dim=10)

    smi_a = "CC(=O)Oc1ccccc1C(=O)O"
    target_vec_a = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    assert cache.register_drug(smi_a, target_vector=target_vec_a) is True
    assert cache.target_vectors[smi_a].shape == (10,)
    assert cache.target_masks[smi_a].item() == 1.0
    assert cache.target_vectors[smi_a][0].item() == 1.0
    assert cache.target_vectors[smi_a][1].item() == 0.0

    # Test populate from master nodes
    nodes_csv = tmp_path / "master_nodes_with_targets.csv"
    df_nodes = pd.DataFrame({
        "drug_id": [smi_a],
        "bindingdb_target_vector": [json.dumps(target_vec_a)],
    })
    df_nodes.to_csv(nodes_csv, index=False)

    fresh_cache = MolecularCache(gene_dim=50, target_dim=10)
    count = fresh_cache.populate_from_master_nodes(nodes_csv)
    assert count == 1
    assert fresh_cache.target_masks[smi_a].item() == 1.0


def test_update_master_nodes_keyword_aliases(mock_bindingdb_dir, tmp_path):
    nodes_csv = tmp_path / "master_nodes.csv"
    out_csv = tmp_path / "out_nodes.csv"
    df = pd.DataFrame({
        "drug_id": ["CC(=O)Oc1ccccc1C(=O)O"],
        "canonical_smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
    })
    df.to_csv(nodes_csv, index=False)

    tsv_file = mock_bindingdb_dir / "BindingDB_All.tsv"
    tsv_file.write_text("dummy tsv content")

    # Call with Colab keyword argument aliases
    updated_df, summary = update_master_nodes_with_bindingdb(
        master_nodes_csv=nodes_csv,
        bindingdb_tsv_path=tsv_file,
        output_csv=out_csv,
        tanimoto_threshold=0.15,
    )
    assert out_csv.is_file()
    assert summary["nodes_with_bindingdb_targets"] >= 1

    # Call with non-existent file path inside valid directory
    non_existent = mock_bindingdb_dir / "BindingDB_All_does_not_exist.tsv"
    updated_df2, summary2 = update_master_nodes_with_bindingdb(
        master_nodes_csv=nodes_csv,
        bindingdb_tsv_path=non_existent,
        output_csv=out_csv,
        tanimoto_threshold=0.15,
    )
    assert summary2["nodes_with_bindingdb_targets"] >= 1


