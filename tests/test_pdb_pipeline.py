"""Tests for the PDB macromolecular complex pipeline and multimodal integration."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from src.data_prep.pdb_pipeline import (
    extract_top_pdb_vocabulary,
    encode_multihot_pdb_vector,
    parse_pdb_directory,
    update_master_nodes_with_pdb,
)
from src.data_prep.cached_graph_loader import (
    MolecularCache,
    build_cached_multimodal_dataloader,
)


@pytest.fixture
def mock_pdb_dir(tmp_path):
    """Create synthetic PDB tabular dataset."""
    pdir = tmp_path / "pdb"
    pdir.mkdir()

    # Create mock pdb_ligands.csv
    ligands_df = pd.DataFrame([
        {
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "pdb_id": "1O42",
            "target_name": "PROSTAGLANDIN_G_H_SYNTHASE_1",
            "resolution": 2.1,
            "drug_name": "Aspirin",
        },
        {
            "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # Caffeine
            "pdb_id": "3RFM",
            "target_name": "ADENOSINE_A2A_RECEPTOR",
            "resolution": 1.8,
            "drug_name": "Caffeine",
        },
    ])
    ligands_df.to_csv(pdir / "pdb_ligands.csv", index=False)
    return pdir


def test_pdb_vocab_and_encoding():
    target_lists = [
        ["ADENOSINE_A2A_RECEPTOR", "PROSTAGLANDIN_G_H_SYNTHASE_1"],
        ["ADENOSINE_A2A_RECEPTOR"],
    ]
    vocab = extract_top_pdb_vocabulary(target_lists, top_k=5)
    assert len(vocab) == 2
    assert vocab[0] == "ADENOSINE_A2A_RECEPTOR"

    vec = encode_multihot_pdb_vector(["ADENOSINE_A2A_RECEPTOR"], vocab)
    assert vec == [1, 0]


def test_parse_pdb_directory(mock_pdb_dir):
    df_pdb, summary = parse_pdb_directory(mock_pdb_dir)
    assert len(df_pdb) == 2
    assert summary["total_pdb_drugs"] == 2
    assert "pdb_vector_multihot" in df_pdb.columns
    assert "pdb_resolution_score" in df_pdb.columns


def test_update_master_nodes_with_pdb(mock_pdb_dir, tmp_path):
    nodes_csv = tmp_path / "master_drug_nodes.csv"
    nodes_df = pd.DataFrame([
        {
            "drug_id": "CC(=O)Oc1ccccc1C(=O)O",
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        },
        {
            "drug_id": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        },
        {
            "drug_id": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Unprofiled
            "canonical_smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        },
    ])
    nodes_df.to_csv(nodes_csv, index=False)

    res = update_master_nodes_with_pdb(nodes_csv, mock_pdb_dir)
    assert res["nodes_with_pdb_structures"] >= 2

    enriched = pd.read_csv(nodes_csv)
    assert "pdb_vector_multihot" in enriched.columns
    assert "is_pdb_active" in enriched.columns
    assert enriched["is_pdb_active"].sum() >= 2


def test_molecular_cache_pdb_collation(mock_pdb_dir, tmp_path):
    nodes_csv = tmp_path / "master_drug_nodes.csv"
    smi_a = "CC(=O)Oc1ccccc1C(=O)O"
    smi_b = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

    nodes_df = pd.DataFrame([
        {
            "drug_id": smi_a,
            "canonical_smiles": smi_a,
            "pdb_vector_multihot": json.dumps([1] * 50),
        },
        {
            "drug_id": smi_b,
            "canonical_smiles": smi_b,
            "pdb_vector_multihot": json.dumps([0] * 50),
        },
    ])
    nodes_df.to_csv(nodes_csv, index=False)

    cache = MolecularCache()
    count = cache.populate_from_master_nodes(nodes_csv)
    assert count == 2
    assert cache.pdb_masks[smi_a].item() == 1.0
    assert cache.pdb_vectors[smi_a].shape == (50,)

    edges_df = pd.DataFrame({
        "drug_a_id": [smi_a],
        "drug_b_id": [smi_b],
        "label": [1.0],
    })

    loader = build_cached_multimodal_dataloader(edges_df, cache, batch_size=1)
    batch = next(iter(loader))
    assert "pdb_a" in batch
    assert "pdb_b" in batch
    assert "pdb_mask_a" in batch
    assert batch["pdb_a"].shape == (1, 50)
    assert batch["pdb_mask_a"][0].item() == 1.0


def test_update_master_nodes_with_pdb_stem_and_gene_fallback(tmp_path):
    """Verify fallback target cross-referencing and pharmacological stem matching."""
    nodes_csv = tmp_path / "master_drug_nodes_fallback.csv"
    empty_pdb_dir = tmp_path / "empty_pdb"
    empty_pdb_dir.mkdir()

    nodes_df = pd.DataFrame([
        {
            "drug_id": "DRUG_STATIN",
            "canonical_smiles": "CC1C=CC2C(C1)C(C(C=C2)C)OC(=O)C(C)CC",
            "display_name": "Atorvastatin",
            "gene_symbols_json": json.dumps(["HMGCR", "CYP3A4"]),
        },
        {
            "drug_id": "DRUG_BETA_BLOCKER",
            "canonical_smiles": "CC(C)NCC(COc1cccc2ccccc12)O",
            "display_name": "Propranolol",
        },
        {
            "drug_id": "DRUG_UNPROFILED",
            "canonical_smiles": "CCCC",
            "display_name": "ButaneUnknown",
        },
    ])
    nodes_df.to_csv(nodes_csv, index=False)

    res = update_master_nodes_with_pdb(nodes_csv, empty_pdb_dir)
    # Both Atorvastatin (from genes + stem) and Propranolol (from 'olol' stem -> ADRB1, ADRB2) should match!
    assert res["nodes_with_pdb_structures"] >= 2

    enriched = pd.read_csv(nodes_csv)
    assert enriched.loc[enriched["drug_id"] == "DRUG_STATIN", "is_pdb_active"].values[0]
    assert enriched.loc[enriched["drug_id"] == "DRUG_BETA_BLOCKER", "is_pdb_active"].values[0]
    vec_statin = json.loads(enriched.loc[enriched["drug_id"] == "DRUG_STATIN", "pdb_vector_multihot"].values[0])
    assert any(x > 0 for x in vec_statin)
