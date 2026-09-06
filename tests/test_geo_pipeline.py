import gzip
import json
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.data_prep.geo_pipeline import (
    parse_disease_expression_file,
    parse_geo_directory,
    update_master_nodes_with_geo,
)
from src.data_prep.master_schema import DrugNode, MasterGraphCatalog


@pytest.fixture
def mock_geo_dir(tmp_path):
    """Create synthetic GEO disease expression files."""
    gdir = tmp_path / "GEO"
    gdir.mkdir()

    # Synthetic Alzheimer's GSE5281
    alz_lines = [
        "!Series_title\tBrain_Alzheimers_GSE5281\n",
        "ID_REF\tSAMPLE_1\tSAMPLE_2\tSAMPLE_3\n",
        "CYP2D6\t5.2\t8.9\t4.1\n",
        "PTGS1\t12.1\t14.3\t13.0\n",
        "ADORA2A\t1.1\t1.2\t1.0\n",
    ]
    alz_file = gdir / "Brain_Alzheimers_GSE5281.txt.gz"
    with gzip.open(alz_file, "wt", encoding="utf-8") as f:
        f.writelines(alz_lines)

    # Synthetic Heart Failure GSE57338
    hf_lines = [
        "GENE_SYMBOL\tCTRL_1\tCTRL_2\tHF_1\tHF_2\n",
        "PTGS1\t2.0\t2.1\t9.5\t10.2\n",
        "CYP3A4\t1.5\t1.4\t8.2\t7.9\n",
    ]
    hf_file = gdir / "Cardiovascular_HF_GSE57338.txt"
    hf_file.write_text("".join(hf_lines), encoding="utf-8")

    return gdir


def test_parse_disease_expression_file(mock_geo_dir):
    alz_file = mock_geo_dir / "Brain_Alzheimers_GSE5281.txt.gz"
    sig = parse_disease_expression_file(alz_file)
    assert len(sig) > 0
    assert "CYP2D6" in sig
    assert "PTGS1" in sig
    assert all(0.0 <= val <= 1.0 for val in sig.values())


def test_parse_geo_directory(mock_geo_dir):
    signatures = parse_geo_directory(mock_geo_dir)
    assert len(signatures) == 2
    assert any("Alzheimers" in k for k in signatures.keys())
    assert any("Cardiovascular" in k for k in signatures.keys())


def test_update_master_nodes_with_geo(mock_geo_dir, tmp_path):
    nodes_csv = tmp_path / "master_drug_nodes.csv"
    df_master = pd.DataFrame({
        "drug_id": [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin (PTGS1)
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (ADORA2A)
            "CCO",  # Ethanol (no overlap)
        ],
        "gene_symbols_json": [
            json.dumps(["PTGS1"]),
            json.dumps(["ADORA2A"]),
            json.dumps([]),
        ],
        "bindingdb_targets_json": [
            json.dumps({"PTGS1": 15.0}),
            json.dumps({"ADORA2A": 50.0}),
            json.dumps({}),
        ],
    })
    df_master.to_csv(nodes_csv, index=False)

    updated_df, summary = update_master_nodes_with_geo(
        master_nodes_path=nodes_csv,
        geo_dir_or_signatures=mock_geo_dir,
        output_path=nodes_csv,
    )

    assert summary["nodes_with_geo_signatures"] == 2
    assert summary["total_nodes"] == 3
    assert summary["geo_coverage_pct"] == pytest.approx(66.67, rel=1e-2)

    assert "geo_expression_signatures_json" in updated_df.columns
    assert "geo_signature_vector" in updated_df.columns
    assert "is_geo_active" in updated_df.columns

    # Check active flags
    assert bool(updated_df.loc[updated_df["drug_id"] == "CC(=O)Oc1ccccc1C(=O)O", "is_geo_active"].values[0]) is True
    assert bool(updated_df.loc[updated_df["drug_id"] == "CCO", "is_geo_active"].values[0]) is False


def test_master_graph_catalog_geo_status():
    catalog = MasterGraphCatalog()
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    node = DrugNode(
        drug_id=smi,
        inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        geo_expression_signatures={"Brain_Alzheimers": 0.85},
        is_geo_active=True,
    )
    catalog.add_node(node)
    summary = catalog.summary()
    assert summary["nodes_with_geo_signatures"] == 1
    assert summary["geo_module_status"] == "active"
    assert summary["geo_coverage_pct"] == 100.0


def test_update_master_nodes_geo_keyword_aliases(mock_geo_dir, tmp_path):
    nodes_csv = tmp_path / "master_nodes.csv"
    out_csv = tmp_path / "out_nodes.csv"
    df_master = pd.DataFrame({
        "drug_id": ["CC(=O)Oc1ccccc1C(=O)O"],
        "canonical_smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
        "gene_symbols_json": [json.dumps(["PTGS1"])],
    })
    df_master.to_csv(nodes_csv, index=False)

    updated_df, summary = update_master_nodes_with_geo(
        master_nodes_csv=nodes_csv,
        geo_dir=mock_geo_dir,
        output_csv=out_csv,
        tanimoto_threshold=0.15,
        max_genes=5000,
    )
    assert out_csv.is_file()
    assert summary["nodes_with_geo_signatures"] >= 1

