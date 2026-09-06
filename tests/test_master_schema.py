import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.data_prep.master_schema import (
    DrugNode,
    DDIEdge,
    MasterGraphCatalog,
    canonicalize_smiles,
    smiles_to_inchikey,
)
from src.data_prep.pubchem_bridge import (
    lookup_pubchem_smiles,
    canonicalize,
)


def test_canonicalize_smiles():
    # Aspirin in non-canonical and canonical form
    raw = "O=C(C)Oc1ccccc1C(=O)O"
    canonical = canonicalize_smiles(raw)
    assert canonical == "CC(=O)Oc1ccccc1C(=O)O"
    assert smiles_to_inchikey(raw) == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    # Malformed smiles returns None
    assert canonicalize_smiles("invalid_chemical_string") is None
    assert canonicalize_smiles("") is None
    assert canonicalize_smiles(None) is None


def test_drug_node_validation():
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    inchikey = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    # Valid node
    node = DrugNode(
        drug_id=smi,
        inchikey=inchikey,
        display_name="Aspirin",
        gene_symbols=["CYP2C9", "PTGS1"],
        gene_vector_multihot=[1, 0, 1],
        toxicity_score=0.12,
        n_faers_reports=450,
    )
    assert node.drug_id == smi
    assert node.display_name == "Aspirin"
    assert node.is_bindingdb_active is False  # Inactive expansion module

    # Non-canonical SMILES should fail strict validation
    with pytest.raises(ValueError, match="must be an exact RDKit canonical SMILES"):
        DrugNode(drug_id="O=C(C)Oc1ccccc1C(=O)O", inchikey=inchikey)

    # Invalid InChIKey pattern should fail
    with pytest.raises(ValueError, match="Invalid InChIKey format"):
        DrugNode(drug_id=smi, inchikey="INVALID-INCHIKEY")

    # Negative reports should fail
    with pytest.raises(ValueError, match="cannot be negative"):
        DrugNode(drug_id=smi, inchikey=inchikey, n_faers_reports=-5)


def test_ddi_edge_validation():
    drug_a = "CC(=O)Oc1ccccc1C(=O)O"
    drug_b = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine

    edge = DDIEdge(
        drug_a_id=drug_a,
        drug_b_id=drug_b,
        interaction_type="GI_Bleed",
        evidence_count=15,
        split_group="train",
    )
    assert edge.drug_a_id == drug_a
    assert edge.interaction_type == "GI_Bleed"

    # Self-loops must fail
    with pytest.raises(ValueError, match="Self-loops are forbidden"):
        DDIEdge(drug_a_id=drug_a, drug_b_id=drug_a, interaction_type="GI_Bleed")

    # Invalid split must fail
    with pytest.raises(ValueError, match="Invalid split_group"):
        DDIEdge(drug_a_id=drug_a, drug_b_id=drug_b, interaction_type="test", split_group="bogus_split")


def test_master_graph_catalog():
    catalog = MasterGraphCatalog()
    drug_a = "CC(=O)Oc1ccccc1C(=O)O"
    drug_b = canonicalize_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    node_a = DrugNode(drug_id=drug_a, inchikey=smiles_to_inchikey(drug_a), display_name="Aspirin")
    node_b = DrugNode(drug_id=drug_b, inchikey=smiles_to_inchikey(drug_b), display_name="Caffeine")

    catalog.add_node(node_a)
    catalog.add_node(node_b)

    # Adding edge with valid endpoints
    edge = DDIEdge(drug_a_id=drug_a, drug_b_id=drug_b, interaction_type="Arrhythmia")
    catalog.add_edge(edge)

    summary = catalog.summary()
    assert summary["total_nodes"] == 2
    assert summary["total_edges"] == 1
    assert summary["bindingdb_module_status"] == "inactive_expansion_module"

    # Adding edge with unregistered node should raise KeyError
    unregistered_smi = "CCO"
    with pytest.raises(KeyError, match="not registered in graph nodes"):
        catalog.add_edge(DDIEdge(drug_a_id=drug_a, drug_b_id=unregistered_smi, interaction_type="X"))

    # Export tables test
    with tempfile.TemporaryDirectory() as tmpdir:
        nodes_path, edges_path = catalog.export_tables(tmpdir)
        assert Path(nodes_path).is_file()
        assert Path(edges_path).is_file()


def test_pubchem_bridge_nan_handling():
    # NaN, None, empty string should gracefully return without error
    res_none = lookup_pubchem_smiles(None)
    assert res_none.status == "invalid_or_nan_name"
    assert res_none.smiles is None

    res_nan = lookup_pubchem_smiles("NaN")
    assert res_nan.status == "invalid_or_nan_name"

    res_empty = lookup_pubchem_smiles("   ")
    assert res_empty.status == "invalid_or_nan_name"


def test_build_unified_graph_pipeline(tmp_path):
    import json
    import pandas as pd
    from src.data_prep.build_unified_graph import build_unified_graph

    smi_a = "CC(=O)Oc1ccccc1C(=O)O"
    smi_b = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

    # Mock TWOSIDES edges
    edges_csv = tmp_path / "twosides_mock.csv"
    pd.DataFrame({
        "source": [smi_a, smi_a],
        "target": [smi_b, smi_a],  # includes a self-loop that must be skipped
        "interaction_type": ["Nausea", "Headache"],
    }).to_csv(edges_csv, index=False)

    # Mock PharmGKB gene profiles
    genes_csv = tmp_path / "genes_mock.csv"
    pd.DataFrame({
        "canonical_smiles": [smi_a],
        "genes_list": [json.dumps(["CYP2C9", "PTGS1"])],
    }).to_csv(genes_csv, index=False)

    # Mock FAERS toxicity bridge
    faers_csv = tmp_path / "faers_mock.csv"
    pd.DataFrame({
        "canonical_smiles": [smi_a],
        "toxicity_score": [0.25],
        "n_reports": [1200],
    }).to_csv(faers_csv, index=False)

    out_dir = tmp_path / "unified_output"
    catalog, summary = build_unified_graph(
        twosides_edges_path=edges_csv,
        pharmgkb_profiles_path=genes_csv,
        faers_bridge_path=faers_csv,
        output_dir=out_dir,
        top_k_genes=5,
    )

    assert summary["total_nodes"] == 2
    assert summary["total_edges"] == 1  # self-loop skipped
    assert summary["skipped_self_loops"] == 1
    assert summary["nodes_with_pharmgkb_genes"] == 1
    assert summary["nodes_with_faers_toxicity"] == 1
    assert (out_dir / "master_drug_nodes.csv").is_file()
    assert (out_dir / "master_ddi_edges.csv").is_file()
    assert (out_dir / "gene_vocabulary.json").is_file()


def test_expanded_pharmgkb_bridge(tmp_path):
    from src.data_prep.expanded_pharmgkb_bridge import build_expanded_pharmgkb_profiles

    smi_a = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    smi_b = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # Caffeine

    edges_csv = tmp_path / "twosides_edges.csv"
    pd.DataFrame({"source": [smi_a], "target": [smi_b]}).to_csv(edges_csv, index=False)

    chem_tsv = tmp_path / "chemicals.tsv"
    pd.DataFrame({
        "PharmGKB Accession Id": ["PA448443", "PA448444"],
        "Name": ["Aspirin", "Caffeine"],
        "Generic Names": ["acetylsalicylic acid", "caffeine"],
        "Trade Names": ["Bayer", "No-Doz"],
        "Brand Mixtures": ["", ""],
        "SMILES": [smi_a, smi_b],
    }).to_csv(chem_tsv, sep="\t", index=False)

    rel_tsv = tmp_path / "relationships.tsv"
    pd.DataFrame({
        "Entity1_id": ["PA448443", "PA448444"],
        "Entity1_name": ["Aspirin", "Caffeine"],
        "Entity1_type": ["Chemical", "Chemical"],
        "Entity2_id": ["GENE1", "GENE2"],
        "Entity2_name": ["CYP2C9", "CYP1A2"],
        "Entity2_type": ["Gene", "Gene"],
    }).to_csv(rel_tsv, sep="\t", index=False)

    out_csv = tmp_path / "expanded_profiles.csv"
    df_prof, summary = build_expanded_pharmgkb_profiles(
        twosides_edges_path=edges_csv,
        pharmgkb_chemicals_path=chem_tsv,
        pharmgkb_relationships_path=rel_tsv,
        output_profiles_path=out_csv,
    )

    assert summary["total_twosides_drugs"] == 2
    assert summary["drugs_with_gene_profiles"] == 2
    assert summary["coverage_pct"] == 100.0
    assert "CYP2C9" in summary["top_10_genes"]
    assert "CYP1A2" in summary["top_10_genes"]
    assert out_csv.is_file()


def test_update_master_nodes_with_faers(tmp_path):
    from src.data_prep.build_unified_graph import update_master_nodes_with_faers

    smi_a = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    smi_b = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # Caffeine

    nodes_csv = tmp_path / "master_nodes.csv"
    pd.DataFrame({
        "canonical_smiles": [smi_a, smi_b],
        "toxicity_score": [None, None],
        "n_faers_reports": [None, None],
    }).to_csv(nodes_csv, index=False)

    faers_csv = tmp_path / "faers_bridge.csv"
    pd.DataFrame({
        "canonical_smiles": [smi_a],
        "toxicity_score": [0.42],
        "n_reports": [540],
    }).to_csv(faers_csv, index=False)

    out_csv = tmp_path / "master_nodes_updated.csv"
    df_up, summary = update_master_nodes_with_faers(
        master_nodes_path=nodes_csv,
        faers_bridge_path=faers_csv,
        output_path=out_csv,
    )

    assert summary["previous_faers_coverage"] == 0
    assert summary["updated_faers_coverage"] == 1
    assert summary["coverage_percentage"] == 50.0
    assert out_csv.is_file()
    assert df_up.loc[df_up["canonical_smiles"] == smi_a, "toxicity_score"].iloc[0] == 0.42
    assert df_up.loc[df_up["canonical_smiles"] == smi_a, "n_faers_reports"].iloc[0] == 540



