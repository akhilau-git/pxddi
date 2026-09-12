"""Unit tests for the Biophysical and Pharmacokinetic Engine."""

import pytest
import numpy as np
from src.data_prep.biophysical_engine import (
    CYP_ENZYMES,
    BIOPHYSICAL_DIM,
    compute_cyp_affinities,
    compute_admet_pharmacokinetics,
    compute_biophysical_vector,
    compute_metabolic_collision_score,
    identify_site_of_metabolism,
)
from rdkit import Chem


def test_cyp_pharmacophore_profiling():
    # Aspirin: Weakly acidic carboxylate -> should show strong CYP2C9 liability
    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    cyp_aspirin = compute_cyp_affinities(aspirin)
    assert cyp_aspirin["CYP2C9"] > 0.4
    assert len(cyp_aspirin) == len(CYP_ENZYMES)

    # Metoprolol: Basic aliphatic amine + aromatic ether -> strong CYP2D6 liability
    metoprolol = Chem.MolFromSmiles("CC(C)NCC(O)COc1ccc(CCOCC)cc1")
    cyp_metoprolol = compute_cyp_affinities(metoprolol)
    assert cyp_metoprolol["CYP2D6"] > 0.5

    # Caffeine: Flat planar purine heteroaromatic -> strong CYP1A2 liability
    caffeine = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(c(=O)n2C)C")
    cyp_caffeine = compute_cyp_affinities(caffeine)
    assert cyp_caffeine["CYP1A2"] > 0.4

    # Ketoconazole: Bulky azole multi-ring lipophile -> strong CYP3A4 liability
    ketoconazole = Chem.MolFromSmiles("CC(=O)N1CCN(CC1)c2ccc(cc2)OCC3COC(O3)(Cn4cncn4)c5ccc(cc5)Cl")
    cyp_keto = compute_cyp_affinities(ketoconazole)
    assert cyp_keto["CYP3A4"] > 0.6


def test_admet_pharmacokinetics():
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smi)
    pk = compute_admet_pharmacokinetics(mol)

    assert "p_eff" in pk
    assert "f_unbound" in pk
    assert 0.0 <= pk["p_eff"] <= 1.0
    assert 0.0 < pk["f_unbound"] < 1.0


def test_biophysical_vector_shape():
    smi = "CC(=O)Nc1ccc(O)cc1"  # Paracetamol
    vec = compute_biophysical_vector(smi)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (BIOPHYSICAL_DIM,)
    assert vec.dtype == np.float32

    # Malformed SMILES returns zeroes
    vec_bad = compute_biophysical_vector("invalid_string")
    assert vec_bad.shape == (BIOPHYSICAL_DIM,)
    assert np.all(vec_bad == 0.0)


def test_metabolic_collision_symmetry():
    smi_a = "CC(C)NCC(O)COc1ccc(CCOCC)cc1"  # Metoprolol (CYP2D6)
    smi_b = "CNCCC(c1ccccc1)Oc2ccc(C(F)(F)F)cc2"  # Fluoxetine (CYP2D6 inhibitor)

    res_ab = compute_metabolic_collision_score(smi_a, smi_b)
    res_ba = compute_metabolic_collision_score(smi_b, smi_a)

    # Collision index must be strictly symmetric: collision(A, B) == collision(B, A)
    assert abs(res_ab["collision_index"] - res_ba["collision_index"]) < 1e-6
    assert abs(res_ab["cyp_overlap"] - res_ba["cyp_overlap"]) < 1e-6
    assert abs(res_ab["displacement_risk"] - res_ba["displacement_risk"]) < 1e-6
    assert res_ab["dominant_cyp"] == "CYP2D6"


def test_site_of_metabolism():
    smi = "CN(C)CCCN1c2ccccc2Sc3ccccc13"  # Promethazine (N-methyl groups)
    som = identify_site_of_metabolism(smi)
    assert len(som) > 0
    motifs = [entry["motif"] for entry in som]
    assert any("dealkylation" in m or "aliphatic" in m for m in motifs)


def test_biophysical_model_integration_and_symmetry():
    import torch
    from src.models.ddi_model import PxDDIModel, MODEL_ARCHITECTURE_EDGE_AWARE
    from src.data_prep.prepare_twosides import smiles_to_graph, FEATURE_SCHEMA_RICH

    smi_a = "CC(=O)Oc1ccccc1C(=O)O"
    smi_b = "CC(C)NCC(O)COc1ccc(CCOCC)cc1"
    g_a = smiles_to_graph(smi_a, feature_schema=FEATURE_SCHEMA_RICH)
    g_b = smiles_to_graph(smi_b, feature_schema=FEATURE_SCHEMA_RICH)
    assert g_a is not None and g_b is not None
    assert g_a.x is not None and g_a.edge_attr is not None

    in_dim = g_a.x.size(1)
    edge_dim = g_a.edge_attr.size(1)

    model = PxDDIModel(
        in_channels=in_dim,
        edge_feature_dim=edge_dim,
        architecture_version=MODEL_ARCHITECTURE_EDGE_AWARE,
        hidden_channels=32,
        use_biophysical_features=True,
        biophysical_dim=BIOPHYSICAL_DIM,
        biophysical_hidden_channels=16,
    )
    model.eval()

    vec_a = torch.from_numpy(compute_biophysical_vector(smi_a)).unsqueeze(0)
    vec_b = torch.from_numpy(compute_biophysical_vector(smi_b)).unsqueeze(0)

    # Forward pass A -> B
    with torch.no_grad():
        out_ab, _, _ = model(g_a, g_b, biophysical_a=vec_a, biophysical_b=vec_b)
        out_ba, _, _ = model(g_b, g_a, biophysical_a=vec_b, biophysical_b=vec_a)

    # Verify risk score symmetry
    assert abs(out_ab.item() - out_ba.item()) < 1e-4

    # Verify graceful fallback when biophysical features are omitted
    with torch.no_grad():
        out_fallback, _, _ = model(g_a, g_b)
        assert torch.isfinite(out_fallback)


def test_clinical_audit_report_generation():
    from src.evaluation.clinical_audit_report import generate_clinical_audit_report

    # Known severe interaction: Fluoxetine (CYP2D6 inhibitor) + Metoprolol (CYP2D6 substrate)
    smi_a = "CNCCC(c1ccccc1)Oc2ccc(C(F)(F)F)cc2"
    smi_b = "CC(C)NCC(O)COc1ccc(CCOCC)cc1"

    report = generate_clinical_audit_report(smi_a, smi_b)

    assert "drug_a" in report
    assert "drug_b" in report
    assert "metabolic_collision" in report
    assert "risk" in report
    assert "decision" in report
    assert "markdown" in report

    # Verify that CYP2D6 is flagged as colliding enzyme
    assert report["metabolic_collision"]["dominant_cyp"] == "CYP2D6"
    assert report["metabolic_collision"]["collision_index"] > 0.35
    assert len(report["markdown"]) > 100
    assert "LC-MS/MS" in report["decision"]["analytical_equipment"]


