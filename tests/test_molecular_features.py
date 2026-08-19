"""Tests for versioned legacy and edge-aware molecular graph schemas."""

import torch
from torch_geometric.data import Batch

from src.data_prep.prepare_twosides import (
    FEATURE_SCHEMA_RICH,
    NUM_ATOM_FEATURES,
    NUM_BOND_FEATURES,
    RICH_NUM_ATOM_FEATURES,
    graph_compatibility_reason,
    smiles_to_graph,
)
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_EDGE_AWARE,
    PxDDIModel,
)


def test_rich_graph_contains_bond_and_stereo_features():
    graph = smiles_to_graph('C/C=C/C', feature_schema=FEATURE_SCHEMA_RICH)

    assert graph is not None
    assert graph.x.shape[1] == RICH_NUM_ATOM_FEATURES
    assert graph.edge_attr.shape == (graph.edge_index.shape[1], NUM_BOND_FEATURES)
    assert graph.x.shape[1] > NUM_ATOM_FEATURES


def test_counterion_only_structures_have_an_explicit_audit_reason():
    assert graph_compatibility_reason('[Na+].[Cl-]') == 'counterion_or_inorganic_only_structure'
    assert graph_compatibility_reason('C') == 'single_atom_or_disconnected_structure'


def test_edge_aware_model_is_symmetric():
    graph_a = smiles_to_graph('C/C=C/C', feature_schema=FEATURE_SCHEMA_RICH)
    graph_b = smiles_to_graph('CC(=O)OC1=CC=CC=C1C(=O)O', feature_schema=FEATURE_SCHEMA_RICH)
    assert graph_a is not None and graph_b is not None
    model = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=16,
        architecture_version=MODEL_ARCHITECTURE_EDGE_AWARE,
        edge_feature_dim=NUM_BOND_FEATURES,
    ).eval()

    with torch.no_grad():
        risk_ab, _, _ = model(Batch.from_data_list([graph_a]), Batch.from_data_list([graph_b]))
        risk_ba, _, _ = model(Batch.from_data_list([graph_b]), Batch.from_data_list([graph_a]))

    assert torch.allclose(risk_ab, risk_ba, atol=1e-6)
