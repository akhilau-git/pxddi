"""Regression tests for bounded offline candidate explanation artifacts."""

import numpy as np
import torch

from src.data_prep.prepare_twosides import (
    FEATURE_SCHEMA_RICH,
    NUM_BOND_FEATURES,
    RICH_NUM_ATOM_FEATURES,
    smiles_to_graph,
)
from src.models.candidate_explainability import (
    EXPLANATION_METHOD,
    explain_pair_with_occlusion,
    select_representative_indices,
)
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
    MODEL_ARCHITECTURE_EDGE_AWARE,
    PxDDIModel,
)


def _rich_graph(smiles: str):
    graph = smiles_to_graph(smiles, feature_schema=FEATURE_SCHEMA_RICH)
    assert graph is not None
    return graph


def test_occlusion_explanation_reports_raw_score_and_quality_checks():
    graph_a, graph_b = _rich_graph('CCO'), _rich_graph('CCN')
    model = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=8,
        architecture_version=MODEL_ARCHITECTURE_EDGE_AWARE,
        edge_feature_dim=NUM_BOND_FEATURES,
    ).eval()

    explanation = explain_pair_with_occlusion(
        model,
        graph_a,
        graph_b,
        'CCO',
        'CCN',
        top_k=2,
        graph_builder=lambda smiles: smiles_to_graph(
            smiles, feature_schema=FEATURE_SCHEMA_RICH
        ),
    )

    assert explanation['method'] == EXPLANATION_METHOD
    assert 0.0 <= explanation['raw_probability'] <= 1.0
    assert len(explanation['drug_a']['top_atom_occlusions']) == 2
    assert len(explanation['drug_b']['top_bond_occlusions']) == 2
    assert explanation['cross_drug_attention_associations']['available'] is False
    assert explanation['canonical_reencoding_stability']['status'] == 'evaluated'
    assert explanation['symmetry_check']['absolute_raw_probability_difference'] < 1e-6


def test_cross_attention_artifact_exposes_pair_isolated_associations_only():
    graph_a, graph_b = _rich_graph('CCO'), _rich_graph('CCN')
    model = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=8,
        architecture_version=MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
        edge_feature_dim=NUM_BOND_FEATURES,
    ).eval()

    explanation = explain_pair_with_occlusion(
        model, graph_a, graph_b, 'CCO', 'CCN', top_k=2
    )
    associations = explanation['cross_drug_attention_associations']

    assert associations['available'] is True
    assert len(associations['drug_a_to_drug_b']) == 2
    assert len(associations['drug_b_to_drug_a']) == 2
    assert 'not validated' in associations['interpretation_warning']


def test_representative_selection_prioritizes_each_confusion_case_deterministically():
    labels = np.array([1, 0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.8, 0.2, 0.55])

    selected = select_representative_indices(
        labels, predictions, threshold=0.5, maximum_examples=4
    )

    assert selected == [0, 1, 2, 3]


def test_cross_attention_maps_are_normalized_for_each_source_atom():
    graph_a, graph_b = _rich_graph('CCO'), _rich_graph('CCN')
    model = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=8,
        architecture_version=MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
        edge_feature_dim=NUM_BOND_FEATURES,
    ).eval()
    from torch_geometric.data import Batch

    with torch.no_grad():
        a_to_b, b_to_a = model.cross_drug_attention_maps(
            Batch.from_data_list([graph_a]), Batch.from_data_list([graph_b])
        )

    assert torch.allclose(a_to_b[0].sum(dim=1), torch.ones(graph_a.num_nodes))
    assert torch.allclose(b_to_a[0].sum(dim=1), torch.ones(graph_b.num_nodes))
