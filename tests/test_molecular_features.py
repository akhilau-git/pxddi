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
from src.data_prep.molecular_motifs import (
    MOTIF_FEATURE_DIM,
    MOTIF_FEATURE_NAMES,
    motif_count_vector_from_smiles,
)
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE,
    PxDDIModel,
    model_from_checkpoint,
)
from src.models.encoder import CrossDrugAttention


def test_rich_graph_contains_bond_and_stereo_features():
    graph = smiles_to_graph('C/C=C/C', feature_schema=FEATURE_SCHEMA_RICH)

    assert graph is not None
    assert graph.x.shape[1] == RICH_NUM_ATOM_FEATURES
    assert graph.edge_attr.shape == (graph.edge_index.shape[1], NUM_BOND_FEATURES)
    assert graph.x.shape[1] > NUM_ATOM_FEATURES


def test_toxicity_head_returns_logits_but_risk_features_remain_probabilities():
    graph_a = smiles_to_graph('CCO')
    graph_b = smiles_to_graph('CCN')
    assert graph_a is not None and graph_b is not None
    batch_a = Batch.from_data_list([graph_a])
    batch_b = Batch.from_data_list([graph_b])
    model = PxDDIModel(in_channels=NUM_ATOM_FEATURES, hidden_channels=8).eval()

    with torch.no_grad():
        embedding_a = model.encoder(batch_a.x, batch_a.edge_index, batch_a.batch)
        embedding_b = model.encoder(batch_b.x, batch_b.edge_index, batch_b.batch)
        expected_toxicity_a_logits = model.toxicity_head(embedding_a)
        expected_toxicity_b_logits = model.toxicity_head(embedding_b)
        expected_risk_features = torch.cat((
            embedding_a + embedding_b,
            torch.abs(embedding_a - embedding_b),
            (
                torch.sigmoid(expected_toxicity_a_logits)
                + torch.sigmoid(expected_toxicity_b_logits)
            ).unsqueeze(-1),
            torch.abs(
                torch.sigmoid(expected_toxicity_a_logits)
                - torch.sigmoid(expected_toxicity_b_logits)
            ).unsqueeze(-1),
        ), dim=1)
        expected_risk = model.risk_classifier(expected_risk_features).squeeze(-1)
        risk, toxicity_a_logits, toxicity_b_logits = model(batch_a, batch_b)

    assert torch.allclose(toxicity_a_logits, expected_toxicity_a_logits)
    assert torch.allclose(toxicity_b_logits, expected_toxicity_b_logits)
    assert torch.allclose(risk, expected_risk)


def test_counterion_only_structures_have_an_explicit_audit_reason():
    assert graph_compatibility_reason('[Na+].[Cl-]') == 'counterion_or_inorganic_only_structure'
    assert graph_compatibility_reason('C') == 'single_atom_or_disconnected_structure'


def test_motif_counts_are_fixed_order_and_capture_known_aspirin_groups():
    features = motif_count_vector_from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')

    assert features.shape == (MOTIF_FEATURE_DIM,)
    assert features[MOTIF_FEATURE_NAMES.index('carboxylic_acid')] == 1
    assert features[MOTIF_FEATURE_NAMES.index('ester')] == 1
    assert features[MOTIF_FEATURE_NAMES.index('aromatic_ring')] >= 1
    assert features[MOTIF_FEATURE_NAMES.index('carbonyl')] == 2


def test_rich_graph_can_carry_motif_features():
    graph = smiles_to_graph(
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        feature_schema=FEATURE_SCHEMA_RICH,
        include_motif_features=True,
    )

    assert graph is not None
    assert graph.motif_features.shape == (1, MOTIF_FEATURE_DIM)


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


def test_motif_edge_aware_model_is_symmetric():
    graph_a = smiles_to_graph(
        'C/C=C/C', feature_schema=FEATURE_SCHEMA_RICH, include_motif_features=True
    )
    graph_b = smiles_to_graph(
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        feature_schema=FEATURE_SCHEMA_RICH,
        include_motif_features=True,
    )
    assert graph_a is not None and graph_b is not None
    model = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=16,
        architecture_version=MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE,
        edge_feature_dim=NUM_BOND_FEATURES,
        motif_feature_dim=MOTIF_FEATURE_DIM,
        motif_hidden_channels=8,
    ).eval()

    with torch.no_grad():
        risk_ab, _, _ = model(Batch.from_data_list([graph_a]), Batch.from_data_list([graph_b]))
        risk_ba, _, _ = model(Batch.from_data_list([graph_b]), Batch.from_data_list([graph_a]))

    assert torch.allclose(risk_ab, risk_ba, atol=1e-6)


def test_cross_drug_attention_isolated_from_other_pairs_in_the_batch():
    attention = CrossDrugAttention(hidden_channels=4).eval()
    pair_a = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]])
    pair_b = torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.4, 0.3, 0.2, 0.1]])
    unrelated_a = torch.full((2, 4), 10.0)
    unrelated_b = torch.full((3, 4), -10.0)

    isolated_a, isolated_b = attention(
        pair_a, torch.tensor([0, 0]), pair_b, torch.tensor([0, 0])
    )
    batched_a, batched_b = attention(
        torch.cat((pair_a, unrelated_a)),
        torch.tensor([0, 0, 1, 1]),
        torch.cat((pair_b, unrelated_b)),
        torch.tensor([0, 0, 1, 1, 1]),
    )

    assert torch.allclose(isolated_a[0], batched_a[0], atol=1e-6)
    assert torch.allclose(isolated_b[0], batched_b[0], atol=1e-6)


def test_cross_attention_edge_aware_model_is_symmetric():
    graph_a = smiles_to_graph('C/C=C/C', feature_schema=FEATURE_SCHEMA_RICH)
    graph_b = smiles_to_graph(
        'CC(=O)OC1=CC=CC=C1C(=O)O', feature_schema=FEATURE_SCHEMA_RICH
    )
    assert graph_a is not None and graph_b is not None
    model = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=16,
        architecture_version=MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
        edge_feature_dim=NUM_BOND_FEATURES,
    ).eval()

    with torch.no_grad():
        risk_ab, _, _ = model(Batch.from_data_list([graph_a]), Batch.from_data_list([graph_b]))
        risk_ba, _, _ = model(Batch.from_data_list([graph_b]), Batch.from_data_list([graph_a]))

    assert torch.allclose(risk_ab, risk_ba, atol=1e-6)


def test_cross_attention_checkpoint_recreates_and_loads_its_architecture():
    source_model = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=8,
        architecture_version=MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
        edge_feature_dim=NUM_BOND_FEATURES,
    ).eval()
    checkpoint = {
        'model_state_dict': source_model.state_dict(),
        'in_channels': RICH_NUM_ATOM_FEATURES,
        'hidden_channels': 8,
        'architecture_version': MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
        'edge_feature_dim': NUM_BOND_FEATURES,
        'use_toxicity_pair_features': True,
    }

    restored_model = model_from_checkpoint(checkpoint).eval()
    restored_model.load_state_dict(checkpoint['model_state_dict'])

    assert restored_model.cross_drug_attention is not None
    for name, value in source_model.state_dict().items():
        assert torch.equal(restored_model.state_dict()[name], value), name
