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
    render_occlusion_svg,
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
    graph_a = _rich_graph('CC(=O)OC1=CC=CC=C1C(=O)O')  # aspirin
    graph_b = _rich_graph('CC(=O)NC1=CC=C(O)C=C1')  # acetaminophen
    model = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=8,
        architecture_version=MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
        edge_feature_dim=NUM_BOND_FEATURES,
    ).eval()

    explanation = explain_pair_with_occlusion(
        model,
        graph_a,
        graph_b,
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CC(=O)NC1=CC=C(O)C=C1',
        top_k=2,
    )
    associations = explanation['cross_drug_attention_associations']

    assert associations['available'] is True
    assert len(associations['drug_a_to_drug_b']) == 2
    assert len(associations['drug_b_to_drug_a']) == 2
    assert 'not validated' in associations['interpretation_warning']
    motif_associations = associations['configured_motif_associations']
    assert motif_associations['available'] is True
    assert motif_associations['drug_a_to_drug_b']
    assert 'not validated' in motif_associations['interpretation_warning']


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


def test_occlusion_svg_is_a_vector_artifact_with_atom_indices(tmp_path):
    destination = tmp_path / 'ethanol_occlusion.svg'
    render_occlusion_svg(
        'CCO',
        [{'atom_index': 2, 'raw_probability_change': 0.2}],
        [{'bond_atom_indices': [1, 2], 'raw_probability_change': -0.1}],
        destination,
    )

    content = destination.read_text(encoding='utf-8')
    assert content.lstrip().startswith('<?xml')
    assert '<svg' in content


def test_explain_multimodal_pair():
    from src.models.candidate_explainability import explain_multimodal_pair
    from src.models.ddi_model import MODEL_ARCHITECTURE_MULTIMODAL

    graph_a, graph_b = _rich_graph('CCO'), _rich_graph('CCN')
    model = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=16,
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        edge_feature_dim=NUM_BOND_FEATURES,
        gene_feature_dim=10,
        gene_hidden_channels=16,
        use_clinical_toxicity=True,
        use_cross_modal_attention=True,
    ).eval()

    fp_a = torch.zeros(1024)
    fp_b = torch.zeros(1024)
    gene_a = torch.tensor([1.0 if i in {0, 2} else 0.0 for i in range(10)])
    gene_b = torch.tensor([1.0 if i in {2, 4} else 0.0 for i in range(10)])
    vocab = [f"GENE_{i}" for i in range(10)]

    exp = explain_multimodal_pair(
        model=model,
        graph_a=graph_a,
        graph_b=graph_b,
        smiles_a='CCO',
        smiles_b='CCN',
        fp_a=fp_a,
        fp_b=fp_b,
        gene_a=gene_a,
        gene_b=gene_b,
        gene_mask_a=torch.tensor(1.0),
        gene_mask_b=torch.tensor(1.0),
        tox_a=torch.tensor(0.35),
        tox_b=torch.tensor(0.20),
        gene_vocabulary=vocab,
    )

    assert 'predicted_raw_probability' in exp
    assert 'modality_marginal_contributions' in exp
    assert 'overall_risk_score' in exp
    assert 'modality_contributions' in exp
    assert 'shared_pharmacogenomic_genes' in exp
    assert 'GENE_2' in exp['pharmacogenomic_context']['shared_cyp_competition']
    assert exp['pharmacogenomic_context']['potential_metabolic_bottleneck'] is True


def test_explain_multimodal_pair_with_cache():
    from src.data_prep.cached_graph_loader import MolecularCache
    from src.models.candidate_explainability import explain_multimodal_pair
    from src.models.ddi_model import MODEL_ARCHITECTURE_MULTIMODAL

    cache = MolecularCache(gene_dim=10)
    drug_a = 'CCO'
    drug_b = 'CCN'
    cache.register_drug(drug_a, gene_vector=[1.0 if i in {0, 2} else 0.0 for i in range(10)], toxicity_score=0.35)
    cache.register_drug(drug_b, gene_vector=[1.0 if i in {2, 4} else 0.0 for i in range(10)], toxicity_score=0.20)

    model = PxDDIModel(
        in_channels=cache.graphs[drug_a].x.size(1),
        hidden_channels=16,
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        edge_feature_dim=cache.graphs[drug_a].edge_attr.size(1),
        gene_feature_dim=10,
        gene_hidden_channels=16,
        use_clinical_toxicity=True,
        use_cross_modal_attention=True,
    ).eval()

    vocab = [f"GENE_{i}" for i in range(10)]

    exp = explain_multimodal_pair(
        model=model,
        cache=cache,
        drug_a_smiles=drug_a,
        drug_b_smiles=drug_b,
        gene_names=vocab,
        mc_dropout=True,
        n_mc_passes=5,
    )

    assert 'overall_risk_score' in exp
    assert 'modality_contributions' in exp
    assert len(exp['shared_pharmacogenomic_genes']) == 1
    assert exp['shared_pharmacogenomic_genes'][0]['gene'] == 'GENE_2'
    assert 'uncertainty' in exp
    assert 'epistemic_std' in exp
    assert exp['epistemic_std'] >= 0.0


