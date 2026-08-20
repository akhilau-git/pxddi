"""Tests for repeated-seed explanation agreement summaries."""

import pytest

from src.models.explanation_stability import compare_explanation_artifacts


def _artifact(
    raw_probability: float,
    top_atom: int,
    motif_pair: tuple[str, str],
    model_seed: int,
):
    return {
        'method': 'single_component_occlusion_v1',
        'model_architecture': 'cross_attention_edge_aware_gat_v1',
        'model_seed': model_seed,
        'split_seed': 42,
        'splits': {
            'S1': {
                'examples': [{
                    'source': 'CCO',
                    'target': 'CCN',
                    'label': 1,
                    'explanation': {
                        'raw_probability': raw_probability,
                        'drug_a': {
                            'top_atom_occlusions': [{'atom_index': top_atom}],
                            'top_motif_occlusions': [{'motif_name': 'alcohol_or_phenol', 'input_count': 1}],
                        },
                        'drug_b': {
                            'top_atom_occlusions': [{'atom_index': 0}],
                            'top_motif_occlusions': [{'motif_name': 'primary_secondary_amine', 'input_count': 1}],
                        },
                        'cross_drug_attention_associations': {
                            'configured_motif_associations': {
                                'drug_a_to_drug_b': [{
                                    'source_motif': motif_pair[0], 'target_motif': motif_pair[1],
                                }],
                            },
                        },
                    },
                }],
            },
        },
    }


def test_cross_seed_stability_compares_shared_explained_pairs_only():
    report = compare_explanation_artifacts([
        _artifact(0.8, 2, ('alcohol_or_phenol', 'primary_secondary_amine'), 11),
        _artifact(0.7, 2, ('alcohol_or_phenol', 'primary_secondary_amine'), 23),
        _artifact(0.6, 1, ('alcohol_or_phenol', 'primary_secondary_amine'), 37),
    ])

    assert report['shared_explained_pair_count'] == 1
    assert report['pairwise_comparison_count'] == 3
    assert report['model_seeds'] == [11, 23, 37]
    assert report['mean_metrics']['cross_motif_pair_jaccard'] == 1.0
    assert report['mean_metrics']['absolute_raw_probability_difference'] == pytest.approx(2 / 15)
    assert 'does not prove' in report['interpretation_warning']


def test_explanation_stability_refuses_different_architectures():
    left = _artifact(0.8, 2, ('a', 'b'), 11)
    right = _artifact(0.7, 2, ('a', 'b'), 23)
    right['model_architecture'] = 'edge_aware_gat_v2'

    with pytest.raises(ValueError, match='shared model architecture'):
        compare_explanation_artifacts([left, right])


def test_explanation_stability_refuses_duplicate_model_seed():
    with pytest.raises(ValueError, match='distinct model_seed'):
        compare_explanation_artifacts([
            _artifact(0.8, 2, ('a', 'b'), 11),
            _artifact(0.7, 2, ('a', 'b'), 11),
        ])
