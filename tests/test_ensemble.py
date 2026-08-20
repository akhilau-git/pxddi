"""Tests for fixed-split ensemble safety and abstention behaviour."""

import pandas as pd
import pytest

from src.models.ensemble import (
    ABSTENTION_LABEL,
    apply_safe_abstention,
    combine_member_prediction_tables,
    summarize_safe_abstention,
    validate_ensemble_member_manifests,
)
from src.training.run_fixed_split_ensemble import _prediction_path
from src.training.train_full_pipeline_v2 import get_file_hash


def _manifest(model_seed: int, artifact_root: str = 'run'):
    split_manifest = {
        name: {
            'path': f'{artifact_root}/{name}.csv',
            'sha256': f'{name}-hash',
            'rows': 10,
            'label_counts': {'0.0': 5, '1.0': 5},
        }
        for name in ('transductive_train', 'validation', 'transductive_test', 's1_test', 's2_test')
    }
    return {
        'input_sha256': {'twosides_edges': 'twosides-hash'},
        'split_manifest': split_manifest,
        'configuration': {
            'model_architecture': 'edge_aware_gat_v2',
            'feature_schema': 'rich_v2',
            'use_toxicity_pair_features': True,
            'toxicity_loss_weight': 0.3,
            'data_cap': 100,
            'model_seed': model_seed,
            'split_seed': 42,
        },
    }


def test_ensemble_members_require_equal_evidence_but_allow_different_output_paths():
    summary = validate_ensemble_member_manifests([
        _manifest(11, 'member_11'), _manifest(23, 'member_23'), _manifest(37, 'member_37'),
    ])

    assert summary['member_count'] == 3
    assert summary['member_model_seeds'] == [11, 23, 37]
    assert summary['split_seed'] == 42


def test_ensemble_members_reject_a_different_split_hash():
    mismatched = _manifest(23, 'member_23')
    mismatched['split_manifest']['s1_test']['sha256'] = 'different-hash'

    with pytest.raises(ValueError, match='does not match'):
        validate_ensemble_member_manifests([
            _manifest(11, 'member_11'), mismatched, _manifest(37, 'member_37'),
        ])


def test_member_scores_are_only_averaged_after_exact_provenance_alignment():
    base = pd.DataFrame({
        'source': ['CCO', 'CCN'],
        'target': ['CCN', 'CCO'],
        'label': [1, 0],
        'label_evidence': ['reported_twosides', 'unreported_twosides_sampled'],
        'raw_prediction_score': [0.8, 0.2],
    })
    second = base.copy()
    second['raw_prediction_score'] = [0.6, 0.4]
    third = base.copy()
    third['raw_prediction_score'] = [0.7, 0.3]

    combined = combine_member_prediction_tables([base, second, third])

    assert combined['raw_prediction_score'].tolist() == pytest.approx([0.7, 0.3])
    assert combined['ensemble_member_standard_deviation'].tolist() == pytest.approx([
        0.08164966, 0.08164966,
    ])
    reversed_rows = third.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match='provenance differs'):
        combine_member_prediction_tables([base, second, reversed_rows])


def test_safe_abstention_exposes_every_reason_instead_of_forcing_a_label():
    abstention = apply_safe_abstention(
        member_standard_deviation=[0.02, 0.12, 0.01],
        conformal_abstain=[False, False, True],
        structural_ood_flag=[False, True, False],
        standard_deviation_threshold=0.10,
    )
    summary = summarize_safe_abstention(abstention)

    assert abstention['safe_abstain'].tolist() == [False, True, True]
    assert abstention['safe_prediction_status'].tolist()[1] == ABSTENTION_LABEL
    assert 'high_ensemble_member_disagreement' in abstention['safe_abstention_reasons'][1]
    assert 'outside_training_drug_structural_domain' in abstention['safe_abstention_reasons'][1]
    assert summary['safe_abstention_rate'] == pytest.approx(2 / 3)


def test_ensemble_reads_validation_and_test_prediction_manifest_field_names(tmp_path):
    validation_path = tmp_path / 'validation_predictions.csv'
    test_path = tmp_path / 's1_predictions.csv'
    validation_path.write_text('label\n1\n', encoding='utf-8')
    test_path.write_text('label\n0\n', encoding='utf-8')
    manifest = {
        'validation_predictions': {
            'path': str(validation_path), 'sha256': get_file_hash(validation_path),
        },
        'results': {
            'S1': {
                'prediction_path': str(test_path),
                'prediction_sha256': get_file_hash(test_path),
            },
        },
    }

    assert _prediction_path(manifest, 'Validation') == validation_path
    assert _prediction_path(manifest, 'S1') == test_path
