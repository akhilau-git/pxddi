"""Tests for controlled experiment-suite aggregation."""

import pandas as pd
import pytest

from src.training.run_experiment_suite import (
    bootstrap_mean_confidence_interval,
    paired_bootstrap_difference_confidence_interval,
    paired_comparison_summary,
    selected_experiments,
    split_manifest_signature,
    validate_study_comparability,
)


def test_bootstrap_confidence_interval_requires_repeated_runs():
    assert bootstrap_mean_confidence_interval([0.7]) is None


def test_bootstrap_confidence_interval_records_mean_and_bounds():
    summary = bootstrap_mean_confidence_interval([0.5, 0.7, 0.9], seed=1, resamples=100)

    assert summary is not None
    assert summary['run_count'] == 3
    assert summary['mean'] == pytest.approx(0.7)
    assert summary['ci_95_lower'] <= summary['mean'] <= summary['ci_95_upper']


def test_paired_confidence_interval_requires_matched_repeated_runs():
    assert paired_bootstrap_difference_confidence_interval([0.7], [0.8]) is None

    summary = paired_bootstrap_difference_confidence_interval(
        [0.5, 0.7, 0.8], [0.6, 0.8, 0.9], seed=1, resamples=100
    )

    assert summary is not None
    assert summary['matched_seed_count'] == 3
    assert summary['mean_difference_candidate_minus_reference'] == pytest.approx(0.1)
    assert summary['ci_95_lower'] <= 0.1 <= summary['ci_95_upper']


def _split_manifest(seed_suffix: str = 'a'):
    return {
        name: {
            'sha256': f'{seed_suffix}_{name}',
            'rows': 10,
            'label_counts': {'0.0': 5, '1.0': 5},
        }
        for name in ('transductive_train', 'validation', 'transductive_test', 's1_test', 's2_test')
    }


def test_split_manifest_signature_changes_when_split_evidence_changes():
    first = split_manifest_signature({'split_manifest': _split_manifest('first')})
    second = split_manifest_signature({'split_manifest': _split_manifest('second')})

    assert len(first) == 64
    assert first != second


def test_study_comparability_rejects_models_with_different_splits():
    table = pd.DataFrame([
        {
            'experiment': 'legacy', 'seed': 42, 'split': 'S1',
            'twosides_input_sha256': 'same', 'split_manifest_signature': 'split_a',
            'negative_label_meaning': 'unreported_twosides_sampled',
        },
        {
            'experiment': 'candidate', 'seed': 42, 'split': 'S1',
            'twosides_input_sha256': 'same', 'split_manifest_signature': 'split_b',
            'negative_label_meaning': 'unreported_twosides_sampled',
        },
    ])

    with pytest.raises(ValueError, match='split_manifest_signature differs'):
        validate_study_comparability(table)


def test_paired_summary_uses_only_matched_seed_metrics():
    rows = []
    for experiment, offset in (('legacy_gat_multitask', 0.0), ('edge_aware_multitask', 0.1)):
        for seed in (11, 23, 37):
            rows.append({
                'experiment': experiment,
                'seed': seed,
                'split': 'S1',
                'auroc': 0.5 + offset,
                'average_precision': 0.5 + offset,
                'f1': 0.5 + offset,
                'mcc': 0.1 + offset,
                'brier_score_calibrated': 0.4 - offset,
            })

    summary = paired_comparison_summary(pd.DataFrame(rows), 'legacy_gat_multitask')
    auroc = next(
        item for item in summary['comparisons']
        if item['candidate_experiment'] == 'edge_aware_multitask' and item['metric'] == 'auroc'
    )

    assert auroc['matched_seeds'] == [11, 23, 37]
    assert auroc['statistics']['mean_difference_candidate_minus_reference'] == pytest.approx(0.1)


def test_paper_preset_default_experiment_subset_is_explicit(monkeypatch):
    monkeypatch.setattr('src.training.run_experiment_suite.PRESET', 'paper')

    assert [item['name'] for item in selected_experiments('')] == [
        'legacy_gat_multitask', 'edge_aware_multitask'
    ]


def test_ecfp_baseline_can_be_selected_explicitly():
    selected = selected_experiments('ecfp_sgd_logistic')

    assert selected[0]['runner'] == 'ecfp_sgd_logistic'
    assert selected[0]['architecture'] == 'ecfp_sgd_logistic_v1'
