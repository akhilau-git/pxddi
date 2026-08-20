"""Tests for internal conformal uncertainty and structural-domain flags."""

import numpy as np

from src.models.applicability_domain import MorganApplicabilityDomain
from src.models.uncertainty import (
    conformal_prediction_sets,
    fit_split_conformal_binary,
    predictive_entropy,
    summarize_conformal_test_labels,
)


def test_split_conformal_sets_are_fitted_on_validation_scores_only():
    state = fit_split_conformal_binary(
        labels=[0, 0, 1, 1], probabilities=[0.1, 0.2, 0.8, 0.9], alpha=0.2
    )
    sets = conformal_prediction_sets([0.15, 0.5, 0.85], state)

    assert state['status'] == 'fitted'
    assert state['fitted_on'] == 'validation_only'
    assert sets['prediction_set'].tolist() == [
        'no_interaction', 'empty_set', 'interaction'
    ]
    assert sets['abstain'].tolist() == [False, True, False]
    assert np.all((sets['no_interaction_p_value'] > 0) & (sets['no_interaction_p_value'] <= 1))


def test_conformal_records_its_exact_fit_partition_role():
    state = fit_split_conformal_binary(
        labels=[0, 0, 1, 1], probabilities=[0.1, 0.2, 0.8, 0.9],
        alpha=0.2, fitted_on='validation_conformal_partition',
    )

    assert state['fitted_on'] == 'validation_conformal_partition'


def test_entropy_and_coverage_summary_do_not_hide_abstentions():
    state = fit_split_conformal_binary(
        labels=[0, 1], probabilities=[0.1, 0.9], alpha=0.34
    )
    sets = conformal_prediction_sets([0.1, 0.5], state)
    summary = summarize_conformal_test_labels([0, 1], sets)

    entropy = predictive_entropy([0.01, 0.5, 0.99])
    assert entropy[1] > entropy[0]
    assert summary['sample_count'] == 2
    assert summary['observed_coverage'] == 0.5
    assert summary['abstention_rate'] >= 0


def test_applicability_domain_flags_structurally_distant_drugs_without_claiming_safety():
    domain = MorganApplicabilityDomain(minimum_similarity=0.6)
    summary = domain.fit(['CCO', 'CCN'])
    scores = domain.score_pairs(['CCO', 'c1ccccc1'], ['CCN', 'CCO'])

    assert summary['reference_unique_training_drugs'] == 2
    assert scores['source_exactly_seen_in_training'].tolist() == [True, False]
    assert scores['structural_ood_flag'].tolist() == [False, True]
    assert 'does not' in summary['interpretation_warning']
