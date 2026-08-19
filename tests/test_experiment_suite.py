"""Tests for controlled experiment-suite aggregation."""

import pytest

from src.training.run_experiment_suite import bootstrap_mean_confidence_interval


def test_bootstrap_confidence_interval_requires_repeated_runs():
    assert bootstrap_mean_confidence_interval([0.7]) is None


def test_bootstrap_confidence_interval_records_mean_and_bounds():
    summary = bootstrap_mean_confidence_interval([0.5, 0.7, 0.9], seed=1, resamples=100)

    assert summary is not None
    assert summary['run_count'] == 3
    assert summary['mean'] == pytest.approx(0.7)
    assert summary['ci_95_lower'] <= summary['mean'] <= summary['ci_95_upper']
