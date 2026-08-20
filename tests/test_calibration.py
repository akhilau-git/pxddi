"""Tests for serialized internal-validation calibration behavior."""

import numpy as np

from src.models.calibration import apply_calibrator, fit_platt_calibrator


def test_platt_calibration_is_serializable_and_returns_probabilities():
    labels = np.array([0, 0, 0, 1, 1, 1])
    raw_scores = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55])

    calibration = fit_platt_calibrator(labels, raw_scores)
    calibrated = apply_calibrator(raw_scores, calibration)

    assert calibration['status'] == 'fitted'
    assert calibration['fitted_on'] == 'validation_only'
    assert calibrated.shape == raw_scores.shape
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert np.all(np.diff(calibrated) > 0)


def test_legacy_or_unfitted_calibration_returns_raw_scores_safely():
    raw_scores = np.array([0.0, 0.5, 1.0])
    returned = apply_calibrator(raw_scores, None)

    assert returned[0] > 0
    assert returned[-1] < 1
    assert returned[1] == 0.5


def test_calibration_records_its_exact_fit_partition_role():
    calibration = fit_platt_calibrator(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9],
        fitted_on='validation_calibration_partition',
    )

    assert calibration['fitted_on'] == 'validation_calibration_partition'
    assert calibration['fitted_partition_sample_count'] == 4
