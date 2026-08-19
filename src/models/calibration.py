"""Saved post-hoc calibration for research-model probability-like scores."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


CALIBRATION_METHOD_PLATT = 'platt_logistic_regression'
SCORE_EPSILON = 1e-6


def _as_probability_array(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError('Scores must be a one-dimensional array.')
    if not np.isfinite(values).all():
        raise ValueError('Scores must contain only finite values.')
    return np.clip(values, SCORE_EPSILON, 1 - SCORE_EPSILON)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    return np.log(probabilities / (1 - probabilities))


def expected_calibration_error(
    labels,
    probabilities,
    bins: int = 10,
) -> float | None:
    """Calculate expected calibration error without conflating it with accuracy."""
    targets = np.asarray(labels, dtype=int)
    scores = _as_probability_array(probabilities)
    if len(targets) == 0:
        return None
    if len(targets) != len(scores):
        raise ValueError('Labels and probabilities must have equal length.')
    if bins < 1:
        raise ValueError('bins must be positive.')

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(float(scores[mask].mean()) - float(targets[mask].mean()))
    return float(total)


def fit_platt_calibrator(labels, raw_probabilities) -> dict[str, Any]:
    """Fit a logistic calibration mapping using validation data only.

    Calibration is explicitly marked as internal-validation calibration. It is
    not evidence of calibrated performance on cold-start or clinical data.
    """
    targets = np.asarray(labels, dtype=int)
    raw_scores = _as_probability_array(raw_probabilities)
    if len(targets) != len(raw_scores):
        raise ValueError('Labels and probabilities must have equal length.')
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return {
            'status': 'not_fitted_insufficient_validation_classes',
            'method': None,
            'fitted_on': 'validation_only',
        }

    logistic_inputs = _logit(raw_scores).reshape(-1, 1)
    estimator = LogisticRegression(solver='lbfgs', random_state=0)
    estimator.fit(logistic_inputs, targets)
    calibrated_scores = estimator.predict_proba(logistic_inputs)[:, 1]
    return {
        'status': 'fitted',
        'method': CALIBRATION_METHOD_PLATT,
        'fitted_on': 'validation_only',
        'input_transform': 'logit_of_clipped_raw_probability',
        'clip_epsilon': SCORE_EPSILON,
        'coefficient': float(estimator.coef_[0, 0]),
        'intercept': float(estimator.intercept_[0]),
        'validation_sample_count': int(len(targets)),
        'validation_brier_raw': float(brier_score_loss(targets, raw_scores)),
        'validation_brier_calibrated': float(brier_score_loss(targets, calibrated_scores)),
        'validation_ece_raw': expected_calibration_error(targets, raw_scores),
        'validation_ece_calibrated': expected_calibration_error(targets, calibrated_scores),
    }


def apply_calibrator(raw_probabilities, calibration: dict[str, Any] | None) -> np.ndarray:
    """Apply a serialized calibrator or return raw scores for legacy artifacts."""
    raw_scores = _as_probability_array(raw_probabilities)
    if not calibration or calibration.get('status') != 'fitted':
        return raw_scores
    if calibration.get('method') != CALIBRATION_METHOD_PLATT:
        raise ValueError(f"Unsupported calibration method: {calibration.get('method')!r}.")
    coefficient = float(calibration['coefficient'])
    intercept = float(calibration['intercept'])
    logits = coefficient * _logit(raw_scores) + intercept
    return 1.0 / (1.0 + np.exp(-logits))
