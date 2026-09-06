"""Internal-validation uncertainty summaries for research-only PxDDI runs."""

from __future__ import annotations

from typing import Any

import numpy as np


CONFORMAL_METHOD = 'split_conformal_binary_v1'
PROBABILITY_EPSILON = 1e-6


def _probabilities(values) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError('Probabilities must be one-dimensional.')
    if not np.isfinite(probabilities).all():
        raise ValueError('Probabilities must be finite.')
    return np.clip(probabilities, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)


def predictive_entropy(probabilities) -> np.ndarray:
    """Return Bernoulli entropy in nats; high values mean score ambiguity."""
    values = _probabilities(probabilities)
    return -(values * np.log(values) + (1 - values) * np.log(1 - values))


def fit_split_conformal_binary(
    labels,
    probabilities,
    alpha: float = 0.1,
    fitted_on: str = 'validation_only',
) -> dict[str, Any]:
    """Fit a finite-sample split-conformal threshold on validation data only.

    The resulting prediction set can contain ``no_interaction``,
    ``interaction``, both labels (ambiguous), or neither label.  It has an
    internal marginal-coverage interpretation only under exchangeability; it
    is not clinical confidence and cannot repair biased unreported negatives.
    """
    targets = np.asarray(labels, dtype=int)
    scores = _probabilities(probabilities)
    if len(targets) != len(scores):
        raise ValueError('Labels and probabilities must have equal length.')
    if len(targets) == 0:
        return {
            'status': 'not_fitted_empty_validation',
            'method': CONFORMAL_METHOD,
            'fitted_on': fitted_on,
        }
    if not np.isin(targets, (0, 1)).all():
        raise ValueError('Conformal labels must be binary 0/1 values.')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one.')
    nonconformity = np.where(targets == 1, 1 - scores, scores)
    rank = min(len(nonconformity), max(1, int(np.ceil((len(nonconformity) + 1) * (1 - alpha)))))
    threshold = float(np.partition(nonconformity, rank - 1)[rank - 1])
    return {
        'status': 'fitted',
        'method': CONFORMAL_METHOD,
        'fitted_on': fitted_on,
        'alpha': float(alpha),
        'validation_sample_count': int(len(targets)),
        'quantile_rank': int(rank),
        'nonconformity_threshold': threshold,
        'validation_nonconformity_scores': nonconformity.tolist(),
        'interpretation_warning': (
            'Internal split-conformal sets require exchangeability for their marginal '
            'coverage statement. They are not clinical confidence, do not establish '
            'out-of-distribution safety, and do not make an unreported pair safe.'
        ),
    }


def conformal_prediction_sets(
    probabilities,
    conformal_state: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return binary prediction sets and validation-calibrated p-values."""
    scores = _probabilities(probabilities)
    if conformal_state.get('status') != 'fitted':
        return {
            'no_interaction_p_value': np.full(len(scores), np.nan),
            'interaction_p_value': np.full(len(scores), np.nan),
            'set_size': np.zeros(len(scores), dtype=int),
            'abstain': np.ones(len(scores), dtype=bool),
            'prediction_set': np.asarray(['not_available'] * len(scores), dtype=object),
        }
    calibration_scores = np.asarray(
        conformal_state['validation_nonconformity_scores'], dtype=float
    )
    threshold = float(conformal_state['nonconformity_threshold'])
    score_no_interaction, score_interaction = scores, 1 - scores
    # ``searchsorted`` avoids materializing an impractical test-by-validation
    # matrix for a full TWOSIDES run.
    sorted_calibration_scores = np.sort(calibration_scores)
    def p_value(nonconformity: np.ndarray) -> np.ndarray:
        count_at_least = len(sorted_calibration_scores) - np.searchsorted(
            sorted_calibration_scores, nonconformity, side='left'
        )
        return (count_at_least + 1) / (len(sorted_calibration_scores) + 1)

    no_interaction_p_value = p_value(score_no_interaction)
    interaction_p_value = p_value(score_interaction)
    include_no_interaction = score_no_interaction <= threshold
    include_interaction = score_interaction <= threshold
    prediction_set = np.asarray([
        'no_interaction|interaction' if no and yes
        else 'no_interaction' if no
        else 'interaction' if yes
        else 'empty_set'
        for no, yes in zip(include_no_interaction, include_interaction)
    ], dtype=object)
    set_size = include_no_interaction.astype(int) + include_interaction.astype(int)
    return {
        'no_interaction_p_value': no_interaction_p_value,
        'interaction_p_value': interaction_p_value,
        'set_size': set_size,
        'abstain': set_size != 1,
        'prediction_set': prediction_set,
    }


def summarize_conformal_test_labels(
    labels,
    prediction_sets: dict[str, np.ndarray],
) -> dict[str, float | int | None]:
    """Report observed internal evaluation coverage without hiding abstention."""
    targets = np.asarray(labels, dtype=int)
    sets = prediction_sets['prediction_set']
    if len(targets) != len(sets):
        raise ValueError('Labels and prediction sets must have equal length.')
    if len(targets) == 0:
        return {'sample_count': 0, 'observed_coverage': None, 'abstention_rate': None}
    covered = np.asarray([
        ('no_interaction' in prediction_set.split('|')) if label == 0
        else ('interaction' in prediction_set.split('|'))
        for label, prediction_set in zip(targets, sets)
    ], dtype=bool)
    return {
        'sample_count': int(len(targets)),
        'observed_coverage': float(covered.mean()),
        'abstention_rate': float(np.asarray(prediction_sets['abstain'], dtype=bool).mean()),
    }


def predict_mc_dropout(
    model: Any,
    batch_a: Any,
    batch_b: Any,
    n_passes: int = 20,
    uncertainty_threshold: float = 0.10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run Monte Carlo Dropout forward passes to estimate epistemic model uncertainty.

    Enables dropout during inference and collects N stochastic predictions.
    """
    import torch

    was_training = model.training
    # Enable dropout modules specifically
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    preds: list[float] = []
    try:
        with torch.no_grad():
            for _ in range(n_passes):
                out = model(batch_a, batch_b, **kwargs)
                logits = out[0] if isinstance(out, tuple) else out
                prob = float(torch.sigmoid(logits).reshape(-1)[0].cpu().item())
                preds.append(prob)
    finally:
        model.train(was_training)

    arr = np.array(preds, dtype=float)
    mean_p = float(np.mean(arr))
    var_p = float(np.var(arr))
    std_p = float(np.std(arr))
    ci_lower = float(np.percentile(arr, 2.5))
    ci_upper = float(np.percentile(arr, 97.5))

    return {
        'n_passes': n_passes,
        'mean_probability': mean_p,
        'epistemic_variance': var_p,
        'epistemic_std': std_p,
        'credible_interval_95': (ci_lower, ci_upper),
        'high_uncertainty_flag': bool(std_p >= uncertainty_threshold),
        'sample_predictions': [float(p) for p in preds],
    }

