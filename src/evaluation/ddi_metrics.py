"""Transparent binary-DDI evaluation utilities.

The helpers in this module intentionally distinguish ranking, thresholded
decision, calibration, and selective-prediction questions.  PxDDI's negative
examples are *unreported* pairs sampled from TWOSIDES; therefore these
quantities describe the experimental label task, not clinical safety or
clinical benefit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The main training script is deliberately executable as
# ``python src/training/train_full_pipeline_v2.py`` in Colab.  In that mode it
# places ``src/`` on ``sys.path`` and loads this module as ``evaluation``.
# Tests and package-style callers instead load it as ``src.evaluation``.  Keep
# the calibration import compatible with both supported entry points.
try:
    from models.calibration import expected_calibration_error
except ModuleNotFoundError:  # Package-style import used by tests/tools.
    from src.models.calibration import expected_calibration_error


DEFAULT_BOOTSTRAP_METRICS = (
    'auroc',
    'average_precision',
    'f1',
    'mcc',
    'balanced_accuracy',
    'brier_score_calibrated',
)


def _arrays(
    labels,
    predictions,
    raw_predictions=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return labels, final scores, and raw scores."""
    targets = np.asarray(labels, dtype=int)
    scores = np.asarray(predictions, dtype=float)
    raw_scores = scores if raw_predictions is None else np.asarray(raw_predictions, dtype=float)
    if targets.ndim != 1 or scores.ndim != 1 or raw_scores.ndim != 1:
        raise ValueError('Labels and prediction scores must be one-dimensional.')
    if len(targets) != len(scores) or len(targets) != len(raw_scores):
        raise ValueError('Labels, final scores, and raw scores must have equal length.')
    if not np.isin(targets, (0, 1)).all():
        raise ValueError('Binary evaluation labels must contain only 0 and 1.')
    if not np.isfinite(scores).all() or not np.isfinite(raw_scores).all():
        raise ValueError('Prediction scores must be finite.')
    if not ((0 <= scores).all() and (scores <= 1).all()):
        raise ValueError('Final prediction scores must lie between 0 and 1.')
    if not ((0 <= raw_scores).all() and (raw_scores <= 1).all()):
        raise ValueError('Raw prediction scores must lie between 0 and 1.')
    return targets, scores, raw_scores


def calibration_slope_intercept_diagnostic(
    labels,
    probabilities,
) -> dict[str, float | None]:
    """Return post-hoc calibration slope/intercept diagnostics.

    These are calculated on the supplied labelled partition and are descriptive
    only.  They must never be used to select a model or recalibrate that same
    partition.  A slope near one and an intercept near zero are desirable, but
    neither establishes cold-start or clinical calibration.
    """
    targets, scores, _ = _arrays(labels, probabilities)
    if len(targets) == 0 or len(np.unique(targets)) < 2 or np.ptp(scores) == 0:
        return {'calibration_slope_diagnostic': None, 'calibration_intercept_diagnostic': None}
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(solver='lbfgs', random_state=0).fit(logits, targets)
    except ValueError:
        return {'calibration_slope_diagnostic': None, 'calibration_intercept_diagnostic': None}
    return {
        'calibration_slope_diagnostic': float(model.coef_[0, 0]),
        'calibration_intercept_diagnostic': float(model.intercept_[0]),
    }


def calculate_binary_metrics(
    labels,
    predictions,
    threshold: float,
    raw_predictions=None,
    include_calibration_diagnostics: bool = True,
) -> dict[str, Any]:
    """Calculate complementary ranking, decision, and calibration metrics.

    Accuracy is included only for comparison with prior papers.  AUROC,
    average precision, MCC, calibration, and confidence intervals are the
    primary evidence because PxDDI uses sampled unreported negatives.
    """
    if not 0 <= float(threshold) <= 1:
        raise ValueError('Threshold must lie between 0 and 1.')
    targets, scores, raw_scores = _arrays(labels, predictions, raw_predictions)
    predicted_labels = (scores >= threshold).astype(int)
    matrix = confusion_matrix(targets, predicted_labels, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    has_rows = len(targets) > 0
    has_two_classes = len(np.unique(targets)) == 2
    negative_denominator = true_negative + false_positive
    positive_denominator = true_positive + false_negative
    prediction_negative_denominator = true_negative + false_negative
    result: dict[str, Any] = {
        'sample_count': int(len(targets)),
        'positive_count': int((targets == 1).sum()),
        'negative_count': int((targets == 0).sum()),
        'positive_prevalence': float(targets.mean()) if has_rows else None,
        'threshold': float(threshold),
        'confusion_matrix': matrix.tolist(),
        'accuracy': float(accuracy_score(targets, predicted_labels)) if has_rows else None,
        'balanced_accuracy': float(balanced_accuracy_score(targets, predicted_labels))
        if has_two_classes else None,
        'f1': float(f1_score(targets, predicted_labels, zero_division=0)) if has_rows else None,
        'precision': float(precision_score(targets, predicted_labels, zero_division=0))
        if has_rows else None,
        'recall': float(recall_score(targets, predicted_labels, zero_division=0)) if has_rows else None,
        'sensitivity': float(true_positive / positive_denominator) if positive_denominator else None,
        'specificity': float(true_negative / negative_denominator) if negative_denominator else None,
        'negative_predictive_value': float(true_negative / prediction_negative_denominator)
        if prediction_negative_denominator else None,
        'false_positive_rate': float(false_positive / negative_denominator)
        if negative_denominator else None,
        'false_negative_rate': float(false_negative / positive_denominator)
        if positive_denominator else None,
        'mcc': float(matthews_corrcoef(targets, predicted_labels)) if has_two_classes else None,
        'brier_score_raw': float(brier_score_loss(targets, raw_scores)) if has_rows else None,
        'brier_score_calibrated': float(brier_score_loss(targets, scores)) if has_rows else None,
        'ece_raw': expected_calibration_error(targets, raw_scores),
        'ece_calibrated': expected_calibration_error(targets, scores),
    }
    if include_calibration_diagnostics:
        result.update(calibration_slope_intercept_diagnostic(targets, scores))
    else:
        result.update({
            'calibration_slope_diagnostic': None,
            'calibration_intercept_diagnostic': None,
        })
    if not has_two_classes:
        result.update({
            'status': 'skipped_one_class_or_empty_split',
            'auroc': None,
            'average_precision': None,
        })
        return result
    result.update({
        'status': 'evaluated',
        'auroc': float(roc_auc_score(targets, scores)),
        'average_precision': float(average_precision_score(targets, scores)),
    })
    return result


def bootstrap_confidence_intervals(
    labels,
    predictions,
    threshold: float,
    raw_predictions=None,
    metric_names: Iterable[str] = DEFAULT_BOOTSTRAP_METRICS,
    resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Return stratified, test-partition bootstrap 95% confidence intervals.

    Resampling separately within the reported and unreported classes preserves
    the experimental class composition.  This quantifies test-partition sample
    uncertainty; it is deliberately separate from the repeated-seed intervals
    written by ``run_experiment_suite.py``.
    """
    if resamples < 0:
        raise ValueError('Bootstrap resamples must be zero or a positive integer.')
    targets, scores, raw_scores = _arrays(labels, predictions, raw_predictions)
    names = tuple(metric_names)
    if not names:
        raise ValueError('At least one bootstrap metric is required.')
    available = set(calculate_binary_metrics(targets, scores, threshold, raw_scores))
    unknown = set(names).difference(available)
    if unknown:
        raise ValueError(f'Unsupported bootstrap metrics: {sorted(unknown)}.')
    if resamples == 0:
        return {
            'status': 'disabled',
            'method': 'stratified_nonparametric_bootstrap',
            'resamples': 0,
            'metrics': {name: None for name in names},
        }
    negative_indices = np.flatnonzero(targets == 0)
    positive_indices = np.flatnonzero(targets == 1)
    if not len(negative_indices) or not len(positive_indices):
        return {
            'status': 'skipped_one_class_or_empty_split',
            'method': 'stratified_nonparametric_bootstrap',
            'resamples': int(resamples),
            'metrics': {name: None for name in names},
        }
    generator = np.random.default_rng(seed)
    values = {name: [] for name in names}
    for _ in range(resamples):
        sampled_indices = np.concatenate((
            generator.choice(negative_indices, size=len(negative_indices), replace=True),
            generator.choice(positive_indices, size=len(positive_indices), replace=True),
        ))
        sampled = calculate_binary_metrics(
            targets[sampled_indices],
            scores[sampled_indices],
            threshold,
            raw_scores[sampled_indices],
            include_calibration_diagnostics=False,
        )
        for name in names:
            value = sampled[name]
            if value is not None and np.isfinite(value):
                values[name].append(float(value))
    point_estimate = calculate_binary_metrics(targets, scores, threshold, raw_scores)
    return {
        'status': 'evaluated',
        'method': 'stratified_nonparametric_bootstrap',
        'confidence_level': 0.95,
        'resamples': int(resamples),
        'seed': int(seed),
        'stratification': 'experimental_binary_label',
        'interpretation_warning': (
            'These are within-split sample confidence intervals. They do not replace '
            'independent repeated-seed experiments or external validation.'
        ),
        'metrics': {
            name: {
                'point_estimate': point_estimate[name],
                'valid_resamples': int(len(values[name])),
                'ci_95_lower': float(np.percentile(values[name], 2.5)) if values[name] else None,
                'ci_95_upper': float(np.percentile(values[name], 97.5)) if values[name] else None,
            }
            for name in names
        },
    }


def selective_prediction_summary(
    labels,
    predictions,
    threshold: float,
    abstain,
) -> dict[str, Any]:
    """Evaluate the retained subset after a pre-specified abstention rule."""
    targets, scores, _ = _arrays(labels, predictions)
    abstained = np.asarray(abstain, dtype=bool)
    if abstained.ndim != 1 or len(abstained) != len(targets):
        raise ValueError('Abstention flags must be one-dimensional and match labels.')
    retained = ~abstained
    summary: dict[str, Any] = {
        'rule': 'pre_fitted_conformal_prediction_set_is_singleton',
        'total_sample_count': int(len(targets)),
        'retained_sample_count': int(retained.sum()),
        'abstained_sample_count': int(abstained.sum()),
        'retained_coverage': float(retained.mean()) if len(targets) else None,
        'abstention_rate': float(abstained.mean()) if len(targets) else None,
        'interpretation_warning': (
            'Selective metrics describe only examples retained by the pre-fitted '
            'abstention rule. They must always be reported with coverage.'
        ),
    }
    if not retained.any():
        summary['metrics'] = None
        return summary
    summary['metrics'] = calculate_binary_metrics(
        targets[retained], scores[retained], threshold
    )
    return summary


def structural_similarity_slices(
    labels,
    predictions,
    threshold: float,
    pair_minimum_similarities,
    raw_predictions=None,
    lower_similarity: float = 0.4,
    upper_similarity: float = 0.7,
) -> dict[str, Any]:
    """Report performance by nearest-training-drug similarity bands.

    The bands are a diagnostic view of structural novelty, not a substitute for
    the scaffold-disjoint or external evaluation protocol.
    """
    if not 0 <= lower_similarity < upper_similarity <= 1:
        raise ValueError('Similarity bands must satisfy 0 <= lower < upper <= 1.')
    targets, scores, raw_scores = _arrays(labels, predictions, raw_predictions)
    similarities = np.asarray(pair_minimum_similarities, dtype=float)
    if similarities.ndim != 1 or len(similarities) != len(targets):
        raise ValueError('Similarity scores must be one-dimensional and match labels.')
    if not np.isfinite(similarities).all() or not ((0 <= similarities).all() & (similarities <= 1).all()):
        raise ValueError('Similarity scores must be finite values between 0 and 1.')
    bands = {
        f'below_{lower_similarity:g}': similarities < lower_similarity,
        f'{lower_similarity:g}_to_{upper_similarity:g}': (similarities >= lower_similarity) & (similarities < upper_similarity),
        f'at_least_{upper_similarity:g}': similarities >= upper_similarity,
    }
    result: dict[str, Any] = {
        'method': 'minimum_of_two_nearest_train_drug_ecfp_tanimoto_similarities',
        'bands': {},
        'interpretation_warning': (
            'Similarity slices are structural-novelty diagnostics. They do not '
            'prove a model is reliable within a band or clinical applicability.'
        ),
    }
    for name, mask in bands.items():
        if not mask.any():
            result['bands'][name] = {'sample_count': 0, 'metrics': None}
            continue
        result['bands'][name] = {
            'sample_count': int(mask.sum()),
            'similarity_minimum': float(similarities[mask].min()),
            'similarity_maximum': float(similarities[mask].max()),
            'metrics': calculate_binary_metrics(
                targets[mask], scores[mask], threshold, raw_scores[mask]
            ),
        }
    return result


def save_confident_error_analysis(
    metadata: list[dict[str, Any]],
    labels,
    raw_predictions,
    calibrated_predictions,
    threshold: float,
    output_dir,
    split_name: str,
    maximum_rows_per_error_type: int = 100,
    additional_columns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save the most confident false positives and false negatives for review."""
    if maximum_rows_per_error_type <= 0:
        raise ValueError('maximum_rows_per_error_type must be positive.')
    targets, final_scores, raw_scores = _arrays(labels, calibrated_predictions, raw_predictions)
    if len(metadata) != len(targets):
        raise ValueError('Metadata must match evaluation labels.')
    table = pd.DataFrame(metadata)
    table['label'] = targets
    table['raw_prediction_score'] = raw_scores
    table['calibrated_prediction_score'] = final_scores
    table['threshold'] = float(threshold)
    table['predicted_label'] = (final_scores >= threshold).astype(int)
    table['confidence_margin_from_threshold'] = np.abs(final_scores - threshold)
    for name, values in (additional_columns or {}).items():
        if len(values) != len(table):
            raise ValueError(f'Error-analysis column {name!r} does not match the evaluation rows.')
        table[name] = values
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    safe_split_name = split_name.lower().replace(' ', '_')
    false_positive = table[(table['label'] == 0) & (table['predicted_label'] == 1)].copy()
    false_negative = table[(table['label'] == 1) & (table['predicted_label'] == 0)].copy()
    false_positive = false_positive.sort_values(
        ['calibrated_prediction_score', 'confidence_margin_from_threshold'], ascending=False
    ).head(maximum_rows_per_error_type)
    false_negative = false_negative.sort_values(
        ['calibrated_prediction_score', 'confidence_margin_from_threshold'], ascending=True
    ).head(maximum_rows_per_error_type)
    false_positive_path = destination / f'{safe_split_name}_false_positives.csv'
    false_negative_path = destination / f'{safe_split_name}_false_negatives.csv'
    false_positive.to_csv(false_positive_path, index=False)
    false_negative.to_csv(false_negative_path, index=False)
    return {
        'method': 'confidence_ranked_threshold_errors',
        'threshold': float(threshold),
        'maximum_rows_per_error_type': int(maximum_rows_per_error_type),
        'total_false_positives': int(((table['label'] == 0) & (table['predicted_label'] == 1)).sum()),
        'total_false_negatives': int(((table['label'] == 1) & (table['predicted_label'] == 0)).sum()),
        'saved_false_positives': int(len(false_positive)),
        'saved_false_negatives': int(len(false_negative)),
        'false_positives_path': str(false_positive_path),
        'false_negatives_path': str(false_negative_path),
        'interpretation_warning': (
            'False positives are unreported pairs predicted as reported; they are not '
            'established safe pairs. False negatives are reported pairs predicted below '
            'the validation-selected research threshold.'
        ),
    }
