"""Tests for Phase 7 DDI metric and error-analysis utilities."""

import os
from pathlib import Path
import subprocess
import sys

import json

import numpy as np
import pandas as pd

from src.evaluation.ddi_metrics import (
    bootstrap_confidence_intervals,
    calculate_binary_metrics,
    save_confident_error_analysis,
    selective_prediction_summary,
    structural_similarity_slices,
)


def test_evaluation_module_supports_colab_direct_script_import_layout(tmp_path):
    """The training script exposes ``src`` as top-level modules in Colab."""
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(project_root / 'src')
    completed = subprocess.run(
        [sys.executable, '-c', 'import evaluation.ddi_metrics'],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_binary_metrics_include_complementary_decision_and_calibration_fields():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = calculate_binary_metrics(labels, scores, threshold=0.5)

    assert metrics['auroc'] == 1.0
    assert metrics['average_precision'] == 1.0
    assert metrics['accuracy'] == 1.0
    assert metrics['balanced_accuracy'] == 1.0
    assert metrics['sensitivity'] == 1.0
    assert metrics['specificity'] == 1.0
    assert metrics['negative_predictive_value'] == 1.0
    assert metrics['false_positive_rate'] == 0.0
    assert metrics['false_negative_rate'] == 0.0
    assert metrics['positive_prevalence'] == 0.5


def test_bootstrap_intervals_are_stratified_and_json_serializable():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])

    intervals = bootstrap_confidence_intervals(
        labels, scores, threshold=0.5, resamples=30, seed=7
    )

    assert intervals['status'] == 'evaluated'
    assert intervals['resamples'] == 30
    assert intervals['metrics']['auroc']['point_estimate'] == 1.0
    assert intervals['metrics']['mcc']['valid_resamples'] == 30
    json.dumps(intervals)


def test_selective_summary_keeps_coverage_with_retained_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.55, 0.8, 0.4])
    abstain = np.array([False, True, False, True])

    summary = selective_prediction_summary(labels, scores, threshold=0.5, abstain=abstain)

    assert summary['retained_sample_count'] == 2
    assert summary['abstained_sample_count'] == 2
    assert summary['retained_coverage'] == 0.5
    assert summary['metrics']['accuracy'] == 1.0


def test_structural_similarity_slices_keep_each_band_auditably_separate():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.2, 0.1, 0.8, 0.9, 0.4, 0.6])
    similarities = np.array([0.2, 0.35, 0.45, 0.6, 0.75, 0.9])

    summary = structural_similarity_slices(
        labels, scores, threshold=0.5, pair_minimum_similarities=similarities
    )

    assert summary['bands']['below_0.4']['sample_count'] == 2
    assert summary['bands']['0.4_to_0.7']['sample_count'] == 2
    assert summary['bands']['at_least_0.7']['sample_count'] == 2


def test_error_analysis_saves_confidence_ranked_false_positives_and_negatives(tmp_path):
    metadata = [
        {'source': 'A', 'target': 'B'},
        {'source': 'C', 'target': 'D'},
        {'source': 'E', 'target': 'F'},
        {'source': 'G', 'target': 'H'},
    ]
    labels = np.array([0, 0, 1, 1])
    raw_scores = np.array([0.8, 0.6, 0.4, 0.2])
    final_scores = np.array([0.9, 0.7, 0.3, 0.1])

    summary = save_confident_error_analysis(
        metadata, labels, raw_scores, final_scores, 0.5, tmp_path, 'S1',
        maximum_rows_per_error_type=1,
        additional_columns={'structural_ood_flag': np.array([True, False, True, False])},
    )

    false_positive = pd.read_csv(summary['false_positives_path'])
    false_negative = pd.read_csv(summary['false_negatives_path'])
    assert summary['total_false_positives'] == 2
    assert summary['total_false_negatives'] == 2
    assert false_positive.iloc[0]['source'] == 'A'
    assert false_negative.iloc[0]['source'] == 'G'
    assert 'structural_ood_flag' in false_positive.columns
