import numpy as np
import pytest
import torch
import pandas as pd
from pathlib import Path

from src.training.benchmark_scaffold_study import (
    compute_comprehensive_metrics,
    paired_bootstrap_comparison,
)


def test_compute_comprehensive_metrics():
    labels = np.array([1, 1, 1, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.2, 0.3, 0.1])
    metrics = compute_comprehensive_metrics(labels, probs, threshold=0.5)

    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["brier_score"] < 0.1
    assert "ece" in metrics


def test_paired_bootstrap_comparison():
    labels = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    probs_b = np.array([0.6, 0.5, 0.7, 0.4, 0.3, 0.5, 0.6, 0.4, 0.5, 0.3])
    probs_m = np.array([0.9, 0.8, 0.9, 0.1, 0.2, 0.1, 0.8, 0.2, 0.9, 0.1])

    res = paired_bootstrap_comparison(labels, probs_b, probs_m, n_bootstraps=50, seed=42)
    assert "delta_auroc_mean" in res
    assert "delta_auroc_ci95" in res
    assert "p_value" in res
    assert res["delta_auroc_mean"] > 0
