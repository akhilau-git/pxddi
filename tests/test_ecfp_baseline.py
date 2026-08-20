"""Tests for the order-independent ECFP + linear-logistic benchmark."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from scipy import sparse


# The baseline reuses figure helpers from the Colab training pipeline.  Plot
# rendering is exercised in Colab; these unit tests only need the data logic.
sys.modules.setdefault('matplotlib', MagicMock())
sys.modules.setdefault('matplotlib.pyplot', MagicMock())

from src.training.train_ecfp_logistic_baseline import (
    MorganFingerprintCache,
    pair_feature_matrix,
    probability_from_linear_weights,
    safe_save_baseline_checkpoint,
)


def test_ecfp_pair_features_are_order_independent():
    cache = MorganFingerprintCache(num_bits=128)
    forward = pd.DataFrame({'source': ['CCO'], 'target': ['CCN']})
    reversed_pair = pd.DataFrame({'source': ['CCN'], 'target': ['CCO']})

    forward_features = pair_feature_matrix(forward, cache)
    reversed_features = pair_feature_matrix(reversed_pair, cache)

    assert forward_features.shape == (1, 256)
    assert np.array_equal(forward_features.toarray(), reversed_features.toarray())


def test_linear_probability_uses_saved_coefficients_without_a_pickled_model():
    features = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    predictions = probability_from_linear_weights(
        features, np.asarray([2.0, -2.0]), intercept=0.0
    )

    assert predictions[0] > 0.5
    assert predictions[1] < 0.5
    assert np.all((predictions > 0.0) & (predictions < 1.0))


def test_ecfp_checkpoint_is_numeric_and_reloadable_without_pickle(tmp_path):
    destination = tmp_path / 'baseline.npz'
    coefficients = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)

    digest = safe_save_baseline_checkpoint(
        coefficients, intercept=0.4, metadata={'epoch': 3}, path=destination
    )

    assert destination.is_file()
    assert len(digest) == 64
    with np.load(destination, allow_pickle=False) as checkpoint:
        assert np.array_equal(checkpoint['coefficients'], coefficients)
        assert checkpoint['intercept'][0] == np.float32(0.4)
