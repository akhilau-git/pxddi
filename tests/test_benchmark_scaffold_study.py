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


def test_ensure_scaffold_splits(tmp_path):
    from src.training.benchmark_scaffold_study import ensure_scaffold_splits

    # Create dummy master nodes
    nodes_p = tmp_path / "master_nodes.csv"
    df_nodes = pd.DataFrame({
        "drug_id": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"],
        "canonical_smiles": [
            "c1ccccc1O",  # Benzene 1
            "c1ccccc1Cl",  # Benzene 2
            "c1ccncc1O",  # Pyridine 1
            "c1ccncc1Cl",  # Pyridine 2
            "c1ccsc1O",  # Thiophene 1
            "c1ccsc1Cl",  # Thiophene 2
            "c1ccc2ccccc2c1",  # Naphthalene 1
            "c1ccc2cc(Cl)ccc2c1",  # Naphthalene 2
        ],
    })
    df_nodes.to_csv(nodes_p, index=False)

    # Create dummy edges within scaffolds
    edges_p = tmp_path / "master_ddi_edges.csv"
    df_edges = pd.DataFrame({
        "drug_a_id": ["D1", "D3", "D5", "D7"],
        "drug_b_id": ["D2", "D4", "D6", "D8"],
        "label": [1.0, 1.0, 1.0, 1.0],
    })
    df_edges.to_csv(edges_p, index=False)

    splits_dir = tmp_path / "scaffold_splits"
    res_dir = ensure_scaffold_splits(
        splits_dir=splits_dir,
        master_nodes_path=nodes_p,
        master_edges_path=edges_p,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=42,
    )

    assert (res_dir / "scaffold_train.csv").is_file()
    assert (res_dir / "scaffold_validation.csv").is_file()
    assert (res_dir / "scaffold_test.csv").is_file()
    assert (res_dir / "scaffold_split_audit.json").is_file()
