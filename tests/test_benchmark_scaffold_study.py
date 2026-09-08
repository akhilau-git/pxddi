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


def test_evaluate_split_handles_tuple_model_output():
    from src.training.benchmark_scaffold_study import evaluate_split
    from src.models.ddi_model import PxDDIModel, MODEL_ARCHITECTURE_EDGE_AWARE
    from src.data_prep.cached_graph_loader import MolecularCache, build_cached_multimodal_dataloader

    # Construct minimal dummy data
    smi1 = "c1ccccc1O"
    smi2 = "c1ccncc1"
    df_nodes = pd.DataFrame({
        "drug_id": [smi1, smi2],
        "canonical_smiles": [smi1, smi2],
    })
    cache = MolecularCache()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df_nodes.to_csv(f.name, index=False)
        cache.populate_from_master_nodes(f.name)

    df_pairs = pd.DataFrame({
        "drug_a_id": [smi1, smi2],
        "drug_b_id": [smi2, smi1],
        "drug_a_smiles": [smi1, smi2],
        "drug_b_smiles": [smi2, smi1],
        "label": [1.0, 0.0],
    })
    loader = build_cached_multimodal_dataloader(df_pairs, cache, batch_size=2, shuffle=False)

    first_graph = cache.graphs[smi1]
    model = PxDDIModel(
        in_channels=first_graph.x.size(1),
        hidden_channels=16,
        architecture_version=MODEL_ARCHITECTURE_EDGE_AWARE,
        edge_feature_dim=first_graph.edge_attr.size(1),
    )

    metrics, probs, labels = evaluate_split(model, loader, device=torch.device("cpu"))
    assert len(probs) == 2
    assert "auroc" in metrics
    assert not np.isnan(metrics["accuracy"])

    # Test training step forward + backward pass
    import torch.nn as nn
    from torch.optim import AdamW
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
    optimizer = AdamW(model.parameters(), lr=1e-3)
    batch = next(iter(loader))
    out = model(
        drug_a=batch["drug_a"],
        drug_b=batch["drug_b"],
        fp_a=batch["fp_a"],
        fp_b=batch["fp_b"],
        gene_a=batch["gene_a"],
        gene_b=batch["gene_b"],
        gene_mask_a=batch["gene_mask_a"],
        gene_mask_b=batch["gene_mask_b"],
        clinical_tox_a=batch["tox_a"],
        clinical_tox_b=batch["tox_b"],
        clinical_tox_mask_a=batch["tox_mask_a"],
        clinical_tox_mask_b=batch["tox_mask_b"],
        target_a=batch["target_a"],
        target_b=batch["target_b"],
        target_mask_a=batch["target_mask_a"],
        target_mask_b=batch["target_mask_b"],
        geo_a=batch["geo_a"],
        geo_b=batch["geo_b"],
        geo_mask_a=batch["geo_mask_a"],
        geo_mask_b=batch["geo_mask_b"],
        pdb_a=batch["pdb_a"],
        pdb_b=batch["pdb_b"],
        pdb_mask_a=batch["pdb_mask_a"],
        pdb_mask_b=batch["pdb_mask_b"],
    )
    risk_logits = out[0] if isinstance(out, tuple) else out
    loss = criterion(risk_logits.view(-1), batch["labels"].float().view(-1))
    loss.backward()
    assert loss.item() > 0.0


