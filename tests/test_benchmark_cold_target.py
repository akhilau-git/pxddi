import numpy as np
import pytest
import torch
import pandas as pd
from pathlib import Path

from src.training.benchmark_cold_target import (
    compute_comprehensive_metrics,
    paired_bootstrap_comparison,
    evaluate_loader_predictions,
)
from src.models.ddi_model import PxDDIModel, MODEL_ARCHITECTURE_MULTIMODAL
from src.data_prep.cached_graph_loader import MolecularCache, build_cached_multimodal_dataloader


def test_cold_target_metrics_and_bootstrap():
    labels = np.array([1, 1, 1, 0, 0, 0])
    probs = np.array([0.95, 0.85, 0.75, 0.15, 0.25, 0.05])
    metrics = compute_comprehensive_metrics(labels, probs, threshold=0.5)

    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0

    probs_base = np.array([0.6, 0.5, 0.7, 0.4, 0.3, 0.5])
    comp = paired_bootstrap_comparison(labels, probs_base, probs, n_bootstraps=50, seed=42)
    assert "delta_auroc_mean" in comp
    assert "p_value" in comp
    assert comp["delta_auroc_mean"] > 0


def test_cold_target_evaluation_handles_protein_sequences(tmp_path):
    smi1 = "c1ccccc1O"
    smi2 = "c1ccncc1"
    df_nodes = pd.DataFrame({
        "drug_id": [smi1, smi2],
        "canonical_smiles": [smi1, smi2],
        "target_sequence": [
            "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTS",
            "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQR",
        ],
    })
    nodes_csv = tmp_path / "master_nodes.csv"
    df_nodes.to_csv(nodes_csv, index=False)

    cache = MolecularCache()
    cache.populate_from_master_nodes(nodes_csv)
    assert cache.target_sequences[smi1] != ""

    df_pairs = pd.DataFrame({
        "drug_a_id": [smi1, smi2],
        "drug_b_id": [smi2, smi1],
        "label": [1.0, 0.0],
    })
    loader = build_cached_multimodal_dataloader(df_pairs, cache, batch_size=2, shuffle=False)

    first_graph = cache.graphs[smi1]
    in_dim = first_graph.x.size(1) if first_graph.x is not None else 78
    edge_dim = first_graph.edge_attr.size(1) if first_graph.edge_attr is not None else 10

    model = PxDDIModel(
        in_channels=in_dim,
        hidden_channels=16,
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        edge_feature_dim=edge_dim,
        use_protein_sequence_encoder=True,
    )

    metrics, probs, labels = evaluate_loader_predictions(model, loader, device=torch.device("cpu"))
    assert len(probs) == 2
    assert "auroc" in metrics


def test_extract_inductive_pair_features_and_hybrid_flow(tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from src.training.benchmark_cold_target import extract_inductive_pair_features

    smi1 = "c1ccccc1O"
    smi2 = "c1ccncc1"
    df_nodes = pd.DataFrame({
        "drug_id": [smi1, smi2],
        "canonical_smiles": [smi1, smi2],
        "target_sequence": [
            "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTS",
            "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQR",
        ],
    })
    nodes_csv = tmp_path / "master_nodes.csv"
    df_nodes.to_csv(nodes_csv, index=False)

    cache = MolecularCache()
    cache.populate_from_master_nodes(nodes_csv)

    df_pairs = pd.DataFrame({
        "drug_a_id": [smi1, smi2, smi1, smi2],
        "drug_b_id": [smi2, smi1, smi1, smi2],
        "label": [1.0, 0.0, 1.0, 0.0],
    })
    loader = build_cached_multimodal_dataloader(df_pairs, cache, batch_size=2, shuffle=False)

    first_graph = cache.graphs[smi1]
    in_dim = first_graph.x.size(1) if first_graph.x is not None else 78
    edge_dim = first_graph.edge_attr.size(1) if first_graph.edge_attr is not None else 10

    model = PxDDIModel(
        in_channels=in_dim,
        hidden_channels=16,
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        edge_feature_dim=edge_dim,
        use_protein_sequence_encoder=True,
        use_biophysical_features=True,
        cold_sim_dropout=0.30,
    )

    device = torch.device("cpu")
    X, y, deep_p = extract_inductive_pair_features(model, loader, device)
    assert X.shape[0] == 4
    assert X.shape[1] == 18
    assert len(y) == 4
    assert len(deep_p) == 4
    assert not np.isnan(X).any()

    clf = HistGradientBoostingClassifier(max_iter=10, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    assert len(probs) == 4
    assert np.all((probs >= 0.0) & (probs <= 1.0))
