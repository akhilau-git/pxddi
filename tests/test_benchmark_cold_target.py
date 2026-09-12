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
    assert X.shape[1] == 27  # Includes continuous physicochemical, similarity, K-NN transfer, and cross terms
    assert len(y) == 4
    assert len(deep_p) == 4
    assert not np.isnan(X).any()

    clf = HistGradientBoostingClassifier(max_iter=10, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    assert len(probs) == 4
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_training_graph_retrieval_index_inductive_s2():
    from src.training.benchmark_cold_target import TrainingGraphRetrievalIndex
    
    # 1. Setup mock training dataframe with known edges
    # Train drugs: D1, D2, D3
    # D1-D2 has DDI (label 1), D1-D3 has no DDI (label 0)
    df_train = pd.DataFrame({
        "drug_a": ["D1", "D1"],
        "drug_b": ["D2", "D3"],
        "label": [1.0, 0.0],
    })
    
    # Mock fingerprint dict (dim 128 for testing)
    rng = np.random.RandomState(42)
    fp_dim = 128
    fps = {
        "D1": rng.randn(fp_dim).astype(np.float32),
        "D2": rng.randn(fp_dim).astype(np.float32),
        "D3": rng.randn(fp_dim).astype(np.float32),
    }
    
    index = TrainingGraphRetrievalIndex(df_train=df_train, training_fps=fps, k=2)
    
    # Test Transductive Pair (both in training) -> is_s2 == 0.0
    t_sc, m_sim, is_s2 = index.query_pair("D1", "D2", fps["D1"], fps["D2"])
    assert is_s2 == 0.0
    
    # Test S2 Pair: Novel drug D_unseen, known training drug D2
    # Make D_unseen very close to D1
    fp_novel = fps["D1"] + 0.01 * rng.randn(fp_dim).astype(np.float32)
    t_sc, m_sim, is_s2 = index.query_pair("D_unseen", "D2", fp_novel, fps["D2"])
    assert is_s2 == 1.0
    assert m_sim > 0.90  # Very high similarity to D1
    # Since D1 interacts with D2 (label 1.0), transfer score should be close to 1.0
    assert t_sc > 0.50
    
    # Test Order-Invariance: query_pair(novel, train) == query_pair(train, novel)
    t_sc_rev, m_sim_rev, is_s2_rev = index.query_pair("D2", "D_unseen", fps["D2"], fp_novel)
    assert is_s2 == is_s2_rev
    assert np.isclose(t_sc, t_sc_rev, atol=1e-5)
    assert np.isclose(m_sim, m_sim_rev, atol=1e-5)
    
    # Test with MolecularCache fingerprints (1024-dim tensors)
    class MockCache:
        def __init__(self):
            self.fingerprints = {
                "D1": torch.randn(1024),
                "D2": torch.randn(1024),
                "D3": torch.randn(1024),
            }
    mock_cache = MockCache()
    index_cache = TrainingGraphRetrievalIndex(df_train=df_train, cache=mock_cache, k=2)
    assert index_cache.train_fps.shape == (3, 1024)
    # Query with 1024-dim tensor converted to numpy
    t_c, m_c, s2_c = index_cache.query_pair("D_unseen", "D2", np.random.randn(1024).astype(np.float32), mock_cache.fingerprints["D2"].numpy())
    assert s2_c == 1.0
    assert 0.0 <= t_c <= 1.0
    # Query with different dim (e.g. 2048) gracefully aligned
    t_align, _, _ = index_cache.query_pair("D_unseen", "D2", np.random.randn(2048).astype(np.float32), mock_cache.fingerprints["D2"].numpy())
    assert 0.0 <= t_align <= 1.0




def test_ensemble_blend_and_calibrated_threshold():
    from sklearn.metrics import roc_curve
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    
    # Model A (deep network logits, slightly conservative on positives)
    probs_a = np.array([0.55, 0.62, 0.48, 0.70, 0.30, 0.25, 0.40, 0.15])
    
    # Model B (invariant GBDT, catches different positives)
    probs_b = np.array([0.65, 0.45, 0.60, 0.58, 0.20, 0.35, 0.18, 0.22])
    
    # Soft blend
    alpha = 0.50
    probs_blend = alpha * probs_a + (1.0 - alpha) * probs_b
    assert len(probs_blend) == len(y_true)
    
    # Standard threshold 0.50
    m_std = compute_comprehensive_metrics(y_true, probs_blend, threshold=0.50)
    assert m_std["auroc"] >= 0.90
    
    # Calibrated threshold via Youden's J
    fpr, tpr, threshs = roc_curve(y_true, probs_blend)
    j_scores = tpr - fpr
    best_thresh = float(threshs[np.argmax(j_scores)])
    m_cal = compute_comprehensive_metrics(y_true, probs_blend, threshold=best_thresh)
    
    # Calibrated threshold maximizes balanced accuracy and recall
    assert m_cal["sensitivity"] >= m_std["sensitivity"]
    assert m_cal["sensitivity"] == 1.0

