import tempfile
from pathlib import Path
import json
import pandas as pd
import torch
import pytest

from src.data_prep.cached_graph_loader import (
    MolecularCache,
    build_cached_multimodal_dataloader,
)
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_MULTIMODAL,
    PxDDIModel,
)


def test_molecular_cache_and_multimodal_dataloader():
    cache = MolecularCache(gene_dim=50)

    smi_a = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    smi_b = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # Caffeine

    # Mock gene vectors
    gene_vec_a = [1 if i in {0, 2, 4} else 0 for i in range(50)]
    gene_vec_b = [1 if i in {1, 3} else 0 for i in range(50)]

    assert cache.register_drug(smi_a, gene_vector=gene_vec_a, toxicity_score=0.15) is True
    assert cache.register_drug(smi_b, gene_vector=gene_vec_b, toxicity_score=0.30) is True

    # Check cached tensors
    assert smi_a in cache.graphs
    assert cache.fingerprints[smi_a].shape == (1024,)
    assert cache.gene_vectors[smi_a].shape == (50,)
    assert cache.gene_masks[smi_a].item() == 1.0
    assert cache.toxicity_scalars[smi_a].item() == pytest.approx(0.15)
    assert cache.toxicity_masks[smi_a].item() == 1.0

    # Build DataLoader with 4 pairs
    df_edges = pd.DataFrame({
        "drug_a_id": [smi_a, smi_b, smi_a, smi_b],
        "drug_b_id": [smi_b, smi_a, smi_b, smi_a],
        "label": [1.0, 0.0, 1.0, 0.0],
    })

    loader = build_cached_multimodal_dataloader(
        edges_df=df_edges,
        molecular_cache=cache,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))
    assert batch["drug_a"].num_graphs == 2
    assert batch["drug_b"].num_graphs == 2
    assert batch["fp_a"].shape == (2, 1024)
    assert batch["gene_a"].shape == (2, 50)
    assert batch["tox_a"].shape == (2,)
    assert batch["labels"].shape == (2,)


def test_multimodal_model_forward():
    cache = MolecularCache(gene_dim=50)

    smi_a = "CC(=O)Oc1ccccc1C(=O)O"
    smi_b = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

    cache.register_drug(smi_a, gene_vector=[1]*50, toxicity_score=0.2)
    cache.register_drug(smi_b, gene_vector=None, toxicity_score=None)  # unprofiled fallback

    df_edges = pd.DataFrame({
        "drug_a_id": [smi_a, smi_b],
        "drug_b_id": [smi_b, smi_a],
        "label": [1.0, 0.0],
    })

    loader = build_cached_multimodal_dataloader(
        edges_df=df_edges,
        molecular_cache=cache,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))

    # Initialize Multimodal Model
    model = PxDDIModel(
        in_channels=batch["drug_a"].x.size(1),
        hidden_channels=32,
        edge_feature_dim=batch["drug_a"].edge_attr.size(1),
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        gene_feature_dim=50,
        gene_hidden_channels=32,
        use_clinical_toxicity=True,
    )
    model.eval()

    with torch.no_grad():
        risk_logits, tox_a_logits, tox_b_logits = model(
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
        )

    assert risk_logits.shape == (2,)
    assert tox_a_logits.shape == (2,)
    assert tox_b_logits.shape == (2,)

    # Test backward compatibility without multimodal args
    with torch.no_grad():
        fallback_risk, _, _ = model(
            drug_a=batch["drug_a"],
            drug_b=batch["drug_b"],
        )
    assert fallback_risk.shape == (2,)


def test_benchmark_model_smoke():
    from src.training.benchmark_cold_start import train_benchmark_model

    cache = MolecularCache(gene_dim=50)
    smi_a = "CC(=O)Oc1ccccc1C(=O)O"
    smi_b = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    cache.register_drug(smi_a, gene_vector=[1]*50, toxicity_score=0.2)
    cache.register_drug(smi_b, gene_vector=[0]*50, toxicity_score=0.1)

    df_train = pd.DataFrame({"drug_a_id": [smi_a, smi_b]*4, "drug_b_id": [smi_b, smi_a]*4, "label": [1.0, 0.0]*4})
    df_val = pd.DataFrame({"drug_a_id": [smi_a, smi_b], "drug_b_id": [smi_b, smi_a], "label": [1.0, 0.0]})
    test_splits = {"s1_cold": df_val}

    res = train_benchmark_model(
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        cache=cache,
        train_df=df_train,
        val_df=df_val,
        test_splits=test_splits,
        epochs=1,
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert "s1_cold_auroc" in res
    assert "avg_epoch_time_sec" in res
    assert res["avg_epoch_time_sec"] > 0.0
