"""Tests for the comprehensive multimodal study suite."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest
import torch

from src.training.train_multimodal_study import (
    analyze_cold_start_coverage_errors,
    evaluate_multimodal_calibration,
    run_full_multimodal_study,
    run_modality_ablation_study,
    train_extended_multimodal,
)
from src.data_prep.cached_graph_loader import MolecularCache
from src.training.benchmark_cold_start import ensure_benchmark_splits


def test_multimodal_study_smoke():
    drugs = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CN1C2CCC1C(C(C2)OC(=O)c3ccccc3)C(=O)OC",
        "CN1CCC[C@H]1c2cccnc2",
        "CC(=O)Nc1ccc(O)cc1",
        "CCO",
        "c1ccccc1",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        graph_dir = tmp_p / "unified_graph"
        graph_dir.mkdir(parents=True)

        nodes_df = pd.DataFrame([
            {
                "drug_id": drug,
                "canonical_smiles": drug,
                "gene_vector_multihot": [1 if i % 2 == 0 else 0 for i in range(50)],
                "toxicity_score": 0.25 if idx % 2 == 0 else None,
            }
            for idx, drug in enumerate(drugs)
        ])
        nodes_path = graph_dir / "master_drug_nodes.csv"
        nodes_df.to_csv(nodes_path, index=False)

        # Generate realistic non-clique pairs so unreported negatives exist
        edges = []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                if (i + j) % 2 == 1:
                    edges.append({
                        "drug_a_id": drugs[i],
                        "drug_b_id": drugs[j],
                        "interaction_type": "adverse_interaction",
                        "evidence_count": 1,
                    })
        edges_df = pd.DataFrame(edges)
        edges_path = graph_dir / "master_ddi_edges.csv"
        edges_df.to_csv(edges_path, index=False)

        splits_p = ensure_benchmark_splits(
            splits_dir=tmp_p / "splits",
            master_nodes_path=nodes_path,
            master_edges_path=edges_path,
            holdout_fraction=0.33,
        )

        out_study = tmp_p / "study_out"
        res = run_full_multimodal_study(
            master_nodes_path=nodes_path,
            splits_dir=splits_p,
            output_dir=out_study,
            master_edges_path=edges_path,
            extended_epochs=1,
            ablation_epochs=1,
            batch_size=4,
            device=torch.device("cpu"),
        )

        assert "extended_metrics" in res
        assert "ablation" in res
        assert "ablation_results" in res
        assert "tier_summary" in res
        assert "calibration_report" in res
        assert "literature_benchmark" in res

        # Verify artifacts written
        assert (out_study / "auditddi_multimodal_v1_best.pt").is_file()
        assert (out_study / "auditddi_multimodal_v1_training_history.csv").is_file()
        assert (out_study / "ablation" / "ablation_study_results.csv").is_file()
        assert (out_study / "error_analysis" / "cold_start_error_analysis.csv").is_file()
        assert (out_study / "calibration" / "calibration_metrics.json").is_file()
        assert (out_study / "literature_benchmark_comparison.csv").is_file()
        assert (out_study / "literature_benchmark_comparison.md").is_file()


def test_run_full_study_alias_import():
    """Verify run_full_study alias is importable and identical."""
    from src.training.train_multimodal_study import run_full_study, run_full_multimodal_study
    assert run_full_study is run_full_multimodal_study


def test_run_full_study_keyword_aliases():
    """Verify run_full_study accepts master_nodes_csv and pretrained_encoder_path."""
    import inspect
    from src.training.train_multimodal_study import run_full_study
    sig = inspect.signature(run_full_study)
    assert "master_nodes_path" in sig.parameters
    assert sig.parameters["master_nodes_path"].default is None
    assert "pretrained_encoder_path" in sig.parameters


def test_memory_dropout_and_noise():
    from src.models.ddi_model import PxDDIModel, MODEL_ARCHITECTURE_MULTIMODAL
    from src.data_prep.prepare_twosides import smiles_to_graph, FEATURE_SCHEMA_RICH

    g = smiles_to_graph("CCO", feature_schema=FEATURE_SCHEMA_RICH, include_fingerprint_features=True)
    assert g is not None
    assert g.x is not None
    assert g.edge_attr is not None

    model = PxDDIModel(
        in_channels=g.x.size(1),
        hidden_channels=32,
        edge_feature_dim=g.edge_attr.size(1),
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        use_neighbor_memory=True,
        memory_dropout=0.50,
        embedding_noise_std=0.05,
    )

    mem_feat = torch.ones((1, 3))
    model.train()
    out_train, _, _ = model(drug_a=g, drug_b=g, memory_features=mem_feat)
    assert out_train is not None

    model.eval()
    out_eval, _, _ = model(drug_a=g, drug_b=g, memory_features=mem_feat)
    assert out_eval is not None
    assert torch.is_tensor(out_eval)


def test_molecular_cache_dynamic_dimensions_and_resilience():
    cache = MolecularCache(gene_dim=50, target_dim=50, geo_dim=2)
    smi = "CCO"

    # Test with non-standard lengths: should pad or truncate cleanly without discarding
    ok = cache.register_drug(
        smi,
        gene_vector=[1.0] * 30,  # shorter than 50 -> padded
        toxicity_score=0.75,
        target_vector=[1.0] * 60,  # longer than 50 -> truncated
        geo_vector=[0.5, 0.8],
    )
    assert ok is True
    assert cache.gene_vectors[smi].shape == (50,)
    assert cache.gene_masks[smi].item() == 1.0
    assert cache.toxicity_scalars[smi].item() == 0.75
    assert cache.toxicity_masks[smi].item() == 1.0
    assert cache.target_vectors[smi].shape == (50,)
    assert cache.target_masks[smi].item() == 1.0
    assert cache.geo_vectors[smi].shape == (2,)
    assert cache.geo_masks[smi].item() == 1.0


def test_model_from_checkpoint_auto_detection():
    from src.models.ddi_model import model_from_checkpoint, MODEL_ARCHITECTURE_MULTIMODAL

    ckpt = {
        'in_channels': 30,
        'hidden_channels': 64,
        'edge_feature_dim': 11,
        'architecture_version': MODEL_ARCHITECTURE_MULTIMODAL,
        'model_state_dict': {
            'target_encoder.0.weight': torch.randn(64, 50),
            'cross_modal_attention.mol_proj.weight': torch.randn(64, 64),
        },
    }

    model = model_from_checkpoint(ckpt)
    assert model.target_encoder is not None
    assert model.cross_modal_attention is not None
    assert model.use_clinical_toxicity is True


def test_pdb_and_geo_encoders_forward():
    from src.models.ddi_model import PxDDIModel, MODEL_ARCHITECTURE_MULTIMODAL
    from src.data_prep.prepare_twosides import smiles_to_graph, FEATURE_SCHEMA_RICH

    g1 = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O", feature_schema=FEATURE_SCHEMA_RICH, include_fingerprint_features=True)
    g2 = smiles_to_graph("CC(=O)Nc1ccc(O)cc1", feature_schema=FEATURE_SCHEMA_RICH, include_fingerprint_features=True)
    assert g1 is not None and g1.x is not None and g1.edge_attr is not None
    assert g2 is not None and g2.x is not None and g2.edge_attr is not None

    model = PxDDIModel(
        in_channels=g1.x.size(1),
        hidden_channels=32,
        edge_feature_dim=g1.edge_attr.size(1),
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        use_pdb_encoder=True,
        pdb_feature_dim=50,
        pdb_hidden_channels=64,
        use_geo_features=True,
        use_geo_encoder=True,
        geo_dim=2,
        geo_hidden_channels=32,
    )

    assert model.pdb_encoder is not None
    assert model.geo_encoder is not None

    pdb_a = torch.randn(1, 50)
    pdb_b = torch.randn(1, 50)
    pdb_mask = torch.ones(1)
    geo_a = torch.randn(1, 2)
    geo_b = torch.randn(1, 2)
    geo_mask = torch.ones(1)

    model.eval()
    risk_out, tox_a, tox_b = model(
        drug_a=g1,
        drug_b=g2,
        pdb_a=pdb_a,
        pdb_b=pdb_b,
        pdb_mask_a=pdb_mask,
        pdb_mask_b=pdb_mask,
        geo_a=geo_a,
        geo_b=geo_b,
        geo_mask_a=geo_mask,
        geo_mask_b=geo_mask,
    )
    assert risk_out is not None
    assert risk_out.shape == (1,)


def test_inductive_bio_features_and_cross_modal_attention():
    from src.models.ddi_model import PxDDIModel, MODEL_ARCHITECTURE_MULTIMODAL
    from src.data_prep.prepare_twosides import smiles_to_graph, FEATURE_SCHEMA_RICH

    g1 = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O", feature_schema=FEATURE_SCHEMA_RICH, include_fingerprint_features=True)
    g2 = smiles_to_graph("CC(C)Cc1ccc(cc1)C(C)C(=O)O", feature_schema=FEATURE_SCHEMA_RICH, include_fingerprint_features=True)
    assert g1 is not None and g1.x is not None and g1.edge_attr is not None
    assert g2 is not None and g2.x is not None and g2.edge_attr is not None

    model = PxDDIModel(
        in_channels=g1.x.size(1),
        hidden_channels=64,
        edge_feature_dim=g1.edge_attr.size(1),
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        use_clinical_toxicity=True,
        use_cross_modal_attention=True,
        use_cross_modal_target_attention=True,
        use_cross_modal_pdb_attention=True,
        use_target_encoder=True,
        target_feature_dim=20,
        target_hidden_channels=64,
        use_pdb_encoder=True,
        pdb_feature_dim=50,
        pdb_hidden_channels=64,
        use_geo_features=True,
        use_geo_encoder=True,
        geo_dim=2,
        geo_hidden_channels=32,
        use_inductive_bio_features=True,
        use_fusion_norm=True,
        mol_dropout=0.20,
    )

    assert model.cross_modal_target_attention is not None
    assert model.cross_modal_pdb_attention is not None
    assert model.fusion_norm is not None
    assert model.use_inductive_bio_features is True

    # Test in training mode (dropout active)
    model.train()
    gene_a = torch.ones(1, 50)
    gene_b = torch.ones(1, 50)
    g_mask = torch.ones(1)
    target_a = torch.randn(1, 20)
    target_b = torch.randn(1, 20)
    t_mask = torch.ones(1)
    pdb_a = torch.randn(1, 50)
    pdb_b = torch.randn(1, 50)
    p_mask = torch.ones(1)
    fp_a = torch.randn(1, 1024)
    fp_b = torch.randn(1, 1024)
    geo_a = torch.randn(1, 2)
    geo_b = torch.randn(1, 2)
    geo_mask = torch.ones(1)
    tox_a = torch.tensor([[0.8]])
    tox_b = torch.tensor([[0.5]])

    risk_out, tox_a_l, tox_b_l = model(
        drug_a=g1,
        drug_b=g2,
        gene_a=gene_a,
        gene_b=gene_b,
        gene_mask_a=g_mask,
        gene_mask_b=g_mask,
        target_a=target_a,
        target_b=target_b,
        target_mask_a=t_mask,
        target_mask_b=t_mask,
        pdb_a=pdb_a,
        pdb_b=pdb_b,
        pdb_mask_a=p_mask,
        pdb_mask_b=p_mask,
        fp_a=fp_a,
        fp_b=fp_b,
        geo_a=geo_a,
        geo_b=geo_b,
        geo_mask_a=geo_mask,
        geo_mask_b=geo_mask,
        clinical_tox_a=tox_a,
        clinical_tox_b=tox_b,
    )

    assert risk_out is not None
    assert risk_out.shape == (1,)
    assert hasattr(model, '_last_ea')
    assert model._last_ea is not None

    # Test in eval mode (deterministic)
    model.eval()
    risk_eval, _, _ = model(
        drug_a=g1,
        drug_b=g2,
        gene_a=gene_a,
        gene_b=gene_b,
        gene_mask_a=g_mask,
        gene_mask_b=g_mask,
        target_a=target_a,
        target_b=target_b,
        target_mask_a=t_mask,
        target_mask_b=t_mask,
        pdb_a=pdb_a,
        pdb_b=pdb_b,
        pdb_mask_a=p_mask,
        pdb_mask_b=p_mask,
        fp_a=fp_a,
        fp_b=fp_b,
        geo_a=geo_a,
        geo_b=geo_b,
        geo_mask_a=geo_mask,
        geo_mask_b=geo_mask,
        clinical_tox_a=tox_a,
        clinical_tox_b=tox_b,
    )
    assert risk_eval is not None
    assert torch.is_tensor(risk_eval)





