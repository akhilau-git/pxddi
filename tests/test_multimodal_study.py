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

        edges = []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
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

        # Verify artifacts written
        assert (out_study / "auditddi_multimodal_v1_best.pt").is_file()
        assert (out_study / "auditddi_multimodal_v1_training_history.csv").is_file()
        assert (out_study / "ablation" / "ablation_study_results.csv").is_file()
        assert (out_study / "error_analysis" / "cold_start_error_analysis.csv").is_file()
        assert (out_study / "calibration" / "calibration_metrics.json").is_file()


def test_run_full_study_alias_import():
    """Verify run_full_study alias is importable and identical."""
    from src.training.train_multimodal_study import run_full_study, run_full_multimodal_study
    assert run_full_study is run_full_multimodal_study

