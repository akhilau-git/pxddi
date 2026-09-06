"""Tests for multi-task polypharmacy side-effect prediction."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest
import torch

from src.training.train_multitask_ddi import (
    extract_top_side_effects,
    prepare_multitask_pairs,
    run_multitask_side_effect_study,
)


def test_multitask_side_effect_study_smoke():
    drugs = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CN1C2CCC1C(C(C2)OC(=O)c3ccccc3)C(=O)OC",
        "[Ca+2]",  # Single ion with no bonds to test defensive filtering
    ]
    effects = ["Hypotension", "Headache", "Hyperkalemia"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)

        # 1. Master Nodes
        nodes_df = pd.DataFrame([
            {
                "canonical_smiles": d,
                "gene_vector_multihot": [1 if i % 2 == 0 else 0 for i in range(50)],
                "toxicity_score": 0.3,
            }
            for d in drugs
        ])
        nodes_path = tmp_p / "master_nodes.csv"
        nodes_df.to_csv(nodes_path, index=False)

        # 2. Master Edges
        edges = []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                for eff in effects:
                    edges.append({
                        "drug_a_id": drugs[i],
                        "drug_b_id": drugs[j],
                        "interaction_type": eff,
                    })
        edges_df = pd.DataFrame(edges)
        edges_path = tmp_p / "master_edges.csv"
        edges_df.to_csv(edges_path, index=False)

        # Verify extraction
        top_se = extract_top_side_effects(edges_path, top_k=2)
        assert len(top_se) == 2

        pairs_df = prepare_multitask_pairs(edges_path, top_se)
        assert len(pairs_df) > 0
        assert len(pairs_df.iloc[0]["multitask_labels"]) == 2

        # Verify end-to-end multi-task training
        out_dir = tmp_p / "multitask_out"
        res = run_multitask_side_effect_study(
            master_nodes_path=nodes_path,
            master_edges_path=edges_path,
            output_dir=out_dir,
            top_k_side_effects=2,
            epochs=1,
            batch_size=2,
            device=torch.device("cpu"),
        )

        assert "test_metrics" in res
        assert "micro_auroc" in res["test_metrics"]
        assert (out_dir / "auditddi_multitask_best.pt").is_file()
        assert (out_dir / "side_effects_vocabulary.json").is_file()
