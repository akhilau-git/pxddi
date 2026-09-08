import json
from pathlib import Path
import pytest
import pandas as pd

from src.data_prep.download_uniprot_data import (
    extract_target_accessions_from_workspace,
    pull_realtime_uniprot_dataset,
)


def test_extract_target_accessions_from_workspace(tmp_path):
    # Create a small dummy master nodes csv
    dummy_csv = tmp_path / "dummy_master_nodes.csv"
    df = pd.DataFrame({
        "drug_id": ["DB00001", "DB00002"],
        "uniprot_targets": ["P08684, P10635", "Q14524"],
    })
    df.to_csv(dummy_csv, index=False)

    targets = extract_target_accessions_from_workspace(master_nodes_path=dummy_csv)
    assert "P08684" in targets
    assert "P10635" in targets
    assert "Q14524" in targets
    # Canonical targets should also be included
    assert "P00533" in targets


def test_pull_realtime_uniprot_dataset_local_test(tmp_path, monkeypatch):
    # Mock network call to test directory creation and deduplication
    out_dir = tmp_path / "uniprot_test"

    # Pre-populate one fasta to test cache recognition
    fastas_dir = out_dir / "fastas"
    fastas_dir.mkdir(parents=True)
    sample_fasta = ">sp|P08684|CP3A4_HUMAN Cytochrome P450 3A4 OS=Homo sapiens OX=9606 GN=CYP3A4 PE=1 SV=2\nMALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFK\n"
    (fastas_dir / "P08684.fasta").write_text(sample_fasta, encoding="utf-8")

    manifest = pull_realtime_uniprot_dataset(
        output_dir=out_dir,
        target_subset=["P08684"],
        rate_limit_delay=0.0,
    )

    assert (out_dir / "uniprot_sequences.fasta").is_file()
    assert (out_dir / "target_sequences.json").is_file()
    assert (out_dir / "uniprot_targets_metadata.csv").is_file()
    assert (out_dir / "uniprot_download_manifest.json").is_file()

    # Verify no duplicate entries in metadata
    df_meta = pd.read_csv(out_dir / "uniprot_targets_metadata.csv")
    assert len(df_meta["uniprot_id"]) == len(df_meta["uniprot_id"].unique())
    assert manifest["total_sequences_saved"] > 0
