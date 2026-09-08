import json
import pytest
from pathlib import Path

from src.data_prep.uniprot_pipeline import (
    clean_protein_sequence,
    parse_fasta_string,
    fetch_uniprot_sequence,
    build_target_sequence_catalog,
    CANONICAL_TARGET_TO_UNIPROT,
    OFFLINE_SEQUENCE_FALLBACKS,
)


def test_clean_protein_sequence_and_fasta_parsing():
    raw_fasta = """>sp|P08684|CP3A4_HUMAN Cytochrome P450 3A4 OS=Homo sapiens OX=9606 GN=CYP3A4 PE=1 SV=2
MALIPDLAME TWLLLAVSLV LLYLYGTHSH GLFKKLGIPG PTPLPFLGNI LSYHKGFCMF
DMECHKKYGK VWGFYDGQQP VLAITDPDMI KTVLVKECYS VFTNRRPFGP VGFMKSAISI
AEDEEWKRLR SLLSPTFTSG KLKEMVPIIA QYGDVLVRNL RREAETGKPV TLKDVFGAYS
"""
    seq = parse_fasta_string(raw_fasta)
    assert len(seq) == 180
    assert seq.startswith("MALIPDLAMETWLLL")
    assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq)

    # Rejection of invalid non-amino-acid sequences
    with pytest.raises(ValueError, match="No valid standard amino acid"):
        clean_protein_sequence("12345!@#$%")


def test_fetch_uniprot_sequence_offline_cache(tmp_path):
    cache_dir = tmp_path / "uniprot_cache"

    # Test resolving by gene symbol
    seq_cyp3a4 = fetch_uniprot_sequence("CYP3A4", cache_dir=cache_dir)
    assert len(seq_cyp3a4) > 100
    assert (cache_dir / "P08684.fasta").is_file()

    # Second read must hit the local disk cache
    seq_cached = fetch_uniprot_sequence("P08684", cache_dir=cache_dir)
    assert seq_cached == seq_cyp3a4


def test_build_target_sequence_catalog(tmp_path):
    cache_dir = tmp_path / "uniprot_cache"
    export_path = tmp_path / "target_sequences.json"

    targets = ["CYP3A4", "CYP2D6", "PTGS2", "EGFR"]
    catalog = build_target_sequence_catalog(targets, cache_dir=cache_dir, export_json_path=export_path)

    assert len(catalog) == 4
    assert "CYP3A4" in catalog
    assert "EGFR" in catalog
    assert export_path.is_file()

    loaded = json.loads(export_path.read_text(encoding="utf-8"))
    assert loaded["CYP3A4"] == catalog["CYP3A4"]
