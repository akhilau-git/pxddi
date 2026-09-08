import pytest
import torch

from src.models.protein_target_encoder import ProteinTargetSequenceEncoder


def test_protein_target_sequence_encoder_forward():
    encoder = ProteinTargetSequenceEncoder(output_dim=64, use_esm=False)
    encoder.eval()

    cyp3a4_seq = (
        "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGK"
        "VWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTS"
    )
    egfr_seq = (
        "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQR"
    )

    batch_seqs = [cyp3a4_seq, egfr_seq]
    with torch.no_grad():
        out = encoder(batch_seqs)

    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Verify invariance to sequence length
    single_out = encoder([cyp3a4_seq])
    assert single_out.shape == (1, 64)
