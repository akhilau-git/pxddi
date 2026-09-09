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


def test_sequence_target_attention_cross_modal_regression():
    from src.models.ddi_model import PxDDIModel, MODEL_ARCHITECTURE_MULTIMODAL
    from torch_geometric.data import Data, Batch

    # Build minimal drug graphs
    g1 = Data(x=torch.randn(3, 78), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), edge_attr=torch.randn(4, 10))
    g2 = Data(x=torch.randn(4, 78), edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]), edge_attr=torch.randn(4, 10))
    da = Batch.from_data_list([g1, g2])
    db = Batch.from_data_list([g2, g1])

    cyp3a4_seq = "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGK"
    egfr_seq = "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQR"

    model = PxDDIModel(
        in_channels=78,
        hidden_channels=64,
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        edge_feature_dim=10,
        target_feature_dim=50,
        target_hidden_channels=64,
        use_target_encoder=True,
        use_protein_sequence_encoder=True,
        use_target_sequence_fusion=True,
        use_cross_modal_attention=True,
        use_cross_modal_target_attention=True,
        use_cross_modal_sequence_attention=True,
    )

    # 1. Forward with valid primary protein sequences (tests the 64-dim sequence attention path)
    out, tox_a, tox_b = model(
        drug_a=da,
        drug_b=db,
        fp_a=torch.randn(2, 1024),
        fp_b=torch.randn(2, 1024),
        target_a=torch.randn(2, 50),
        target_b=torch.randn(2, 50),
        target_seq_a=[cyp3a4_seq, egfr_seq],
        target_seq_b=[egfr_seq, cyp3a4_seq],
        target_mask_a=torch.tensor([1.0, 1.0]),
        target_mask_b=torch.tensor([1.0, 1.0]),
    )
    assert out.shape == (2,)
    loss = out.sum()
    loss.backward()
    assert model.target_sequence_fusion is not None
    assert model.target_sequence_fusion[0].weight.grad is not None

    # 2. Forward with empty sequences (tests fallback to multi-hot target vectors without crashing)
    model.zero_grad()
    out_fallback, _, _ = model(
        drug_a=da,
        drug_b=db,
        fp_a=torch.randn(2, 1024),
        fp_b=torch.randn(2, 1024),
        target_a=torch.randn(2, 50),
        target_b=torch.randn(2, 50),
        target_seq_a=["", ""],
        target_seq_b=["", ""],
        target_mask_a=torch.tensor([1.0, 1.0]),
        target_mask_b=torch.tensor([1.0, 1.0]),
    )
    assert out_fallback.shape == (2,)
