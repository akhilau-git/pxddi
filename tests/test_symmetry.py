"""Regression tests for the order-independent PxDDI pair architecture."""

from pathlib import Path
import sys

import pytest
import torch
from torch_geometric.data import Batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_prep.prepare_twosides import smiles_to_graph
from models.ddi_model import PxDDIModel


CHECKPOINT_PATH = PROJECT_ROOT / 'backend' / 'checkpoints' / 'pxddi_model.pt'
TEST_PAIRS = [
    (
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CC(=O)NC1=CC=C(C=C1)O',
    ),
    (
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    ),
]


@pytest.fixture(scope='module')
def model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    loaded_model = PxDDIModel(
        in_channels=checkpoint['in_channels'],
        hidden_channels=checkpoint['hidden_channels'],
        use_chemberta=checkpoint.get('use_chemberta', False),
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    return loaded_model


@pytest.mark.parametrize(('smiles_a', 'smiles_b'), TEST_PAIRS)
def test_shipped_model_is_order_independent(model, smiles_a, smiles_b):
    """The same pair must produce the same risk score in either input order."""
    graph_a = smiles_to_graph(smiles_a)
    graph_b = smiles_to_graph(smiles_b)
    assert graph_a is not None
    assert graph_b is not None

    batch_a = Batch.from_data_list([graph_a])
    batch_b = Batch.from_data_list([graph_b])

    with torch.no_grad():
        risk_ab, _, _ = model(batch_a, batch_b)
        risk_ba, _, _ = model(batch_b, batch_a)

    difference = abs(
        torch.sigmoid(risk_ab).item() - torch.sigmoid(risk_ba).item()
    )
    assert difference < 1e-6, f'Model is order-sensitive: diff={difference:.8f}'
