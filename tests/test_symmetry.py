"""Quick check: does the model give the same score for (A,B) as (B,A)?"""
import torch, sys
sys.path.append('src')
from models.ddi_model import PxDDIModel
from data_prep.prepare_twosides import smiles_to_graph, NUM_ATOM_FEATURES
from torch_geometric.data import Batch

checkpoint = torch.load('backend/checkpoints/pxddi_model.pt', map_location='cpu')
model = PxDDIModel(in_channels=checkpoint['in_channels'], hidden_channels=checkpoint['hidden_channels'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

smiles_a = "CC(=O)OC1=CC=CC=C1C(=O)O"
smiles_b = "CC(=O)NC1=CC=C(C=C1)O"
ga_data = smiles_to_graph(smiles_a)
gb_data = smiles_to_graph(smiles_b)
assert ga_data is not None and gb_data is not None
ga, gb = Batch.from_data_list([ga_data]), Batch.from_data_list([gb_data])

with torch.no_grad():
    risk_ab, _, _ = model(ga, gb)
    risk_ba, _, _ = model(gb, ga)

diff = abs(torch.sigmoid(risk_ab).item() - torch.sigmoid(risk_ba).item())
print(f"A+B risk: {torch.sigmoid(risk_ab).item():.4f}")
print(f"B+A risk: {torch.sigmoid(risk_ba).item():.4f}")
print(f"Difference: {diff:.4f}")
print("PASS (symmetric enough)" if diff < 0.05 else "FAIL (model is order-sensitive — document this limitation)")
