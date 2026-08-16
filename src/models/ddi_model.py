import torch, torch.nn as nn
from .encoder import MolecularEncoder
from .toxicity_model import ToxicityHead
from .patient_context import PatientContextEncoder

class PxDDIModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, n_comorbidities=10, use_chemberta=False):
        super().__init__()
        self.use_chemberta = use_chemberta
        if use_chemberta:
            from .encoder import MolecularEncoderChemBERTa
            self.encoder = MolecularEncoderChemBERTa(hidden_channels)
        else:
            self.encoder = MolecularEncoder(in_channels, hidden_channels)
            
        self.toxicity_head = ToxicityHead(hidden_channels)
        self.patient_encoder = PatientContextEncoder(n_comorbidities, hidden_channels)
        self.risk_classifier = nn.Sequential(
            nn.Linear(hidden_channels*2+2, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64,1))
    def forward(self, drug_a, drug_b, patient=None):
        if self.use_chemberta:
            device = next(self.parameters()).device
            ea = self.encoder(drug_a.smiles, device)
            eb = self.encoder(drug_b.smiles, device)
        else:
            ea = self.encoder(drug_a.x, drug_a.edge_index, drug_a.batch)
            eb = self.encoder(drug_b.x, drug_b.edge_index, drug_b.batch)
        ta = torch.sigmoid(self.toxicity_head(ea))
        tb = torch.sigmoid(self.toxicity_head(eb))
        if patient is not None:
            g = self.patient_encoder(patient['age_band'], patient['sex'], patient['comorbidities'])
            ea, eb = ea*g, eb*g
        combined = torch.cat([ea, eb, ta.unsqueeze(-1), tb.unsqueeze(-1)], dim=1)
        return self.risk_classifier(combined).squeeze(-1), ta, tb
