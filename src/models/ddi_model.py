import torch, torch.nn as nn
from .encoder import EdgeAwareMolecularEncoder, MolecularEncoder
from .toxicity_model import ToxicityHead
from .patient_context import PatientContextEncoder


MODEL_ARCHITECTURE_LEGACY = 'legacy_gat_v1'
MODEL_ARCHITECTURE_EDGE_AWARE = 'edge_aware_gat_v2'


def model_from_checkpoint(checkpoint):
    """Construct either versioned GNN safely from checkpoint metadata."""
    return PxDDIModel(
        in_channels=checkpoint['in_channels'],
        hidden_channels=checkpoint['hidden_channels'],
        use_chemberta=checkpoint.get('use_chemberta', False),
        architecture_version=checkpoint.get(
            'architecture_version', MODEL_ARCHITECTURE_LEGACY
        ),
        edge_feature_dim=checkpoint.get('edge_feature_dim'),
        use_toxicity_pair_features=checkpoint.get('use_toxicity_pair_features', True),
    )

class PxDDIModel(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        n_comorbidities=10,
        use_chemberta=False,
        architecture_version=MODEL_ARCHITECTURE_LEGACY,
        edge_feature_dim=None,
        use_toxicity_pair_features=True,
    ):
        super().__init__()
        self.use_chemberta = use_chemberta
        self.architecture_version = architecture_version
        self.use_toxicity_pair_features = use_toxicity_pair_features
        if use_chemberta:
            from .encoder import MolecularEncoderChemBERTa
            self.encoder = MolecularEncoderChemBERTa(hidden_channels)
        elif architecture_version == MODEL_ARCHITECTURE_LEGACY:
            self.encoder = MolecularEncoder(in_channels, hidden_channels)
        elif architecture_version == MODEL_ARCHITECTURE_EDGE_AWARE:
            if edge_feature_dim is None or edge_feature_dim <= 0:
                raise ValueError('edge_feature_dim must be positive for the edge-aware GNN.')
            self.encoder = EdgeAwareMolecularEncoder(
                in_channels,
                edge_feature_dim,
                hidden_channels,
            )
        else:
            raise ValueError(f'Unknown model architecture: {architecture_version}.')
            
        self.toxicity_head = ToxicityHead(hidden_channels)
        self.patient_encoder = PatientContextEncoder(n_comorbidities, hidden_channels)
        risk_input_channels = hidden_channels * 2 + (2 if use_toxicity_pair_features else 0)
        self.risk_classifier = nn.Sequential(
            nn.Linear(risk_input_channels, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64,1))
    def forward(self, drug_a, drug_b, patient=None):
        if self.use_chemberta:
            device = next(self.parameters()).device
            ea = self.encoder(drug_a.smiles, device)
            eb = self.encoder(drug_b.smiles, device)
        elif self.architecture_version == MODEL_ARCHITECTURE_LEGACY:
            ea = self.encoder(drug_a.x, drug_a.edge_index, drug_a.batch)
            eb = self.encoder(drug_b.x, drug_b.edge_index, drug_b.batch)
        else:
            ea = self.encoder(drug_a.x, drug_a.edge_index, drug_a.edge_attr, drug_a.batch)
            eb = self.encoder(drug_b.x, drug_b.edge_index, drug_b.edge_attr, drug_b.batch)
        ta = torch.sigmoid(self.toxicity_head(ea))
        tb = torch.sigmoid(self.toxicity_head(eb))
        if patient is not None:
            g = self.patient_encoder(patient['age_band'], patient['sex'], patient['comorbidities'])
            ea, eb = ea*g, eb*g
            
        emb_sum = ea + eb
        emb_diff = torch.abs(ea - eb)
        features = [emb_sum, emb_diff]
        if self.use_toxicity_pair_features:
            features.extend([
                (ta + tb).unsqueeze(-1),
                torch.abs(ta - tb).unsqueeze(-1),
            ])
        combined = torch.cat(features, dim=1)
        
        return self.risk_classifier(combined).squeeze(-1), ta, tb
