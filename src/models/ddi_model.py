import torch, torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .encoder import CrossDrugAttention, EdgeAwareMolecularEncoder, MolecularEncoder
from .toxicity_model import ToxicityHead
from .patient_context import PatientContextEncoder


MODEL_ARCHITECTURE_LEGACY = 'legacy_gat_v1'
MODEL_ARCHITECTURE_EDGE_AWARE = 'edge_aware_gat_v2'
MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE = 'motif_edge_aware_gat_v1'
MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE = 'cross_attention_edge_aware_gat_v1'


def architecture_uses_edge_features(architecture_version: str) -> bool:
    """Return whether a checkpoint consumes rich atom and bond graph features."""
    return architecture_version in {
        MODEL_ARCHITECTURE_EDGE_AWARE,
        MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE,
        MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
    }


def architecture_requires_motif_features(architecture_version: str) -> bool:
    """Return whether a checkpoint needs the experimental motif graph field."""
    return architecture_version == MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE


def architecture_requires_cross_drug_attention(architecture_version: str) -> bool:
    """Return whether a checkpoint needs pair-isolated atom-level attention."""
    return architecture_version == MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE


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
        motif_feature_dim=checkpoint.get('motif_feature_dim'),
        motif_hidden_channels=checkpoint.get('motif_hidden_channels'),
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
        motif_feature_dim=None,
        motif_hidden_channels=None,
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
        elif architecture_uses_edge_features(architecture_version):
            if edge_feature_dim is None or edge_feature_dim <= 0:
                raise ValueError('edge_feature_dim must be positive for the edge-aware GNN.')
            self.encoder = EdgeAwareMolecularEncoder(
                in_channels,
                edge_feature_dim,
                hidden_channels,
            )
        else:
            raise ValueError(f'Unknown model architecture: {architecture_version}.')
        self.motif_feature_dim = motif_feature_dim
        self.motif_hidden_channels = motif_hidden_channels
        if architecture_requires_motif_features(architecture_version):
            if motif_feature_dim is None or motif_feature_dim <= 0:
                raise ValueError('motif_feature_dim must be positive for the motif candidate.')
            if motif_hidden_channels is None or motif_hidden_channels <= 0:
                raise ValueError('motif_hidden_channels must be positive for the motif candidate.')
            self.motif_encoder = nn.Sequential(
                nn.Linear(motif_feature_dim, motif_hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        else:
            self.motif_encoder = None

        self.cross_drug_attention = (
            CrossDrugAttention(hidden_channels)
            if architecture_requires_cross_drug_attention(architecture_version)
            else None
        )

        self.toxicity_head = ToxicityHead(hidden_channels)
        self.patient_encoder = PatientContextEncoder(n_comorbidities, hidden_channels)
        pair_embedding_channels = hidden_channels + (
            motif_hidden_channels if self.motif_encoder is not None else 0
        ) + (hidden_channels if self.cross_drug_attention is not None else 0)
        risk_input_channels = pair_embedding_channels * 2 + (
            2 if use_toxicity_pair_features else 0
        )
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
        elif self.cross_drug_attention is not None:
            node_embeddings_a = self.encoder.encode_nodes(
                drug_a.x, drug_a.edge_index, drug_a.edge_attr
            )
            node_embeddings_b = self.encoder.encode_nodes(
                drug_b.x, drug_b.edge_index, drug_b.edge_attr
            )
            ea = global_mean_pool(node_embeddings_a, drug_a.batch)
            eb = global_mean_pool(node_embeddings_b, drug_b.batch)
            cross_a, cross_b = self.cross_drug_attention(
                node_embeddings_a, drug_a.batch, node_embeddings_b, drug_b.batch
            )
        else:
            ea = self.encoder(drug_a.x, drug_a.edge_index, drug_a.edge_attr, drug_a.batch)
            eb = self.encoder(drug_b.x, drug_b.edge_index, drug_b.edge_attr, drug_b.batch)
        # Keep raw logits for ``BCEWithLogitsLoss``.  The interaction head still
        # receives sigmoid-transformed toxicity features, preserving the legacy
        # checkpoint's interaction-risk computation exactly.
        toxicity_a_logits = self.toxicity_head(ea)
        toxicity_b_logits = self.toxicity_head(eb)
        toxicity_a_probability = torch.sigmoid(toxicity_a_logits)
        toxicity_b_probability = torch.sigmoid(toxicity_b_logits)
        if patient is not None:
            g = self.patient_encoder(patient['age_band'], patient['sex'], patient['comorbidities'])
            ea, eb = ea*g, eb*g
            
        if self.motif_encoder is not None:
            if not hasattr(drug_a, 'motif_features') or not hasattr(drug_b, 'motif_features'):
                raise ValueError('The motif candidate requires motif_features on both drug graphs.')
            motif_a = self.motif_encoder(drug_a.motif_features.float())
            motif_b = self.motif_encoder(drug_b.motif_features.float())
            ea_for_risk = torch.cat((ea, motif_a), dim=1)
            eb_for_risk = torch.cat((eb, motif_b), dim=1)
        else:
            ea_for_risk, eb_for_risk = ea, eb

        if self.cross_drug_attention is not None:
            ea_for_risk = torch.cat((ea_for_risk, cross_a), dim=1)
            eb_for_risk = torch.cat((eb_for_risk, cross_b), dim=1)

        emb_sum = ea_for_risk + eb_for_risk
        emb_diff = torch.abs(ea_for_risk - eb_for_risk)
        features = [emb_sum, emb_diff]
        if self.use_toxicity_pair_features:
            features.extend([
                (toxicity_a_probability + toxicity_b_probability).unsqueeze(-1),
                torch.abs(toxicity_a_probability - toxicity_b_probability).unsqueeze(-1),
            ])
        combined = torch.cat(features, dim=1)
        
        return (
            self.risk_classifier(combined).squeeze(-1),
            toxicity_a_logits,
            toxicity_b_logits,
        )

    def cross_drug_attention_maps(self, drug_a, drug_b):
        """Return pair-isolated attention maps for an offline candidate audit.

        This is intentionally separate from ``forward`` so training and API
        inference keep their stable three-output contract.  Consumers must
        label the returned weights as model-internal associations rather than
        validated molecular mechanisms.
        """
        if self.cross_drug_attention is None:
            raise ValueError(
                'Cross-drug attention maps are available only for the '
                'cross_attention_edge_aware_gat_v1 candidate.'
            )
        node_embeddings_a = self.encoder.encode_nodes(
            drug_a.x, drug_a.edge_index, drug_a.edge_attr
        )
        node_embeddings_b = self.encoder.encode_nodes(
            drug_b.x, drug_b.edge_index, drug_b.edge_attr
        )
        return self.cross_drug_attention.attention_maps(
            node_embeddings_a, drug_a.batch, node_embeddings_b, drug_b.batch
        )
