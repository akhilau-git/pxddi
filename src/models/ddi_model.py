from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .encoder import CrossDrugAttention, EdgeAwareMolecularEncoder, MolecularEncoder
from .toxicity_model import ToxicityHead
from .patient_context import PatientContextEncoder


MODEL_ARCHITECTURE_LEGACY = 'legacy_gat_v1'
MODEL_ARCHITECTURE_EDGE_AWARE = 'edge_aware_gat_v2'
MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE = 'motif_edge_aware_gat_v1'
MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE = 'cross_attention_edge_aware_gat_v1'
MODEL_ARCHITECTURE_GRAPH_FP_FUSION = 'graph_fp_fusion_v1'
MODEL_ARCHITECTURE_AUDITDDI_MEMORY = 'auditddi_memory_fusion_v1'
MODEL_ARCHITECTURE_MULTIMODAL = 'auditddi_multimodal_v1'
MODEL_ARCHITECTURE_ABLATION_GENES = 'auditddi_ablation_genes'
MODEL_ARCHITECTURE_ABLATION_FAERS = 'auditddi_ablation_faers'


def architecture_uses_edge_features(architecture_version: str) -> bool:
    """Return whether a checkpoint consumes rich atom and bond graph features."""
    return architecture_version in {
        MODEL_ARCHITECTURE_EDGE_AWARE,
        MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE,
        MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
        MODEL_ARCHITECTURE_GRAPH_FP_FUSION,
        MODEL_ARCHITECTURE_AUDITDDI_MEMORY,
        MODEL_ARCHITECTURE_MULTIMODAL,
        MODEL_ARCHITECTURE_ABLATION_GENES,
        MODEL_ARCHITECTURE_ABLATION_FAERS,
    }


def architecture_requires_motif_features(architecture_version: str) -> bool:
    """Return whether a checkpoint needs the experimental motif graph field."""
    return architecture_version == MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE


def architecture_requires_cross_drug_attention(architecture_version: str) -> bool:
    """Return whether a checkpoint needs pair-isolated atom-level attention."""
    return architecture_version == MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE


def architecture_requires_fingerprint_features(architecture_version: str) -> bool:
    """Return whether a checkpoint needs molecular ECFP fingerprints."""
    return architecture_version in {
        MODEL_ARCHITECTURE_GRAPH_FP_FUSION,
        MODEL_ARCHITECTURE_AUDITDDI_MEMORY,
        MODEL_ARCHITECTURE_MULTIMODAL,
        MODEL_ARCHITECTURE_ABLATION_GENES,
        MODEL_ARCHITECTURE_ABLATION_FAERS,
    }


def architecture_requires_multimodal_features(architecture_version: str) -> bool:
    """Return whether a checkpoint consumes pharmacogenomic gene features."""
    return architecture_version in {
        MODEL_ARCHITECTURE_MULTIMODAL,
        MODEL_ARCHITECTURE_ABLATION_GENES,
    }


def model_from_checkpoint(checkpoint):
    """Construct either versioned GNN safely from checkpoint metadata with dimension auto-detection."""
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    classifier_w = state_dict.get('risk_classifier.0.weight')
    use_neighbor_mem = bool(checkpoint.get('use_neighbor_memory', False))
    use_geo = bool(checkpoint.get('use_geo_features', False))
    use_target = bool(checkpoint.get('use_target_encoder', any('target_encoder' in k for k in state_dict)))
    use_cross_modal = bool(checkpoint.get('use_cross_modal_attention', any('cross_modal_attention' in k for k in state_dict)))
    use_cross_drug = bool(checkpoint.get('use_cross_drug_attention', any('cross_drug_attention' in k for k in state_dict)))

    if classifier_w is not None and isinstance(classifier_w, torch.Tensor):
        h_dim = int(checkpoint.get('hidden_channels', 64))
        target_extra = (64 * 3) if use_target else 0
        base_dim = (h_dim + 128 + 64) * 3 + target_extra + 4
        diff = classifier_w.shape[1] - base_dim
        if diff in [3, 7, 11]:
            use_neighbor_mem = True
        if diff in [4, 7, 11]:
            use_geo = True

    arch = checkpoint.get('architecture_version', MODEL_ARCHITECTURE_LEGACY)
    use_clin_tox = bool(checkpoint.get('use_clinical_toxicity', arch in {MODEL_ARCHITECTURE_MULTIMODAL, MODEL_ARCHITECTURE_ABLATION_FAERS}))

    return PxDDIModel(
        in_channels=checkpoint['in_channels'],
        hidden_channels=checkpoint['hidden_channels'],
        use_chemberta=checkpoint.get('use_chemberta', False),
        architecture_version=arch,
        edge_feature_dim=checkpoint.get('edge_feature_dim'),
        use_toxicity_pair_features=checkpoint.get('use_toxicity_pair_features', True),
        motif_feature_dim=checkpoint.get('motif_feature_dim'),
        motif_hidden_channels=checkpoint.get('motif_hidden_channels'),
        use_neighbor_memory=use_neighbor_mem,
        gene_feature_dim=checkpoint.get('gene_feature_dim', 50),
        gene_hidden_channels=checkpoint.get('gene_hidden_channels', 64),
        use_clinical_toxicity=use_clin_tox,
        num_side_effects=checkpoint.get('num_side_effects', 1),
        use_cross_modal_attention=use_cross_modal,
        use_cross_drug_attention=use_cross_drug,
        use_target_encoder=use_target,
        target_feature_dim=checkpoint.get('target_feature_dim', 50),
        target_hidden_channels=checkpoint.get('target_hidden_channels', 64),
        use_geo_features=use_geo,
        geo_dim=checkpoint.get('geo_dim', 2),
        memory_dropout=float(checkpoint.get('memory_dropout', 0.50)),
    )


class CrossModalGeneAttention(nn.Module):
    """Pair-isolated cross-modal attention between molecular graph and pharmacogenomic gene vectors."""

    def __init__(self, mol_dim: int, gene_dim: int, hidden_dim: int = 64, n_heads: int = 2):
        super().__init__()
        self.mol_proj = nn.Linear(mol_dim, hidden_dim)
        self.gene_proj = nn.Linear(gene_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        mol_emb: torch.Tensor,
        gene_vec: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.mol_proj(mol_emb).unsqueeze(1)
        k = v = self.gene_proj(gene_vec).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, k, v)
        attn_out = attn_out.squeeze(1)
        gated = self.gate(attn_out) * attn_out
        if mask is not None:
            gated = gated * mask.view(-1, 1)
        return self.norm(gated)


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
        use_neighbor_memory=False,
        gene_feature_dim=50,
        gene_hidden_channels=64,
        use_clinical_toxicity=False,
        num_side_effects=1,
        use_cross_modal_attention=False,
        **kwargs,
    ):
        super().__init__()
        self.num_side_effects = num_side_effects
        self.use_cross_modal_attention = use_cross_modal_attention or kwargs.get('use_cross_modal_attention', False)
        self.use_chemberta = use_chemberta
        self.architecture_version = architecture_version
        self.use_toxicity_pair_features = use_toxicity_pair_features
        self.gene_feature_dim = gene_feature_dim
        self.gene_hidden_channels = gene_hidden_channels
        self.memory_dropout = float(kwargs.get('memory_dropout', 0.50))
        self.embedding_noise_std = float(kwargs.get('embedding_noise_std', 0.0))
        self.use_clinical_toxicity = use_clinical_toxicity or (
            architecture_version in {MODEL_ARCHITECTURE_MULTIMODAL, MODEL_ARCHITECTURE_ABLATION_FAERS}
        )

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

        if architecture_requires_fingerprint_features(architecture_version):
            self.fp_encoder = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
            )
        else:
            self.fp_encoder = None

        if architecture_requires_multimodal_features(architecture_version):
            self.gene_encoder = nn.Sequential(
                nn.Linear(gene_feature_dim, gene_hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.25),
            )
            self.gene_gate = nn.Sequential(
                nn.Linear(gene_hidden_channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.gene_encoder = None
            self.gene_gate = None

        self.use_cross_drug_attention = bool(
            kwargs.get('use_cross_drug_attention', False)
            or architecture_requires_cross_drug_attention(architecture_version)
        )
        self.cross_drug_attention = (
            CrossDrugAttention(hidden_channels)
            if self.use_cross_drug_attention
            else None
        )

        self.use_target_encoder = bool(kwargs.get('use_target_encoder', False))
        self.target_feature_dim = kwargs.get('target_feature_dim', 50)
        self.target_hidden_channels = kwargs.get('target_hidden_channels', 64)
        if self.use_target_encoder and architecture_requires_multimodal_features(architecture_version):
            self.target_encoder = nn.Sequential(
                nn.Linear(self.target_feature_dim, self.target_hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.25),
            )
            self.target_gate = nn.Sequential(
                nn.Linear(self.target_hidden_channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.target_encoder = None
            self.target_gate = None

        self.toxicity_head = ToxicityHead(hidden_channels)
        self.patient_encoder = PatientContextEncoder(n_comorbidities, hidden_channels)
        self.uses_multiplicative_fusion = (
            architecture_version != MODEL_ARCHITECTURE_LEGACY
        )
        motif_dim = int(motif_hidden_channels) if (self.motif_encoder is not None and motif_hidden_channels is not None) else 0
        gene_dim = int(gene_hidden_channels) if (self.gene_encoder is not None and gene_hidden_channels is not None) else 0
        target_dim = int(self.target_hidden_channels) if (self.target_encoder is not None) else 0
        pair_feature_multiplier = 3 if self.uses_multiplicative_fusion else 2
        pair_embedding_channels = (
            hidden_channels
            + motif_dim
            + (hidden_channels if self.cross_drug_attention is not None else 0)
            + (128 if self.fp_encoder is not None else 0)
            + gene_dim
            + target_dim
        )
        self.use_neighbor_memory = (
            use_neighbor_memory or architecture_version == MODEL_ARCHITECTURE_AUDITDDI_MEMORY
        )
        self.use_geo_features = bool(
            kwargs.get('use_geo_features', False)
            and architecture_requires_multimodal_features(architecture_version)
        )
        self.geo_dim = kwargs.get('geo_dim', 2)
        risk_input_channels = pair_embedding_channels * pair_feature_multiplier + (
            2 if use_toxicity_pair_features else 0
        ) + (
            3 if self.use_neighbor_memory else 0
        ) + (
            2 if self.use_clinical_toxicity else 0
        ) + (
            (self.geo_dim * 2) if self.use_geo_features else 0
        )
        if self.use_cross_modal_attention and architecture_requires_multimodal_features(architecture_version):
            self.cross_modal_attention = CrossModalGeneAttention(
                mol_dim=hidden_channels,
                gene_dim=gene_feature_dim,
                hidden_dim=gene_hidden_channels,
            )
        else:
            self.cross_modal_attention = None

        self.risk_classifier = nn.Sequential(
            nn.Linear(risk_input_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_side_effects),
        )

    def load_pretrained_encoder(self, checkpoint_path: str | Path) -> None:
        """Load weights from a self-supervised pretrained encoder checkpoint."""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_p}")
        ckpt = torch.load(ckpt_p, map_location=next(self.parameters()).device)
        state_dict = ckpt.get('encoder_state_dict', ckpt.get('model_state_dict', ckpt))
        encoder_dict = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items()
            if 'encoder' in k or k in self.encoder.state_dict()
        }
        self.encoder.load_state_dict(encoder_dict if encoder_dict else state_dict, strict=False)
        print(f"Successfully loaded pretrained encoder weights from: {ckpt_p}")

    def forward(
        self,
        drug_a,
        drug_b,
        patient=None,
        memory_features=None,
        fp_a=None,
        fp_b=None,
        gene_a=None,
        gene_b=None,
        gene_mask_a=None,
        gene_mask_b=None,
        clinical_tox_a=None,
        clinical_tox_b=None,
        clinical_tox_mask_a=None,
        clinical_tox_mask_b=None,
        target_a=None,
        target_b=None,
        target_mask_a=None,
        target_mask_b=None,
        geo_a=None,
        geo_b=None,
        geo_mask_a=None,
        geo_mask_b=None,
        **kwargs,
    ):
        cross_a: torch.Tensor | None = None
        cross_b: torch.Tensor | None = None
        if self.use_chemberta:
            device = next(self.parameters()).device
            ea = self.encoder(drug_a.smiles, device)
            eb = self.encoder(drug_b.smiles, device)
        elif self.architecture_version == MODEL_ARCHITECTURE_LEGACY:
            ea = self.encoder(drug_a.x, drug_a.edge_index, drug_a.batch)
            eb = self.encoder(drug_b.x, drug_b.edge_index, drug_b.batch)
        elif self.cross_drug_attention is not None:
            if not isinstance(self.encoder, EdgeAwareMolecularEncoder):
                raise TypeError('Expected EdgeAwareMolecularEncoder for cross-drug attention.')
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

        if self.fp_encoder is not None:
            if fp_a is not None and fp_b is not None:
                enc_fp_a = self.fp_encoder(fp_a.float().view(-1, 1024))
                enc_fp_b = self.fp_encoder(fp_b.float().view(-1, 1024))
            elif hasattr(drug_a, 'fingerprint_features') and hasattr(drug_b, 'fingerprint_features'):
                enc_fp_a = self.fp_encoder(drug_a.fingerprint_features.float().view(-1, 1024))
                enc_fp_b = self.fp_encoder(drug_b.fingerprint_features.float().view(-1, 1024))
            else:
                enc_fp_a = torch.zeros((ea.size(0), 128), device=ea.device, dtype=ea.dtype)
                enc_fp_b = torch.zeros((eb.size(0), 128), device=eb.device, dtype=eb.dtype)
            ea_for_risk = torch.cat((ea_for_risk, enc_fp_a), dim=1)
            eb_for_risk = torch.cat((eb_for_risk, enc_fp_b), dim=1)

        if self.gene_encoder is not None and self.gene_gate is not None:
            if gene_a is not None and gene_b is not None:
                g_in_a = gene_a.float().view(-1, self.gene_feature_dim)
                g_in_b = gene_b.float().view(-1, self.gene_feature_dim)
                ga = self.gene_encoder(g_in_a)
                gb = self.gene_encoder(g_in_b)
                if self.cross_modal_attention is not None:
                    ga = ga + self.cross_modal_attention(ea, g_in_a, gene_mask_a)
                    gb = gb + self.cross_modal_attention(eb, g_in_b, gene_mask_b)
                ga_gate = self.gene_gate(ga)
                if gene_mask_a is not None:
                    ga_gate = ga_gate * gene_mask_a.view(-1, 1)
                ga_rep = ga_gate * ga

                gb_gate = self.gene_gate(gb)
                if gene_mask_b is not None:
                    gb_gate = gb_gate * gene_mask_b.view(-1, 1)
                gb_rep = gb_gate * gb
            else:
                ga_rep = torch.zeros((ea.size(0), self.gene_hidden_channels), device=ea.device, dtype=ea.dtype)
                gb_rep = torch.zeros((eb.size(0), self.gene_hidden_channels), device=eb.device, dtype=eb.dtype)
            ea_for_risk = torch.cat((ea_for_risk, ga_rep), dim=1)
            eb_for_risk = torch.cat((eb_for_risk, gb_rep), dim=1)

        if self.target_encoder is not None and self.target_gate is not None:
            if target_a is not None and target_b is not None:
                t_in_a = target_a.float().view(-1, self.target_feature_dim)
                t_in_b = target_b.float().view(-1, self.target_feature_dim)
                ta = self.target_encoder(t_in_a)
                tb = self.target_encoder(t_in_b)
                if self.cross_modal_attention is not None and t_in_a.size(-1) == self.cross_modal_attention.gene_proj.in_features:
                    ta = ta + self.cross_modal_attention(ea, t_in_a, target_mask_a)
                    tb = tb + self.cross_modal_attention(eb, t_in_b, target_mask_b)
                ta_gate = self.target_gate(ta)
                tb_gate = self.target_gate(tb)
                if target_mask_a is not None:
                    ta_gate = ta_gate * target_mask_a.view(-1, 1)
                if target_mask_b is not None:
                    tb_gate = tb_gate * target_mask_b.view(-1, 1)
                ta_rep = ta_gate * ta
                tb_rep = tb_gate * tb
            else:
                ta_rep = torch.zeros((ea.size(0), self.target_hidden_channels), device=ea.device, dtype=ea.dtype)
                tb_rep = torch.zeros((eb.size(0), self.target_hidden_channels), device=eb.device, dtype=eb.dtype)
            ea_for_risk = torch.cat((ea_for_risk, ta_rep), dim=1)
            eb_for_risk = torch.cat((eb_for_risk, tb_rep), dim=1)

        if self.cross_drug_attention is not None and cross_a is not None and cross_b is not None:
            ea_for_risk = torch.cat((ea_for_risk, cross_a), dim=1)
            eb_for_risk = torch.cat((eb_for_risk, cross_b), dim=1)

        emb_sum = ea_for_risk + eb_for_risk
        emb_diff = torch.abs(ea_for_risk - eb_for_risk)
        if self.uses_multiplicative_fusion:
            emb_prod = ea_for_risk * eb_for_risk
            features = [emb_sum, emb_diff, emb_prod]
        else:
            features = [emb_sum, emb_diff]
        if self.use_toxicity_pair_features:
            features.extend([
                (toxicity_a_probability + toxicity_b_probability).unsqueeze(-1),
                torch.abs(toxicity_a_probability - toxicity_b_probability).unsqueeze(-1),
            ])
        if self.use_clinical_toxicity:
            if clinical_tox_a is not None and clinical_tox_b is not None:
                features.append(torch.stack([clinical_tox_a.float().view(-1), clinical_tox_b.float().view(-1)], dim=1))
            else:
                features.append(torch.zeros((ea.size(0), 2), device=ea.device, dtype=ea.dtype))
        if self.training and self.embedding_noise_std > 0.0:
            ea = ea + torch.randn_like(ea) * self.embedding_noise_std
            eb = eb + torch.randn_like(eb) * self.embedding_noise_std

        if self.use_neighbor_memory:
            if memory_features is None:
                memory_features = torch.zeros(
                    (ea.size(0), 3), device=ea.device, dtype=ea.dtype
                )
            elif self.training and self.memory_dropout > 0.0:
                # Stochastic memory dropout prevents topological shortcut memorization
                keep_prob = max(1.0 - self.memory_dropout, 1e-4)
                mask = (torch.rand((memory_features.size(0), 1), device=memory_features.device) < keep_prob).float()
                memory_features = (memory_features * mask) / keep_prob
            features.append(memory_features)
        if self.use_geo_features:
            if geo_a is not None and geo_b is not None:
                g_a = geo_a.float().view(ea.size(0), -1)
                g_b = geo_b.float().view(eb.size(0), -1)
                if geo_mask_a is not None:
                    g_a = g_a * geo_mask_a.view(-1, 1)
                if geo_mask_b is not None:
                    g_b = g_b * geo_mask_b.view(-1, 1)
                features.append(torch.cat([g_a + g_b, torch.abs(g_a - g_b)], dim=1))
            else:
                features.append(torch.zeros((ea.size(0), self.geo_dim * 2), device=ea.device, dtype=ea.dtype))
        combined = torch.cat(features, dim=1)

        risk_out = self.risk_classifier(combined)
        if self.num_side_effects == 1:
            risk_out = risk_out.squeeze(-1)

        return (
            risk_out,
            toxicity_a_logits,
            toxicity_b_logits,
        )

    def get_drug_representations(
        self,
        drug,
        fp=None,
        gene=None,
        gene_mask=None,
    ) -> torch.Tensor:
        """Extract multi-modal drug embedding for diagnostic and contrastive analysis."""
        device = next(self.parameters()).device
        b = drug.batch if hasattr(drug, 'batch') and drug.batch is not None else torch.zeros(drug.x.size(0), dtype=torch.long, device=drug.x.device)
        e = self.encoder(drug.x.to(device), drug.edge_index.to(device), drug.edge_attr.to(device), b.to(device))
        reps = [e]
        if self.fp_encoder is not None and fp is not None:
            reps.append(self.fp_encoder(fp.float().view(-1, 1024).to(device)))
        if self.gene_encoder is not None and gene is not None:
            g = self.gene_encoder(gene.float().to(device))
            g_gate = self.gene_gate(g)
            if gene_mask is not None:
                g_gate = g_gate * gene_mask.view(-1, 1).to(device)
            reps.append(g_gate * g)
        return torch.cat(reps, dim=1)

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
        if not isinstance(self.encoder, EdgeAwareMolecularEncoder):
            raise TypeError('Expected EdgeAwareMolecularEncoder for cross-drug attention.')
        node_embeddings_a = self.encoder.encode_nodes(
            drug_a.x, drug_a.edge_index, drug_a.edge_attr
        )
        node_embeddings_b = self.encoder.encode_nodes(
            drug_b.x, drug_b.edge_index, drug_b.edge_attr
        )
        return self.cross_drug_attention.attention_maps(
            node_embeddings_a, drug_a.batch, node_embeddings_b, drug_b.batch
        )


AuditDDIModel = PxDDIModel
