"""Reusable self-supervised warm-start components for the edge-aware encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from torch_geometric.nn import global_mean_pool

from .encoder import EdgeAwareMolecularEncoder


PRETRAINING_ARTIFACT_TYPE = 'chembl_edge_aware_contrastive_encoder_v1'


def augment_edge_aware_batch(
    graph_batch,
    *,
    atom_feature_mask_rate: float,
    bond_feature_mask_rate: float,
):
    """Mask feature values while retaining molecular connectivity in each view."""
    if not 0 <= atom_feature_mask_rate < 1 or not 0 <= bond_feature_mask_rate < 1:
        raise ValueError('Feature mask rates must lie in [0, 1).')
    if not hasattr(graph_batch, 'edge_attr'):
        raise ValueError('Contrastive pretraining requires rich graph edge_attr features.')
    augmented = graph_batch.clone()
    if atom_feature_mask_rate:
        atom_mask = torch.rand(
            augmented.x.size(0), device=augmented.x.device
        ) < atom_feature_mask_rate
        augmented.x[atom_mask] = 0.0
    if bond_feature_mask_rate:
        bond_mask = torch.rand(
            augmented.edge_attr.size(0), device=augmented.edge_attr.device
        ) < bond_feature_mask_rate
        augmented.edge_attr[bond_mask] = 0.0
    return augmented


class EdgeAwareContrastivePretrainer(nn.Module):
    """Edge-aware encoder plus projection head for unlabeled graph contrastive loss."""

    def __init__(self, in_channels: int, edge_feature_dim: int, hidden_channels: int):
        super().__init__()
        self.encoder = EdgeAwareMolecularEncoder(
            in_channels, edge_feature_dim, hidden_channels
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, graph_batch: Any) -> torch.Tensor:
        nodes = self.encoder.encode_nodes(
            graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr
        )
        graph_embeddings = global_mean_pool(nodes, graph_batch.batch)
        return functional.normalize(self.projection(graph_embeddings), dim=1)


def bidirectional_nt_xent_loss(
    first_view: torch.Tensor,
    second_view: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Return an in-batch symmetric contrastive objective for paired views."""
    if first_view.ndim != 2 or second_view.ndim != 2:
        raise ValueError('Contrastive views must be two-dimensional tensors.')
    if first_view.shape != second_view.shape or first_view.shape[0] < 2:
        raise ValueError('Contrastive views must have equal shape and batch size at least two.')
    if temperature <= 0:
        raise ValueError('temperature must be positive.')
    labels = torch.arange(first_view.size(0), device=first_view.device)
    first_to_second = first_view @ second_view.transpose(0, 1) / temperature
    second_to_first = second_view @ first_view.transpose(0, 1) / temperature
    return 0.5 * (
        functional.cross_entropy(first_to_second, labels)
        + functional.cross_entropy(second_to_first, labels)
    )


def load_pretrained_edge_aware_encoder(
    encoder: EdgeAwareMolecularEncoder,
    path: str | Path,
    *,
    expected_in_channels: int,
    expected_edge_feature_dim: int,
    expected_hidden_channels: int,
    expected_split_audit: dict[str, Any] | None = None,
    map_location: str | torch.device = 'cpu',
) -> dict[str, Any]:
    """Load a checked ChEMBL encoder checkpoint without loading prediction heads."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Pretrained encoder checkpoint was not found: {checkpoint_path}')
    bundle = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError('Pretrained encoder checkpoint must contain a dictionary.')
    if bundle.get('artifact_type') != PRETRAINING_ARTIFACT_TYPE:
        raise ValueError(
            'Checkpoint is not a compatible ChEMBL edge-aware pretraining artifact.'
        )
    configuration = bundle.get('encoder_configuration')
    if not isinstance(configuration, dict):
        raise ValueError('Pretrained encoder checkpoint is missing encoder_configuration.')
    expected = {
        'in_channels': expected_in_channels,
        'edge_feature_dim': expected_edge_feature_dim,
        'hidden_channels': expected_hidden_channels,
    }
    mismatches = {
        key: {'expected': value, 'observed': configuration.get(key)}
        for key, value in expected.items()
        if configuration.get(key) != value
    }
    if mismatches:
        raise ValueError(f'Pretrained encoder schema does not match the DDI model: {mismatches}.')
    if bundle.get('pretraining_leakage_policy') != 'exclude_all_non_train_twosides_structures_v1':
        raise ValueError('Pretrained encoder lacks the required strict leakage-exclusion policy.')
    observed_split_audit = bundle.get('pretraining_split_audit')
    if not isinstance(observed_split_audit, dict):
        raise ValueError('Pretrained encoder checkpoint is missing pretraining_split_audit.')
    if expected_split_audit is not None:
        split_mismatches = {
            key: {'expected': value, 'observed': observed_split_audit.get(key)}
            for key, value in expected_split_audit.items()
            if observed_split_audit.get(key) != value
        }
        if split_mismatches:
            raise ValueError(
                'Pretrained encoder split contract does not match this DDI run: '
                f'{split_mismatches}.'
            )
    state_dict = bundle.get('encoder_state_dict')
    if not isinstance(state_dict, dict):
        raise ValueError('Pretrained encoder checkpoint is missing encoder_state_dict.')
    encoder.load_state_dict(state_dict, strict=True)
    return {
        'path': str(checkpoint_path),
        'artifact_type': bundle['artifact_type'],
        'encoder_configuration': configuration,
        'pretraining_leakage_policy': bundle['pretraining_leakage_policy'],
        'source_corpus': bundle.get('source_corpus'),
        'pretraining_split_audit': observed_split_audit,
    }
