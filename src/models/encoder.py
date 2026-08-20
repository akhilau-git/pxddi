import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool

class MolecularEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels*heads, hidden_channels, heads=1)
    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()
        return global_mean_pool(x, batch)


class EdgeAwareMolecularEncoder(nn.Module):
    """GATv2 encoder that consumes bond order, stereo, and ring edge features."""
    def __init__(self, in_channels, edge_channels, hidden_channels=64, heads=2):
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            edge_dim=edge_channels,
        )
        self.gat2 = GATv2Conv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            edge_dim=edge_channels,
        )

    def encode_nodes(self, x, edge_index, edge_attr):
        """Encode atoms while retaining node embeddings for pair interaction."""
        x = self.gat1(x, edge_index, edge_attr).relu()
        return self.gat2(x, edge_index, edge_attr).relu()

    def forward(self, x, edge_index, edge_attr, batch):
        return global_mean_pool(self.encode_nodes(x, edge_index, edge_attr), batch)


class CrossDrugAttention(nn.Module):
    """Pair-isolated atom-level attention between the two drugs in one batch row.

    The same query, key, value, and update projections are shared in both
    directions.  For each pair the module computes A-to-B and B-to-A messages
    separately, then the caller applies an order-independent pair head.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.query = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.key = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.value = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.update = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
        )
        self.scale = hidden_channels ** -0.5

    def _attend(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention_weights(source, target)
        context = attention_weights @ self.value(target)
        return self.update(torch.cat((source, context), dim=1))

    def attention_weights(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Return normalized source-to-target attention for one drug pair.

        The values are exposed solely for offline candidate analysis.  They
        describe an internal model association, not a causal chemical
        interaction or a validated molecular mechanism.
        """
        attention_logits = self.query(source) @ self.key(target).transpose(0, 1)
        return torch.softmax(attention_logits * self.scale, dim=1)

    def attention_maps(
        self,
        node_embeddings_a: torch.Tensor,
        batch_a: torch.Tensor,
        node_embeddings_b: torch.Tensor,
        batch_b: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return pair-isolated A-to-B and B-to-A matrices for offline audit.

        Each list item corresponds to one aligned row in the mini-batch.  The
        method shares the same alignment checks as ``forward`` so an
        explanation cannot silently combine atoms from different drug pairs.
        """
        pair_ids_a = torch.unique(batch_a, sorted=True)
        pair_ids_b = torch.unique(batch_b, sorted=True)
        if not torch.equal(pair_ids_a, pair_ids_b):
            raise ValueError('Cross-drug attention requires aligned graph batches.')

        maps_a_to_b, maps_b_to_a = [], []
        for pair_id in pair_ids_a:
            atoms_a = node_embeddings_a[batch_a == pair_id]
            atoms_b = node_embeddings_b[batch_b == pair_id]
            if atoms_a.numel() == 0 or atoms_b.numel() == 0:
                raise ValueError('Cross-drug attention received an empty molecular graph.')
            maps_a_to_b.append(self.attention_weights(atoms_a, atoms_b))
            maps_b_to_a.append(self.attention_weights(atoms_b, atoms_a))
        return maps_a_to_b, maps_b_to_a

    def forward(
        self,
        node_embeddings_a: torch.Tensor,
        batch_a: torch.Tensor,
        node_embeddings_b: torch.Tensor,
        batch_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one interaction-aware embedding per aligned drug pair.

        The loop is intentionally over batch rows, not nodes.  It prevents an
        atom from one DDI pair attending to a drug in another DDI pair.
        """
        pair_ids_a = torch.unique(batch_a, sorted=True)
        pair_ids_b = torch.unique(batch_b, sorted=True)
        if not torch.equal(pair_ids_a, pair_ids_b):
            raise ValueError('Cross-drug attention requires aligned graph batches.')

        pooled_a, pooled_b = [], []
        for pair_id in pair_ids_a:
            atoms_a = node_embeddings_a[batch_a == pair_id]
            atoms_b = node_embeddings_b[batch_b == pair_id]
            if atoms_a.numel() == 0 or atoms_b.numel() == 0:
                raise ValueError('Cross-drug attention received an empty molecular graph.')
            pooled_a.append(self._attend(atoms_a, atoms_b).mean(dim=0))
            pooled_b.append(self._attend(atoms_b, atoms_a).mean(dim=0))
        return torch.stack(pooled_a), torch.stack(pooled_b)

class MolecularEncoderChemBERTa(nn.Module):
    def __init__(self, hidden_channels=64, model_name='DeepChem/ChemBERTa-77M-MTR'):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name)
        # Freeze ChemBERTa base layers to save memory/time, or leave unfreezed to fine-tune
        for param in self.chemberta.parameters():
            param.requires_grad = False
            
        self.project = nn.Linear(self.chemberta.config.hidden_size, hidden_channels)
        
    def forward(self, smiles_list, device):
        # We tokenize on the fly
        encoded = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors='pt', max_length=128)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Disable gradients for the frozen base model to save memory
        with torch.no_grad():
            outputs = self.chemberta(**encoded)
        
        cls_rep = outputs.last_hidden_state[:, 0, :] 
        return self.project(cls_rep).relu()
