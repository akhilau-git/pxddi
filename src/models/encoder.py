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

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr).relu()
        x = self.gat2(x, edge_index, edge_attr).relu()
        return global_mean_pool(x, batch)

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
