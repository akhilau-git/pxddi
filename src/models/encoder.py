import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class MolecularEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels*heads, hidden_channels, heads=1)
    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()
        return global_mean_pool(x, batch)
