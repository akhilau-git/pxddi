import torch.nn as nn
class ToxicityHead(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(hidden_channels,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, emb): return self.classifier(emb).squeeze(-1)
