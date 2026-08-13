import torch, torch.nn as nn
class PatientContextEncoder(nn.Module):
    def __init__(self, n_comorbidities=10, hidden_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2+n_comorbidities, 32), nn.ReLU(),
            nn.Linear(32, hidden_channels), nn.Sigmoid())
    def forward(self, age_band, sex, comorbidities):
        x = torch.cat([age_band.unsqueeze(-1), sex.unsqueeze(-1), comorbidities], dim=-1)
        return self.encoder(x)
