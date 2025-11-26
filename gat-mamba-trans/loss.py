import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        sim_matrix = torch.mm(features1, features2.t()) / self.temperature
        labels = torch.arange(features1.size(0), device=features1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
