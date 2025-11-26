import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """对比损失类"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """计算两个特征表示之间的对比损失"""
        # 特征归一化
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(features1, features2.t()) / self.temperature
        
        # 创建标签（对角线为正样本）
        labels = torch.arange(features1.size(0), device=features1.device)
        
        # 计算InfoNCE损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss