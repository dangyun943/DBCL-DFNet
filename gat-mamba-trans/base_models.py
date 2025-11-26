from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

from utils import xavier_init, bias_init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels: List[int], out_channels, num_layers, num_heads: int = 1, bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        self.convs = nn.ModuleList()
        for layer_id in range(num_layers):
            conv = HeteroConv({
                ('patient', 'similar', 'patient'): GATConv(-1, hidden_channels[layer_id], heads=num_heads, concat=False,
                                                           dropout=dropout, add_self_loops=True, edge_dim=1),
                ('feature', 'similar', 'feature'): GATConv(-1, hidden_channels[layer_id], heads=num_heads, concat=False,
                                                           dropout=dropout, add_self_loops=True, edge_dim=2),
                ('feature', 'belong', 'patient'): GATConv((-1, -1), hidden_channels[layer_id], heads=num_heads,
                                                          concat=False, dropout=dropout, add_self_loops=False,
                                                          edge_dim=None),
            }, aggr='mean')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels[num_layers - 1], out_channels, bias=bias, weight_initializer='glorot')

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['patient'])


class DynamicMultiheadAttentionFusion(nn.Module):
    def __init__(self, num_modalities: int, num_classes: int,input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.num_heads = num_heads
        if hidden_dim % self.num_heads != 0:
            hidden_dim = ((hidden_dim + self.num_heads - 1) // self.num_heads) * self.num_heads        
        # 模态特征增强投影
        self.hidden_dim = hidden_dim
        
        self.modality_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim*2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_modalities)
        ])
        
        # 动态多头注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 自适应权重生成网络
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # 输出变换层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, multimodal_input: List[torch.Tensor],multimodal_input1: List[torch.Tensor]) -> torch.Tensor:
        
        # 特征增强与维度对齐
        projected = []
        for i in range(self.num_modalities):
            combined = torch.cat([multimodal_input[i], multimodal_input1[i]], dim=1)
            proj = self.modality_proj[i](combined)  # [B, hidden_dim]
            projected.append(proj.unsqueeze(1))  # [B, 1, D]
        
        # 构建多模态特征序列
        features = torch.cat(projected, dim=1)  # [B, M, D]


        
        # 跨模态注意力交互
        attn_output, _ = self.cross_attn(
            query=features,
            key=features,
            value=features
        )  # [B, M, D]
        
        # 动态权重生成
        weight = self.weight_net(features.flatten(start_dim=1))  # [B, M]
        weighted_output = (attn_output * weight.unsqueeze(-1)).sum(dim=1)  # [B, D]
        
        # 最终输出变换
        return self.output_layer(weighted_output)  # [B, C]


def xavier_init(module):
    """Xavier初始化权重"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def bias_init(module):
    """偏置初始化"""
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


class VCDN(nn.Module):
    def __init__(self, num_modalities: int, num_classes: int,
                 input_dim: int, hidden_dim: int = 32) -> None:
        """
        VCDN模型 - 修改版

        参数:
        num_modalities: 模态数量
        num_classes: 分类类别数
        input_dim: 每个模态的输入特征维度
        hidden_dim: VCDN隐藏层维度
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes

        # 模态特征投影层 - 处理每个模态的两个特征
        self.modality_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * 2, num_classes),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])

        # VCDN核心部分
        self.vcdn_model = nn.Sequential(
            nn.Linear(pow(self.num_classes, self.num_modalities), hidden_dim),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """应用初始化"""
        self.modality_projectors.apply(xavier_init)
        self.modality_projectors.apply(bias_init)
        self.vcdn_model.apply(xavier_init)
        self.vcdn_model.apply(bias_init)

    def forward(self, multimodal_input: List[torch.Tensor],
                multimodal_input1: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播，处理两个特征序列

        参数:
        - multimodal_input: 第一个特征序列 (每个模态: [B, input_dim])
        - multimodal_input1: 第二个特征序列 (每个模态: [B, input_dim])

        返回:
        - 分类结果 [B, num_classes]
        """
        # 1. 投影每个模态的特征并应用sigmoid
        projected_modalities = []
        for i in range(self.num_modalities):
            # 拼接同一模态的两个特征
            combined = torch.cat([multimodal_input[i], multimodal_input1[i]], dim=1)
            # 投影并应用sigmoid
            proj = self.modality_projectors[i](combined)  # [B, num_classes]
            projected_modalities.append(proj)

        # 2. 执行特征张量积计算
        x = torch.reshape(
            torch.matmul(
                projected_modalities[0].unsqueeze(-1),  # [B, num_classes, 1]
                projected_modalities[1].unsqueeze(1)  # [B, 1, num_classes]
            ),
            (-1, pow(self.num_classes, 2), 1),  # [B, num_classes^2, 1]
        )

        # 3. 处理更多模态（如果存在）
        for modality in range(2, self.num_modalities):
            x = torch.reshape(
                torch.matmul(
                    x,  # [B, num_classes^{modality}, 1]
                    projected_modalities[modality].unsqueeze(1)  # [B, 1, num_classes]
                ),
                (-1, pow(self.num_classes, modality + 1), 1)
            )

        # 4. 准备VCDN输入
        input_tensor = torch.reshape(x, (-1, pow(self.num_classes, self.num_modalities)))

        # 5. 通过VCDN模型
        return self.vcdn_model(input_tensor)