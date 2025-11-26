import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
class AdaptiveDimTransformer(nn.Module):
    def __init__(self, 
                 out_channels,
                 input_dim: int = 20000,
                 embed_dim: int = 100,     # 最终嵌入维度
                 num_layers: int = 3,      # 压缩层数
                 base_dim: int = 512,      # 初始维度
                 dropout: float = 0.2
                 ):
        super().__init__()
        
        # 1. 动态生成维度序列
        self.dim_sequence = self._gen_dim_sequence(base_dim, embed_dim, num_layers)
        print(f"维度压缩路径: {self.dim_sequence}")
        
        # 2. 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.dim_sequence[0]),
            nn.GELU(),
            nn.LayerNorm(self.dim_sequence[0])
        )
        
        # 3. 压缩层堆叠
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(
                CompressionLayer(
                    in_dim=self.dim_sequence[i],
                    out_dim=self.dim_sequence[i+1],
                    dropout=dropout
                )
            )
        
        # 4. 分类输出
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, out_channels) 
        )

    def _gen_dim_sequence(self, base_dim, target_dim, num_layers):
        """生成维度压缩路径"""
        dims = [base_dim]
        current_dim = base_dim
        
        # 前n-1层按减半压缩
        for _ in range(num_layers-1):
            next_dim = max(current_dim // 2, target_dim)
            dims.append(next_dim)
            current_dim = next_dim
        
        # 最后一层强制对齐目标维度
        dims.append(target_dim)
        return dims

    def forward(self, x, stage):
        # 输入投影
        x = self.input_proj(x)
        
        # 逐层压缩
        for layer in self.encoder:
            x = layer(x)
        
        # 中间嵌入
        embed = x
        
        # 分类结果
        logits = self.classifier(embed)
    
        return  embed,logits

class CompressionLayer(nn.Module):
    """带维度压缩的Transformer层"""
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()

        # 动态计算可整除的注意力头数
        original_heads = max(1, in_dim // 64)
        
        # 寻找最大可整除的头数
        num_heads = original_heads
        while num_heads > 0:
            if in_dim % num_heads == 0:
                break
            num_heads -= 1
        if num_heads == 0:
            num_heads = 1  # 保证至少1个头
        
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,  # 动态头数
            dropout=dropout,
            batch_first=True
        )
        
        # 维度压缩
        self.dim_reduction = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )
        
        # 残差连接
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim*4),
            nn.ReLU(),
            nn.Linear(out_dim*4, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 自注意力
        attn_out, _ = self.self_attn(x, x, x)
        
        # 残差连接+维度压缩
        x = self.dim_reduction(attn_out) + self.residual(x)
        
        # 前馈网络
        return self.ffn(x)