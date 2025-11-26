import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
class AdaptiveDimTransformer(nn.Module):
    def __init__(self, 
                 out_channels,
                 input_dim: int = 20000,
                 embed_dim: int = 100,   
                 num_layers: int = 3,  
                 base_dim: int = 512,   
                 dropout: float = 0.2
                 ):
        super().__init__()
    
        self.dim_sequence = self._gen_dim_sequence(base_dim, embed_dim, num_layers)
    
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.dim_sequence[0]),
            nn.GELU(),
            nn.LayerNorm(self.dim_sequence[0])
        )

        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(
                CompressionLayer(
                    in_dim=self.dim_sequence[i],
                    out_dim=self.dim_sequence[i+1],
                    dropout=dropout
                )
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, out_channels) 
        )

    def _gen_dim_sequence(self, base_dim, target_dim, num_layers):
        dims = [base_dim]
        current_dim = base_dim
        for _ in range(num_layers-1):
            next_dim = max(current_dim // 2, target_dim)
            dims.append(next_dim)
            current_dim = next_dim

        dims.append(target_dim)
        return dims

    def forward(self, x, stage):
        x = self.input_proj(x)

        for layer in self.encoder:
            x = layer(x)

        embed = x

        logits = self.classifier(embed)
    
        return  embed,logits

class CompressionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()

        original_heads = max(1, in_dim // 64)

        num_heads = original_heads
        while num_heads > 0:
            if in_dim % num_heads == 0:
                break
            num_heads -= 1
        if num_heads == 0:
            num_heads = 1 
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dim_reduction = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim*4),
            nn.ReLU(),
            nn.Linear(out_dim*4, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.dim_reduction(attn_out) + self.residual(x)
        return self.ffn(x)
