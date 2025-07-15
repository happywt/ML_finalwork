import torch
import torch.nn as nn

class SequenceTransformer(nn.Module):
    def __init__(self, seq_len, input_dim, output_dim, d_model=128, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))  # B, T, D
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed  # 加位置编码
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pool(x).squeeze(-1)
        return self.out(x)