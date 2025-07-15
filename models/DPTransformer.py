import torch
import torch.nn as nn
import torch.nn.functional as F

class DualPathTransformer(nn.Module):
    def __init__(self, seq_len, input_dim, output_dim, d_model=128, n_heads=4, window_size=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))  # B, T, D
        
        # 双路径注意力 局部和全局
        self.local_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.global_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),  # 引入非线性激活
            nn.Tanh()   # 调整门控权重范围为 [-1, 1]
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)  # 额外的归一化层
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(d_model, output_dim)

        # 动态窗口大小
        self.window_size = window_size if window_size is not None else min(7, seq_len // 10)

    def create_local_mask(self, seq_len, window_size):
        """创建局部注意力掩码"""
        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 0
        return mask

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        
        x = self.input_proj(x) + self.pos_embed  # 加位置编码
        
        # 创新点1: 双路径注意力
        # 局部注意力 - 关注邻近时间步
        local_mask = self.create_local_mask(seq_len, self.window_size).to(device)
        local_out, _ = self.local_attn(x, x, x, attn_mask=local_mask)
        
        # 全局注意力 - 关注所有时间步
        global_out, _ = self.global_attn(x, x, x)
        
        # 创新点2: 门控融合机制
        # 自适应地融合局部和全局特征
        combined = torch.cat([local_out, global_out], dim=-1)  # [B, T, 2*d_model]
        gate_weights = self.gate(combined)  # [B, T, d_model]
        
        # 门控融合
        fused_attn = gate_weights * local_out + (1 - gate_weights) * global_out
        
        x = self.norm1(x + fused_attn)
        x = self.norm2(x + self.ffn(x))
        x = self.norm3(x)  # 额外的归一化
        
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pool(x).squeeze(-1)
        return self.out(x)
