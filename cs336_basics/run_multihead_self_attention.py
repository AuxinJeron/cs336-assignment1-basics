
import torch
import torch.nn as nn
from einops import rearrange, einsum
import einx
from cs336_basics.run_linear import Linear
from cs336_basics.run_scaled_dot_production_attention import scaled_dot_product_attention
from cs336_basics.run_rope import RoPE


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, rope=None):
        super(CausalMultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_q = self.d_v = int(d_model / num_heads)
        self.linear_q = Linear(d_model, num_heads * self.d_q)
        self.linear_k = Linear(d_model, num_heads * self.d_k)
        self.linear_v = Linear(d_model, num_heads * self.d_v)
        self.linear_o = Linear(num_heads * self.d_v, d_model)
        self.rope = rope
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        # x.shape = (batch, seq, d_model)
        batch_size, seq_len, _ = x.shape
        # token_positions = torch.arange(seq_len)
        # Q.shape = (batch, num_heads, seq, d_q)
        Q = rearrange(self.linear_q(x), "b s (h d) -> b h s d", h=self.num_heads)
        # K.shape = (batch, num_heads, seq, d_k)
        K = rearrange(self.linear_k(x), "b s (h d) -> b h s d", h=self.num_heads)
        # V.shape = (batch, num_heads, seq, d_v)
        V = rearrange(self.linear_v(x), "b s (h d) -> b h s d", h=self.num_heads)
        
        if token_positions is not None and self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
                
        # mask.shape = (seq_q, seq_k)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        attention = scaled_dot_product_attention(Q, K, V, mask=mask)
        attention = rearrange(attention, "b h s d -> b s (h d)")
        
        return self.linear_o(attention)
