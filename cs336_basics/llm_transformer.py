

import math
import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    """
    Args:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b =3 * std)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "batch sequence d_in, d_out d_in -> batch sequence d_out")    
    
    

class Embedding(nn.Module):
    """
    Args:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.d_model = embedding_dim
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(self.embeddings, mean=0.0, std=std, a=-3 * std, b =3 * std)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
    

class RMSNorm(nn.Module):
    """
    Args:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        # self.scale.shape = (d_model)
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # x.shape = (batch, sequence, d_model)
        x = x.to(torch.float32)
        
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # x_normed.shape = (batch, sequence, d_model)
        x_normed = x / rms 
        x_normed = x_normed * self.scale
        return x_normed.to(in_dtype)
    

class SwiGLU(nn.Module):
    """
    Args:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    """
    def __init__(self, d_model, d_ff=None, eps: float = 1e-5, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.eps = eps
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = math.ceil((8 / 3 * d_model) / 64) * 64
        else:
            self.d_ff = d_ff
        self.linear_1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.linear_2 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.linear_3 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch, sequence, d_model)
        # silu.shape = (batch, sequence, d_ff)
        linear_1_result = self.linear_1(x)
        silu_result = linear_1_result * torch.sigmoid(linear_1_result)
        linear_2_result = self.linear_2(x)
        # linear_1_result.shape = (batch, sequence, d_ff)
        # linear_2_result.shape = (batch, sequence, d_ff)
        linear_3_result = self.linear_3(silu_result * linear_2_result)
        return linear_3_result
    
    
    
class RoPE(nn.Module):
    """
    Args:
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super(RoPE, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device("cpu")
        
        # angles.shape == (d_k // 2)
        # NOTE: The range mentioned in the assigned pdf, e.g. [1, dk // 2) is inaccurate 
        # since the test cannot be passed
        angles = theta ** (2 * torch.arange(0, d_k // 2, device=device, dtype=torch.float32) / d_k)
        angles = 1 / angles
        # positions.shape ==  (max_seq_len)
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        # angles.shape == (max_seq_len, d_k // 2)
        angles = torch.outer(positions, angles)
        # sin/cos.shape == (max_seq_len, d_k // 2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        
        # self.rot_mat.shape == (sequence, d_k, d_k)
        rot_mat = torch.zeros(max_seq_len, d_k, d_k, device=device, dtype=torch.float32)
        for k in range(0, d_k // 2):
            rot_mat[:, 2 * k, 2 * k] = cos[:, k]
            rot_mat[:, 2 * k, 2 * k + 1] = -sin[:, k]
            rot_mat[:, 2 * k + 1, 2 * k] = sin[:, k]
            rot_mat[:, 2 * k + 1, 2 * k + 1] = cos[:, k]
        self.register_buffer("rot_mat", rot_mat.to(self.device), persistent=False)
        
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x.shape == (batch, sequence, heads, d_in)
        # token_position.shape == (batch, heads, sequence)
        # return.shape == (batch, sequence, heads, d_k)
        rot_mat = self.rot_mat[token_positions]
        return einsum(x, rot_mat, "... d_in, ... d_out d_in -> ... d_out")

    
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    # K.shape = (batch_size, ..., seq_k, d_k)
    # Q.shape = (batch_size, ..., seq_q, d_q)
    # V.shape = (batch_size, ..., seq_v, d_v)
    # seq_k == seq_v 
    # mask.shape = (seq_q, seq_k)
    # output.shape = (batch_size, ..., seq_q, d_v)
    d_k, d_v = K.shape[-1], V.shape[-1]
    pre_softmax = einsum(Q, K, "batch ... seq_q d_k, batch ... seq_k d_k -> batch ... seq_q seq_k")
    # pre_softmax.shape = (batch, seq_q, seq_k)
    pre_softmax = pre_softmax / math.sqrt(d_k)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
    softmax = torch.softmax(pre_softmax, dim=-1)
    return einsum(softmax, V, "batch ... seq_q seq_k, batch ... seq_k d_v -> batch ... seq_q d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Args:
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention
    """
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
    
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None, rope=None):
        super(TransformerBlock, self).__init__()
        # First layer multi-head attention 
        self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.multi_head_self_attention = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype, rope=rope)
        # Second layer feed forward 
        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.feed_forward = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    # NOTE: Don't miss the token positions
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        attn_output = x + self.multi_head_self_attention(self.rms_norm_1(x), token_positions)
        # NOTE: Don't do ffn_output = x + self.feed_forward(self.rms_norm_2(attn_output))
        ffn_output = attn_output + self.feed_forward(self.rms_norm_2(attn_output))
        return ffn_output
    
    
class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size:int, 
        context_length: int, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int,
        rope_theta: float, 
        device=None, 
        dtype=None
    ):
        super(TransformerLM, self).__init__()
        # Emebdding layer
        # TODO: Revisit the embedding dimensions
        
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Transformer blocks
        # TODO: Revisit the theta value
        rope = RoPE(rope_theta, d_model // num_heads, context_length)
        # (batch, seq, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope=rope)
                for _ in range(num_layers)
            ]
        )
        # (batch, seq, d_model)
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        # (batch, seq, vocab_size)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        token_embeddings = self.embedding(token_ids)
        # Note: The token positions need to be injected into each layer
        transformer_block_input = token_embeddings
        for transformer_block in self.transformer_blocks:
            transformer_block_output = transformer_block(transformer_block_input, token_positions)
            transformer_block_input = transformer_block_output
        norm_output = self.norm(transformer_block_output)
        linear_output = self.linear(norm_output)
        # NOTE: Softmax would be applied after transformer LM 
        return linear_output 
