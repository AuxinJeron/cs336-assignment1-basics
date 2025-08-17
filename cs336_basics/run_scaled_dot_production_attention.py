
import math
import torch
import torch.nn as nn
from einops import einsum


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
