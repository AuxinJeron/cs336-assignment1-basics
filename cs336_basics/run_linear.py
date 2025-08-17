
import math
import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b =3 * std)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "batch sequence d_in, d_out d_in -> batch sequence d_out")    
