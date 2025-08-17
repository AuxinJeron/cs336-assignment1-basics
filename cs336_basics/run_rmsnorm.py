
import math
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        # self.scale.shape = (d_model)
        self.scale = nn.Parameter(torch.ones(d_model))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # x.shape = (batch, sequence, d_model)
        x = x.to(torch.float32)
        
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # x_normed.shape = (batch, sequence, d_model)
        x_normed = x / rms 
        x_normed = x_normed * self.scale
        return x_normed.to(in_dtype)
        
