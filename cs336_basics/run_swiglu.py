
import math
import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, os.path.dirname(os.getcwd()))
from cs336_basics.run_linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, eps: float = 1e-5, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.eps = eps
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = math.ceil((8 / 3 * d_model) / 64) * 64
        else:
            self.d_ff = d_ff
        self.linear_1 = Linear(self.d_model, self.d_ff)
        self.linear_2 = Linear(self.d_model, self.d_ff)
        self.linear_3 = Linear(self.d_ff, self.d_model)
        
    
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
