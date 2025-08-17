

import torch
import torch.nn as nn
from einops import einsum


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RoPE, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device("cpu")
        
        # angles.shape == (d_k // 2)
        # NOTE: The range mentioned in the assigned pdf, e.g. [1, dk // 2) is inaccurate 
        # since the test cannot be passed
        angles = theta ** (2 * torch.arange(0, d_k // 2, dtype=torch.float32) / d_k)
        angles = 1 / angles
        # positions.shape ==  (max_seq_len)
        positions = torch.arange(0, max_seq_len, dtype=torch.float32)
        # angles.shape == (max_seq_len, d_k // 2)
        angles = torch.outer(positions, angles)
        # sin/cos.shape == (max_seq_len, d_k // 2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        
        # self.rot_mat.shape == (sequence, d_k, d_k)
        rot_mat = torch.zeros(max_seq_len, d_k, d_k, dtype=torch.float32)
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
        
