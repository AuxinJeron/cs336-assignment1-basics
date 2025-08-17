
import math
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.d_model = embedding_dim
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(self.embeddings, mean=0.0, std=std, a=-3 * std, b =3 * std)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
        
