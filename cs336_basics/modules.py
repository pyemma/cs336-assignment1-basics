import math

import torch
import torch.nn as nn

# TODO: reimplement using einops
class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.w = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        
        # initialization
        std = 2 / (in_features + out_features)
        torch.nn.init.trunc_normal_(self.w.data, mean=0, std=std, a=-3*math.sqrt(std), b=3*math.sqrt(std))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w.T)


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.table = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        
        # initialization
        torch.nn.init.trunc_normal_(self.table.data, mean=0, std=1, a=-3, b=-3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.table[token_ids % self.num_embeddings]  # take mod to make sure the result is within range
