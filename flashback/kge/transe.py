from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int, p_norm: int = 1):
        super().__init__()
        self.entity = nn.Embedding(n_entities, dim)
        self.relation = nn.Embedding(n_relations, dim)
        self.p_norm = p_norm
        bound = 6.0 / max(1, dim) ** 0.5
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)
        self.normalize_entities()
    def normalize_entities(self):
        with torch.no_grad(): self.entity.weight.copy_(F.normalize(self.entity.weight, p=2, dim=1))
    def distance(self, h, r, t):
        return torch.linalg.vector_norm(self.entity(h) + self.relation(r) - self.entity(t), ord=self.p_norm, dim=-1)
    def forward(self, triplets):
        return self.distance(triplets[:,0], triplets[:,1], triplets[:,2])
