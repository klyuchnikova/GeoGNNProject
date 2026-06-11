import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row, col = edge_index
    if edge_weight is None:
        edge_weight = torch.ones(row.size(0), device=row.device, dtype=torch.float32)
    else:
        edge_weight = edge_weight.float()

    deg = torch.zeros(num_nodes, device=row.device, dtype=torch.float32)
    deg.index_add_(0, col, edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return row, col, norm_weight


def graph_propagate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros_like(x)
    row, col, norm_weight = normalize_adjacency(edge_index, num_nodes, edge_weight)
    out = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
    messages = x[row] * norm_weight.unsqueeze(-1)
    out.index_add_(0, col, messages)
    return out


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        neigh = graph_propagate(x, edge_index, num_nodes, edge_weight)
        out = self.self_lin(x) + self.neigh_lin(neigh)
        if out.size(-1) == x.size(-1):
            out = out + x
        return self.norm(F.leaky_relu(out, negative_slope=0.2))


class GCNStack(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [GCNLayer(in_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        return h
