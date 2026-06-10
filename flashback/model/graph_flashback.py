from __future__ import annotations

import math
from pathlib import Path

import scipy.sparse as sp
import torch
from torch import nn

from flashback.config import ExperimentConfig
from flashback.utils import (
    load_sparse,
    random_walk_normalize,
    read_table,
    scipy_to_torch_sparse,
)


def _normalized_with_self(matrix: sp.spmatrix, self_weight: float, edge_weight: float) -> sp.csr_matrix:
    matrix = matrix.astype("float32") * edge_weight
    matrix = matrix + sp.eye(matrix.shape[0], dtype="float32") * self_weight
    return random_walk_normalize(matrix)


def _haversine_pairwise(coords: torch.Tensor) -> torch.Tensor:
    """Pairwise haversine distance in kilometres for [B, S, 2] degree coordinates."""
    radians = torch.deg2rad(coords)
    latitude = radians[..., 0]
    longitude = radians[..., 1]
    d_latitude = latitude[:, :, None] - latitude[:, None, :]
    d_longitude = longitude[:, :, None] - longitude[:, None, :]
    a = (
        torch.sin(d_latitude / 2) ** 2
        + torch.cos(latitude[:, :, None])
        * torch.cos(latitude[:, None, :])
        * torch.sin(d_longitude / 2) ** 2
    )
    return 6371.0088 * 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))


class GraphFlashback(nn.Module):
    """STKG/TransE Graph-Flashback with vectorised flashback aggregation.

    With ``graph_weight_projection=false`` this uses the LightGCN-style graph
    propagation from the paper: normalized adjacency multiplied by embeddings,
    without an extra trainable matrix.
    """

    def __init__(
        self,
        n_users: int,
        n_pois: int,
        cfg: ExperimentConfig,
        transition: sp.spmatrix,
        preference: sp.spmatrix,
        spatial: sp.spmatrix | None = None,
        friends: sp.spmatrix | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg.model.hidden_dim
        self.n_pois = n_pois

        self.poi_embedding = nn.Embedding(n_pois, hidden_dim)
        self.user_embedding = nn.Embedding(n_users, hidden_dim)
        projection = cfg.model.graph_weight_projection
        self.poi_graph_projection = nn.Linear(hidden_dim, hidden_dim, bias=False) if projection else nn.Identity()
        self.spatial_graph_projection = nn.Linear(hidden_dim, hidden_dim, bias=False) if projection else nn.Identity()
        self.user_graph_projection = nn.Linear(hidden_dim, hidden_dim, bias=False) if projection else nn.Identity()

        rnn_class = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cfg.model.rnn]
        self.rnn = rnn_class(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.output = nn.Linear(2 * hidden_dim, n_pois)

        self.register_buffer(
            "transition",
            scipy_to_torch_sparse(_normalized_with_self(transition, 1.0, cfg.model.lambda_loc)),
        )
        self.register_buffer(
            "preference",
            scipy_to_torch_sparse(random_walk_normalize(preference)),
        )

        self.spatial_enabled = spatial is not None and cfg.model.use_spatial_graph
        self.friend_enabled = friends is not None and cfg.model.use_friend_graph
        if self.spatial_enabled:
            self.register_buffer(
                "spatial",
                scipy_to_torch_sparse(_normalized_with_self(spatial, 1.0, cfg.model.lambda_loc)),
            )
        if self.friend_enabled:
            self.register_buffer(
                "friends",
                scipy_to_torch_sparse(_normalized_with_self(friends, 1.0, cfg.model.lambda_user)),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def graph_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        poi = self.poi_graph_projection(torch.sparse.mm(self.transition, self.poi_embedding.weight))
        if self.spatial_enabled:
            spatial = self.spatial_graph_projection(torch.sparse.mm(self.spatial, self.poi_embedding.weight))
            poi = 0.5 * (poi + spatial)

        user = self.user_embedding.weight
        if self.friend_enabled:
            friend = self.user_graph_projection(torch.sparse.mm(self.friends, user))
            user = 0.5 * (user + friend)
        return poi, user

    def forward(
        self,
        locations: torch.Tensor,
        timestamps: torch.Tensor,
        coordinates: torch.Tensor,
        user_id: torch.Tensor,
        valid_input: torch.Tensor | None = None,
        hidden=None,
    ):
        poi_embeddings, user_embeddings = self.graph_embeddings()
        inputs = poi_embeddings[locations]
        timestamps = timestamps.to(dtype=inputs.dtype)
        coordinates = coordinates.to(dtype=inputs.dtype)
        recurrent, hidden = self.rnn(self.dropout(inputs), hidden)

        _, sequence_length, _ = recurrent.shape
        time_delta = torch.clamp(timestamps[:, :, None] - timestamps[:, None, :], min=0.0)
        day_delta = time_delta / 86400.0
        temporal_weight = (
            0.5 * (torch.cos(2 * math.pi * day_delta) + 1.0)
            * torch.exp(-self.cfg.model.lambda_t * day_delta)
        )

        if self.cfg.model.coordinate_distance == "haversine_km":
            distance = _haversine_pairwise(coordinates)
        else:
            # This is the coordinate-space distance used by the upstream code.
            distance = torch.cdist(coordinates, coordinates, p=2)
        spatial_weight = torch.exp(-self.cfg.model.lambda_s * distance)

        user_preference = torch.sparse.mm(self.preference, poi_embeddings)
        preference_similarity = torch.exp(
            -torch.linalg.vector_norm(
                inputs - user_preference[user_id][:, None, :], ord=2, dim=-1
            )
        )

        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=inputs.device, dtype=torch.bool)
        )[None, :, :]
        weights = temporal_weight * spatial_weight * preference_similarity[:, None, :]
        if valid_input is not None:
            weights = weights * valid_input[:, None, :]
        weights = weights * causal_mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)

        flashback = torch.bmm(weights, recurrent)
        user = user_embeddings[user_id][:, None, :].expand(-1, sequence_length, -1)
        logits = self.output(self.dropout(torch.cat([flashback, user], dim=-1)))
        return logits, hidden, weights


def load_model(
    cfg: ExperimentConfig,
    checkins_path: str | Path,
    graph_dir: str | Path | None = None,
) -> GraphFlashback:
    graph_dir = Path(graph_dir or cfg.graphs.output_dir)
    frame = read_table(checkins_path)
    spatial_path = graph_dir / "poi_spatial.npz"
    friend_path = graph_dir / "user_friend.npz"
    return GraphFlashback(
        int(frame.user_id.max()) + 1,
        int(frame.poi_id.max()) + 1,
        cfg,
        load_sparse(graph_dir / "poi_transition.npz"),
        load_sparse(graph_dir / "user_poi_preference.npz"),
        load_sparse(spatial_path) if spatial_path.exists() else None,
        load_sparse(friend_path) if friend_path.exists() else None,
    )
