from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from flashback.config import ExperimentConfig
from flashback.utils import load_sparse, random_walk_normalize, read_table, scipy_to_torch_sparse


def _normalized_with_self(matrix: sp.spmatrix, self_weight: float, edge_weight: float) -> sp.csr_matrix:
    matrix = matrix.astype("float32") * edge_weight
    matrix = matrix + sp.eye(matrix.shape[0], dtype="float32") * self_weight
    return random_walk_normalize(matrix)


def _haversine_pairwise(coords: torch.Tensor) -> torch.Tensor:
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


def _haversine_candidates(origin: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Great-circle distance from B origins to P candidate POIs, shape B x P."""
    origin_radians = torch.deg2rad(origin)
    candidate_radians = torch.deg2rad(candidates)
    lat1 = origin_radians[:, 0:1]
    lon1 = origin_radians[:, 1:2]
    lat2 = candidate_radians[:, 0][None, :]
    lon2 = candidate_radians[:, 1][None, :]
    d_latitude = lat2 - lat1
    d_longitude = lon2 - lon1
    a = (
        torch.sin(d_latitude / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(d_longitude / 2) ** 2
    )
    return 6371.0088 * 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def _build_context_statistics(
    frame: pd.DataFrame,
    n_users: int,
    n_pois: int,
    n_categories: int,
    personal_topk: int,
):
    """Create train-only priors plus static candidate metadata.

    Check-in targets from validation/test are never counted. POI coordinates and
    categories are static metadata and may safely be read from the prepared table.
    """
    train = frame[frame["split"] == "train"].sort_values(["user_id", "timestamp"])

    global_counts = np.bincount(
        train["poi_id"].to_numpy(np.int64), minlength=n_pois
    ).astype(np.float32)
    global_values = np.log1p(global_counts)
    if global_values.max() > 0:
        global_values /= global_values.max()

    keep = max(1, min(int(personal_topk), n_pois))
    personal_indices = np.zeros((n_users, keep), dtype=np.int64)
    personal_values = np.zeros((n_users, keep), dtype=np.float32)
    for user_id, group in train.groupby("user_id", sort=False):
        counts = group["poi_id"].value_counts()
        ids = counts.index.to_numpy(np.int64)[:keep]
        scores = np.log1p(counts.to_numpy(np.float32)[:keep])
        if scores.size and scores.max() > 0:
            scores /= scores.max()
        personal_indices[int(user_id), : len(ids)] = ids
        personal_values[int(user_id), : len(ids)] = scores

    poi_coordinates = (
        frame.groupby("poi_id")[["latitude", "longitude"]]
        .median()
        .reindex(range(n_pois))
        .fillna(0.0)
        .to_numpy(np.float32)
    )
    poi_categories = (
        frame.groupby("poi_id")["category_id"]
        .agg(lambda values: int(values.mode().iloc[0]) if not values.mode().empty else 0)
        .reindex(range(n_pois))
        .fillna(0)
        .to_numpy(np.int64)
    )

    # Smoothed transition probabilities between POI categories, derived only
    # from consecutive train events of the same user.
    transitions = np.full((max(1, n_categories), max(1, n_categories)), 1e-3, dtype=np.float32)
    for _, group in train.groupby("user_id", sort=False):
        categories = group["category_id"].to_numpy(np.int64)
        if len(categories) > 1:
            np.add.at(transitions, (categories[:-1], categories[1:]), 1.0)
    transitions /= np.maximum(transitions.sum(axis=1, keepdims=True), 1e-12)
    transitions /= np.maximum(transitions.max(axis=1, keepdims=True), 1e-12)

    return (
        torch.from_numpy(global_values),
        torch.from_numpy(personal_indices),
        torch.from_numpy(personal_values),
        torch.from_numpy(poi_coordinates),
        torch.from_numpy(poi_categories),
        torch.from_numpy(transitions),
    )


def _row_max_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.astype("float32").tocsr(copy=True)
    if matrix.nnz == 0:
        return matrix
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        if start == end:
            continue
        maximum = float(matrix.data[start:end].max())
        if maximum > 0:
            matrix.data[start:end] /= maximum
    return matrix


def _empirical_graphs(frame: pd.DataFrame, n_users: int, n_pois: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    train = frame[frame["split"] == "train"].sort_values(["user_id", "timestamp"])
    transition = sp.lil_matrix((n_pois, n_pois), dtype=np.float32)
    preference = sp.lil_matrix((n_users, n_pois), dtype=np.float32)
    for user_id, group in train.groupby("user_id", sort=False):
        pois = group["poi_id"].to_numpy(np.int64)
        for poi_id in pois:
            preference[int(user_id), int(poi_id)] += 1.0
        for left, right in zip(pois[:-1], pois[1:]):
            if left != right:
                transition[int(left), int(right)] += 1.0
    return _row_max_normalize(transition), _row_max_normalize(preference)


class GraphFlashback(nn.Module):
    """Graph-Flashback with optional ranking-oriented enhancements.

    The faithful path follows the original structure: graph-propagated POI
    embeddings -> recurrent encoder -> temporal/spatial/preference flashback ->
    user-conditioned POI classifier.

    The tuned path keeps that core and optionally adds features that are useful
    for sparse next-POI data: category/time inputs, LightGCN-style multi-hop
    propagation, train-only personal/global priors, a recent-history prior,
    candidate geography/category priors and a learned repeat/explore gate.
    Every enhancement is configurable and disabled in the faithful preset.
    """

    def __init__(
        self,
        n_users: int,
        n_pois: int,
        n_categories: int,
        cfg: ExperimentConfig,
        transition: sp.spmatrix,
        preference: sp.spmatrix,
        global_prior: torch.Tensor | None = None,
        personal_prior_index: torch.Tensor | None = None,
        personal_prior_value: torch.Tensor | None = None,
        poi_coordinates: torch.Tensor | None = None,
        poi_categories: torch.Tensor | None = None,
        category_transition: torch.Tensor | None = None,
        spatial: sp.spmatrix | None = None,
        friends: sp.spmatrix | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg.model.hidden_dim
        self.n_pois = n_pois
        self.n_users = n_users

        self.poi_embedding = nn.Embedding(n_pois, hidden_dim)
        self.user_embedding = nn.Embedding(n_users, hidden_dim)
        self.category_embedding = (
            nn.Embedding(max(1, n_categories), hidden_dim, padding_idx=0)
            if cfg.model.use_category_embedding
            else None
        )
        self.hour_embedding = nn.Embedding(168, hidden_dim) if cfg.model.use_time_embedding else None
        self.gap_embedding = nn.Embedding(32, hidden_dim) if cfg.model.use_time_embedding else None

        projection = cfg.model.graph_weight_projection
        self.poi_graph_projection = nn.Linear(hidden_dim, hidden_dim, bias=False) if projection else nn.Identity()
        self.spatial_graph_projection = nn.Linear(hidden_dim, hidden_dim, bias=False) if projection else nn.Identity()
        self.user_graph_projection = nn.Linear(hidden_dim, hidden_dim, bias=False) if projection else nn.Identity()

        rnn_class = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cfg.model.rnn]
        self.rnn = rnn_class(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.output = nn.Linear(2 * hidden_dim, n_pois)
        self.repeat_gate = nn.Linear(2 * hidden_dim, 1) if cfg.model.use_repeat_gate else None

        self.register_buffer(
            "transition",
            scipy_to_torch_sparse(_normalized_with_self(transition, 1.0, cfg.model.lambda_loc)),
        )
        self.register_buffer("preference", scipy_to_torch_sparse(random_walk_normalize(preference)))
        if cfg.model.transition_graph_prior_weight > 0:
            self.register_buffer("transition_prior", torch.from_numpy(_row_max_normalize(transition).toarray()))
        else:
            self.register_buffer("transition_prior", torch.empty(0))

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

        prior_k = max(1, cfg.model.personal_prior_topk)
        self.register_buffer("global_prior", global_prior if global_prior is not None else torch.zeros(n_pois))
        self.register_buffer(
            "personal_prior_index",
            personal_prior_index
            if personal_prior_index is not None
            else torch.zeros((n_users, prior_k), dtype=torch.long),
        )
        self.register_buffer(
            "personal_prior_value",
            personal_prior_value
            if personal_prior_value is not None
            else torch.zeros((n_users, prior_k)),
        )
        self.register_buffer(
            "poi_coordinates",
            poi_coordinates if poi_coordinates is not None else torch.zeros((n_pois, 2)),
        )
        self.register_buffer(
            "poi_categories",
            poi_categories if poi_categories is not None else torch.zeros(n_pois, dtype=torch.long),
        )
        self.register_buffer(
            "category_transition",
            category_transition
            if category_transition is not None
            else torch.ones((max(1, n_categories), max(1, n_categories))),
        )

        if cfg.model.learnable_decay:
            self.raw_lambda_t = nn.Parameter(torch.tensor(_inverse_softplus(cfg.model.lambda_t)))
            self.raw_lambda_s = nn.Parameter(torch.tensor(_inverse_softplus(cfg.model.lambda_s)))
        else:
            self.register_buffer("fixed_lambda_t", torch.tensor(float(cfg.model.lambda_t)))
            self.register_buffer("fixed_lambda_s", torch.tensor(float(cfg.model.lambda_s)))

        self._register_prior_weight("personal", cfg.model.personal_prior_weight)
        self._register_prior_weight("recent", cfg.model.recent_prior_weight)
        self._register_prior_weight("global", cfg.model.global_prior_weight)
        self._register_prior_weight("geo", cfg.model.geo_prior_weight)
        self._register_prior_weight("category", cfg.model.category_transition_weight)
        self._register_prior_weight("dynamic_graph", cfg.model.dynamic_graph_prior_weight)
        self._register_prior_weight("transition_graph", cfg.model.transition_graph_prior_weight)
        self.reset_parameters()

    def _register_prior_weight(self, name: str, value: float) -> None:
        raw_name = f"raw_{name}_prior_weight"
        fixed_name = f"fixed_{name}_prior_weight"
        if self.cfg.model.learnable_prior_weights and value > 0:
            self.register_parameter(raw_name, nn.Parameter(torch.tensor(_inverse_softplus(value))))
        else:
            self.register_buffer(fixed_name, torch.tensor(float(value)))

    def prior_weight(self, name: str) -> torch.Tensor:
        raw = getattr(self, f"raw_{name}_prior_weight", None)
        if raw is not None:
            return F.softplus(raw)
        return getattr(self, f"fixed_{name}_prior_weight")

    @property
    def lambda_t(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda_t) if self.cfg.model.learnable_decay else self.fixed_lambda_t

    @property
    def lambda_s(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda_s) if self.cfg.model.learnable_decay else self.fixed_lambda_s

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)
        if self.repeat_gate is not None:
            nn.init.zeros_(self.repeat_gate.weight)
            nn.init.constant_(self.repeat_gate.bias, 0.5)
        if self.category_embedding is not None:
            nn.init.normal_(self.category_embedding.weight, std=0.02)
            with torch.no_grad():
                self.category_embedding.weight[0].zero_()
        if self.hour_embedding is not None:
            nn.init.normal_(self.hour_embedding.weight, std=0.02)
            nn.init.normal_(self.gap_embedding.weight, std=0.02)

    def _propagate(self, adjacency: torch.Tensor, embeddings: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if self.cfg.model.graph_layers <= 0:
            return embeddings
        values = [embeddings] if self.cfg.model.graph_include_initial else []
        current = embeddings
        for _ in range(self.cfg.model.graph_layers):
            current = projection(torch.sparse.mm(adjacency, current))
            values.append(current)
        return torch.stack(values, dim=0).mean(dim=0) if len(values) > 1 else values[0]

    def graph_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        poi = self.poi_embedding.weight
        if self.cfg.model.use_transition_graph:
            poi = self._propagate(self.transition, poi, self.poi_graph_projection)
        if self.spatial_enabled:
            spatial = self._propagate(self.spatial, self.poi_embedding.weight, self.spatial_graph_projection)
            poi = 0.5 * (poi + spatial)

        user = self.user_embedding.weight
        if self.friend_enabled:
            friend = self._propagate(self.friends, user, self.user_graph_projection)
            user = 0.5 * (user + friend)
        return poi, user

    def _context_inputs(
        self,
        poi_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
        locations: torch.Tensor,
        user_id: torch.Tensor,
        category_ids: torch.Tensor | None,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pure_poi_inputs = poi_embeddings[locations]
        inputs = pure_poi_inputs
        if self.category_embedding is not None and category_ids is not None:
            inputs = inputs + self.category_embedding(category_ids)
        if self.hour_embedding is not None:
            hour_of_week = torch.remainder(
                torch.div(timestamps.long(), 3600, rounding_mode="floor"), 168
            )
            gaps = torch.zeros_like(timestamps)
            gaps[:, 1:] = torch.clamp(timestamps[:, 1:] - timestamps[:, :-1], min=0)
            gap_bucket = torch.clamp(torch.log2(gaps / 60.0 + 1.0).long(), 0, 31)
            inputs = inputs + self.hour_embedding(hour_of_week) + self.gap_embedding(gap_bucket)
        if self.cfg.model.use_user_context:
            inputs = inputs + user_embeddings[user_id][:, None, :]
        return inputs, pure_poi_inputs

    def _dense_personal_prior(self, user_id: torch.Tensor, output_steps: int) -> torch.Tensor:
        dense = torch.zeros(
            (len(user_id), output_steps, self.n_pois),
            device=user_id.device,
            dtype=self.personal_prior_value.dtype,
        )
        index = self.personal_prior_index[user_id][:, None, :].expand(-1, output_steps, -1)
        value = self.personal_prior_value[user_id][:, None, :].expand(-1, output_steps, -1)
        return dense.scatter_add(2, index, value)

    def _dense_recent_prior(self, locations: torch.Tensor, output_steps: int) -> torch.Tensor:
        batch_size, sequence_length = locations.shape
        age = torch.arange(
            sequence_length - 1, -1, -1, device=locations.device, dtype=self.poi_embedding.weight.dtype
        )
        value = torch.exp(-age / float(self.cfg.model.recent_prior_tau))
        value = value[None, None, :].expand(batch_size, output_steps, -1)
        index = locations[:, None, :].expand(-1, output_steps, -1)
        dense = torch.zeros(
            (batch_size, output_steps, self.n_pois),
            device=locations.device,
            dtype=value.dtype,
        ).scatter_add(2, index, value)
        return dense / (dense.amax(dim=-1, keepdim=True) + 1e-12)

    def _dense_dynamic_graph_prior(
        self,
        index: torch.Tensor,
        value: torch.Tensor,
        output_steps: int,
    ) -> torch.Tensor:
        if self.cfg.data.sequence_mode == "window_last" and index.shape[1] != output_steps:
            index = index[:, -1:, :]
            value = value[:, -1:, :]
        dense = torch.zeros(
            (index.shape[0], output_steps, self.n_pois),
            device=index.device,
            dtype=value.dtype,
        )
        return dense.scatter_add(2, index[:, :output_steps, :], value[:, :output_steps, :])

    def _direct_transition_graph_prior(self, locations: torch.Tensor, output_steps: int) -> torch.Tensor:
        if self.transition_prior.numel() == 0:
            return torch.zeros(
                (locations.shape[0], output_steps, self.n_pois),
                device=locations.device,
                dtype=self.poi_embedding.weight.dtype,
            )
        source = locations[:, -1:] if self.cfg.data.sequence_mode == "window_last" else locations
        return self.transition_prior[source[:, :output_steps]]

    def _candidate_geo_prior(self, coordinates: torch.Tensor, output_steps: int) -> torch.Tensor:
        if self.cfg.data.sequence_mode == "window_last":
            origin = coordinates[:, -1, :]
            distances = _haversine_candidates(origin, self.poi_coordinates)
            score = torch.exp(-distances / float(self.cfg.model.geo_prior_scale_km))
            return score[:, None, :]
        # In block mode each output step has its own current coordinate.
        batch_size, sequence_length, _ = coordinates.shape
        flat = coordinates.reshape(-1, 2)
        distance = _haversine_candidates(flat, self.poi_coordinates)
        score = torch.exp(-distance / float(self.cfg.model.geo_prior_scale_km))
        return score.reshape(batch_size, sequence_length, self.n_pois)

    def _candidate_category_prior(
        self,
        category_ids: torch.Tensor | None,
        output_steps: int,
    ) -> torch.Tensor | None:
        if category_ids is None:
            return None
        source = category_ids[:, -1:] if self.cfg.data.sequence_mode == "window_last" else category_ids
        category_scores = self.category_transition[source]
        candidate_index = self.poi_categories[None, None, :].expand(
            source.shape[0], source.shape[1], -1
        )
        return torch.gather(category_scores, 2, candidate_index)

    def repeat_labels(
        self,
        user_id: torch.Tensor,
        locations: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        history_seen = (locations[:, None, :] == targets[:, :, None]).any(dim=-1)
        index = self.personal_prior_index[user_id]
        personal_seen = (index[:, None, :] == targets[:, :, None]).any(dim=-1)
        return (history_seen | personal_seen).float()

    def forward(
        self,
        locations: torch.Tensor,
        timestamps: torch.Tensor,
        coordinates: torch.Tensor,
        user_id: torch.Tensor,
        valid_input: torch.Tensor | None = None,
        category_ids: torch.Tensor | None = None,
        history_prior_index: torch.Tensor | None = None,
        history_prior_value: torch.Tensor | None = None,
        hidden=None,
    ):
        poi_embeddings, user_embeddings = self.graph_embeddings()
        timestamps = timestamps.to(dtype=poi_embeddings.dtype)
        coordinates = coordinates.to(dtype=poi_embeddings.dtype)
        inputs, pure_poi_inputs = self._context_inputs(
            poi_embeddings, user_embeddings, locations, user_id, category_ids, timestamps
        )
        recurrent, hidden = self.rnn(self.dropout(inputs), hidden)

        _, sequence_length, _ = recurrent.shape
        time_delta = torch.clamp(timestamps[:, :, None] - timestamps[:, None, :], min=0.0)
        day_delta = time_delta / 86400.0
        temporal_weight = (
            0.5 * (torch.cos(2 * math.pi * day_delta) + 1.0)
            * torch.exp(-self.lambda_t * day_delta)
        )
        distance = (
            _haversine_pairwise(coordinates)
            if self.cfg.model.coordinate_distance == "haversine_km"
            else torch.cdist(coordinates, coordinates, p=2)
        )
        spatial_weight = torch.exp(-self.lambda_s * distance)

        if self.cfg.model.use_preference_graph:
            user_preference = torch.sparse.mm(self.preference, poi_embeddings)
            # Compare the user preference with graph-propagated POI embeddings,
            # not with category/time-enriched RNN inputs from a different space.
            preference_similarity = torch.exp(
                -torch.linalg.vector_norm(
                    pure_poi_inputs - user_preference[user_id][:, None, :], ord=2, dim=-1
                )
            )
        else:
            preference_similarity = torch.ones_like(timestamps)

        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=inputs.device, dtype=torch.bool)
        )[None, :, :]
        weights = temporal_weight * spatial_weight * preference_similarity[:, None, :]
        if valid_input is not None:
            weights = weights * valid_input[:, None, :]
        weights = weights * causal_mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)

        flashback = torch.bmm(weights, recurrent)
        if self.cfg.data.sequence_mode == "window_last":
            flashback = flashback[:, -1:, :]
            output_steps = 1
        else:
            output_steps = sequence_length
        user = user_embeddings[user_id][:, None, :].expand(-1, output_steps, -1)
        representation = torch.cat([flashback, user], dim=-1)
        logits = self.output(self.dropout(representation))
        if self.cfg.model.bounded_residual_scale > 0:
            logits = float(self.cfg.model.bounded_residual_scale) * torch.tanh(logits)

        repeat_probability = (
            torch.sigmoid(self.repeat_gate(representation)) if self.repeat_gate is not None else None
        )
        repeat_multiplier = repeat_probability if repeat_probability is not None else 1.0

        if float(self.prior_weight("global").detach()) > 0:
            logits = logits + self.prior_weight("global") * self.global_prior[None, None, :]
        if float(self.prior_weight("personal").detach()) > 0:
            logits = logits + repeat_multiplier * self.prior_weight("personal") * self._dense_personal_prior(user_id, output_steps)
        if float(self.prior_weight("recent").detach()) > 0:
            logits = logits + repeat_multiplier * self.prior_weight("recent") * self._dense_recent_prior(locations, output_steps)
        if float(self.prior_weight("geo").detach()) > 0:
            logits = logits + self.prior_weight("geo") * self._candidate_geo_prior(coordinates, output_steps)
        if float(self.prior_weight("category").detach()) > 0:
            category_prior = self._candidate_category_prior(category_ids, output_steps)
            if category_prior is not None:
                logits = logits + self.prior_weight("category") * category_prior
        if (
            history_prior_index is not None
            and history_prior_value is not None
            and float(self.prior_weight("dynamic_graph").detach()) > 0
        ):
            logits = logits + self.prior_weight("dynamic_graph") * self._dense_dynamic_graph_prior(
                history_prior_index,
                history_prior_value.to(dtype=logits.dtype),
                output_steps,
            )
        if float(self.prior_weight("transition_graph").detach()) > 0:
            logits = logits + self.prior_weight("transition_graph") * self._direct_transition_graph_prior(
                locations,
                output_steps,
            ).to(dtype=logits.dtype)

        auxiliary = {
            "flashback_weights": weights,
            "repeat_probability": repeat_probability,
        }
        return logits, hidden, auxiliary


def load_model(
    cfg: ExperimentConfig,
    checkins_path: str | Path,
    graph_dir: str | Path | None = None,
) -> GraphFlashback:
    graph_dir = Path(graph_dir or cfg.graphs.output_dir)
    frame = read_table(checkins_path)
    n_users = int(frame.user_id.max()) + 1
    n_pois = int(frame.poi_id.max()) + 1
    n_categories = int(frame.get("category_id", pd.Series([0])).max()) + 1
    statistics = _build_context_statistics(
        frame, n_users, n_pois, n_categories, cfg.model.personal_prior_topk
    )
    transition_path = graph_dir / "poi_transition.npz"
    preference_path = graph_dir / "user_poi_preference.npz"
    if transition_path.exists() and preference_path.exists():
        transition = load_sparse(transition_path)
        preference = load_sparse(preference_path)
    else:
        transition, preference = _empirical_graphs(frame, n_users, n_pois)
    spatial_path = graph_dir / "poi_spatial.npz"
    friend_path = graph_dir / "user_friend.npz"
    return GraphFlashback(
        n_users,
        n_pois,
        n_categories,
        cfg,
        transition,
        preference,
        *statistics,
        load_sparse(spatial_path) if spatial_path.exists() else None,
        load_sparse(friend_path) if friend_path.exists() else None,
    )
