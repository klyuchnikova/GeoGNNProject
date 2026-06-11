from dataclasses import dataclass

import torch
import torch.nn as nn

from data.graph_builder import PoiGraphs
from .base import BaseModel
from .gcn import GCNStack


@dataclass
class GuGenGraphConfig:
    num_pois: int
    num_users: int
    num_categories: int
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    gcn_layers: int = 2
    dropout: float = 0.1


class PoiGraphEncoder(nn.Module):
    """Graph refinement on top of base POI embeddings (residual, not replacement)."""

    def __init__(self, cfg: GuGenGraphConfig):
        super().__init__()
        self.cfg = cfg
        self.poi_emb = nn.Embedding(cfg.num_pois, cfg.hidden_dim)
        self.cat_emb = nn.Embedding(cfg.num_categories, cfg.hidden_dim)
        self.coord_proj = nn.Linear(2, cfg.hidden_dim)
        self.transition_gcn = GCNStack(cfg.hidden_dim, cfg.hidden_dim, cfg.gcn_layers)
        self.geo_gcn = GCNStack(cfg.hidden_dim, cfg.hidden_dim, cfg.gcn_layers)
        self.fuse = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)

    def _normalized_coords(self, coords: torch.Tensor) -> torch.Tensor:
        mask = coords.abs().sum(dim=1) > 0
        if not mask.any():
            return coords
        mean = coords[mask].mean(dim=0)
        std = coords[mask].std(dim=0).clamp(min=1e-4)
        out = torch.zeros_like(coords)
        out[mask] = (coords[mask] - mean) / std
        return out

    def forward(self, graphs: PoiGraphs) -> torch.Tensor:
        poi_ids = torch.arange(self.cfg.num_pois, device=graphs.poi_coords.device)
        base = self.poi_emb(poi_ids)
        coords = self._normalized_coords(graphs.poi_coords)
        x = base + self.cat_emb(graphs.poi_category) + self.coord_proj(coords)

        trans_delta = self.transition_gcn(
            x, graphs.transition_edge_index, graphs.transition_edge_weight
        ) - x
        geo_delta = self.geo_gcn(
            x, graphs.geo_edge_index, graphs.geo_edge_weight
        ) - x
        graph_delta = self.fuse(torch.cat([trans_delta, geo_delta], dim=-1))
        return self.out_norm(base + graph_delta)


class GuGenGraph(BaseModel):
    """GuGen-Lite extended with transition + geo-spatial POI graph encoders."""

    def __init__(self, cfg: GuGenGraphConfig, graphs: PoiGraphs):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "transition_edge_index",
            graphs.transition_edge_index.clone(),
            persistent=False,
        )
        self.register_buffer(
            "transition_edge_weight",
            graphs.transition_edge_weight.clone(),
            persistent=False,
        )
        self.register_buffer(
            "geo_edge_index",
            graphs.geo_edge_index.clone(),
            persistent=False,
        )
        self.register_buffer(
            "geo_edge_weight",
            graphs.geo_edge_weight.clone(),
            persistent=False,
        )
        self.register_buffer("poi_coords", graphs.poi_coords.clone(), persistent=False)
        self.register_buffer("poi_category", graphs.poi_category.clone(), persistent=False)

        self.graph_encoder = PoiGraphEncoder(cfg)
        self.user_emb = nn.Embedding(cfg.num_users, cfg.hidden_dim)
        self.cat_emb = nn.Embedding(cfg.num_categories, cfg.hidden_dim)
        self.gru = nn.GRU(cfg.hidden_dim * 3, cfg.hidden_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(cfg.hidden_dim, cfg.num_pois)
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(cfg.dropout)
        self._cached_poi_repr: torch.Tensor | None = None

    def _graphs(self) -> PoiGraphs:
        return PoiGraphs(
            transition_edge_index=self.transition_edge_index,
            transition_edge_weight=self.transition_edge_weight,
            geo_edge_index=self.geo_edge_index,
            geo_edge_weight=self.geo_edge_weight,
            poi_coords=self.poi_coords,
            poi_category=self.poi_category,
        )

    def invalidate_poi_cache(self) -> None:
        self._cached_poi_repr = None

    def poi_embeddings(self) -> torch.Tensor:
        if self._cached_poi_repr is None or self.training:
            self._cached_poi_repr = self.graph_encoder(self._graphs())
        return self._cached_poi_repr

    def encode_sequence(self, poi: torch.Tensor, user: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        p = self.poi_embeddings()[poi]
        u = self.user_emb(user).unsqueeze(1).expand(-1, p.size(1), -1)
        c = self.cat_emb(category)
        x = torch.cat([p, u, c], dim=-1)
        h, _ = self.gru(x)
        return self.dropout(h)

    def forward(self, batch):
        h = self.encode_sequence(batch["poi"], batch["user"], batch["category"])
        h = self.transformer(h)
        return self.head(h)

    def loss(self, batch):
        logits = self(batch)[:, -1, :]
        return self.criterion(logits, batch["target"])

    def predict(self, batch):
        logits = self(batch)[:, -1, :]
        return logits.argmax(dim=-1)

    def predict_batch(self, batch):
        logits = self(batch)[:, -1, :]
        return logits, batch["target"]
