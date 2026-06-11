from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from data.graph_builder import PoiGraphs
from .base import BaseModel
from .gcn import GCNStack


@dataclass
class GuGenPosGraphConfig:
    num_pois: int
    num_users: int
    num_categories: int
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    gcn_layers: int = 2
    dropout: float = 0.1
    use_time_embedding: bool = True
    use_positional_encoding: bool = True
    max_seq_len: int = 100


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeEncoder(nn.Module):
    """Encode time information (hour of day, day of week)."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hour_emb = nn.Embedding(24, hidden_dim // 2)
        self.dow_emb = nn.Embedding(7, hidden_dim // 2)  # day of week
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: Unix timestamps (seconds) - shape (batch, seq_len)
        Returns:
            Time embeddings - shape (batch, seq_len, hidden_dim)
        """
        # Convert to hour of day (0-23) and day of week (0-6)
        hours = (timestamps // 3600) % 24  # Assuming timestamps in seconds
        days = (timestamps // 86400) % 7   # Day of week
        
        hour_emb = self.hour_emb(hours.long())
        dow_emb = self.dow_emb(days.long())
        
        # Concatenate and project
        time_emb = torch.cat([hour_emb, dow_emb], dim=-1)
        time_emb = self.proj(time_emb)
        time_emb = self.norm(time_emb)
        
        return time_emb


class PoiGraphEncoder(nn.Module):
    """Graph refinement on top of base POI embeddings (residual, not replacement)."""

    def __init__(self, cfg: GuGenPosGraphConfig):
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


class GuGenPosGraph(BaseModel):
    """GuGen-Lite extended with transition + geo-spatial POI graph encoders.
    Enhanced with time embeddings and positional encoding.
    """

    def __init__(self, cfg: GuGenPosGraphConfig, graphs: PoiGraphs):
        super().__init__()
        self.cfg = cfg
        
        # Graph components
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

        # Graph encoder
        self.graph_encoder = PoiGraphEncoder(cfg)
        
        # Embeddings
        self.user_emb = nn.Embedding(cfg.num_users, cfg.hidden_dim)
        self.cat_emb = nn.Embedding(cfg.num_categories, cfg.hidden_dim)
        
        # NEW: Time encoder
        self.time_encoder = TimeEncoder(cfg.hidden_dim) if cfg.use_time_embedding else None
        
        # GRU: input dim = poi(128) + user(128) + category(128) + time(128 if used)
        gru_input_dim = cfg.hidden_dim * 3
        if cfg.use_time_embedding:
            gru_input_dim += cfg.hidden_dim
        self.gru = nn.GRU(gru_input_dim, cfg.hidden_dim, batch_first=True)
        
        # NEW: Positional encoding for Transformer
        self.pos_encoder = PositionalEncoding(
            cfg.hidden_dim, 
            cfg.max_seq_len, 
            cfg.dropout
        ) if cfg.use_positional_encoding else None
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        
        # Output head
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

    def encode_sequence(self, poi: torch.Tensor, user: torch.Tensor, 
                        category: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        """Encode sequence with optional time information."""
        p = self.poi_embeddings()[poi]
        u = self.user_emb(user).unsqueeze(1).expand(-1, p.size(1), -1)
        c = self.cat_emb(category)
        
        # Concatenate embeddings
        components = [p, u, c]
        
        # Add time embeddings if available
        if self.time_encoder is not None and timestamps is not None:
            t = self.time_encoder(timestamps)
            components.append(t)
        
        x = torch.cat(components, dim=-1)
        h, _ = self.gru(x)
        return self.dropout(h)

    def forward(self, batch):
        # Encode sequence with time information
        h = self.encode_sequence(
            batch["poi"], 
            batch["user"], 
            batch["category"],
            batch.get("timestamp")  # Pass timestamps if available
        )
        
        # Apply positional encoding to Transformer input
        if self.pos_encoder is not None:
            h = self.pos_encoder(h)
        
        # Transformer encoding
        h = self.transformer(h)
        
        # Predict next POI (use last position)
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