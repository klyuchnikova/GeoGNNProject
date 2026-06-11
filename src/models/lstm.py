from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import BaseModel


@dataclass
class LstmConfig:
    num_pois: int
    num_users: int
    num_categories: int
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


class LstmNextPOI(BaseModel):
    """Simple LSTM baseline for next-POI prediction on the unified pipeline."""

    def __init__(self, cfg: LstmConfig):
        super().__init__()
        self.cfg = cfg
        self.poi_emb = nn.Embedding(cfg.num_pois, cfg.hidden_dim)
        self.user_emb = nn.Embedding(cfg.num_users, cfg.hidden_dim)
        self.cat_emb = nn.Embedding(cfg.num_categories, cfg.hidden_dim)
        input_dim = cfg.hidden_dim * 3
        self.lstm = nn.LSTM(
            input_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden_dim, cfg.num_pois)
        self.criterion = nn.CrossEntropyLoss()

    def encode(self, poi: torch.Tensor, user: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        p = self.poi_emb(poi)
        u = self.user_emb(user).unsqueeze(1).expand(-1, p.size(1), -1)
        c = self.cat_emb(category)
        x = torch.cat([p, u, c], dim=-1)
        h, _ = self.lstm(x)
        return h

    def forward(self, batch):
        return self.head(self.encode(batch["poi"], batch["user"], batch["category"]))

    def loss(self, batch):
        logits = self(batch)[:, -1, :]
        return self.criterion(logits, batch["target"])

    def predict(self, batch):
        logits = self(batch)[:, -1, :]
        return logits.argmax(dim=-1)

    def predict_batch(self, batch):
        logits = self(batch)[:, -1, :]
        return logits, batch["target"]
