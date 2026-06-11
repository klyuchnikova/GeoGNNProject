from dataclasses import dataclass
import torch
import torch.nn as nn
from .base import BaseModel

@dataclass
class GuGenConfig:
    num_pois: int
    num_users: int
    num_categories: int
    hidden_dim: int = 128
    graph_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2

class HistoryEncoder(nn.Module):
    def __init__(self, num_pois, num_users, num_categories, hidden_dim):
        super().__init__()
        self.poi_emb = nn.Embedding(num_pois, hidden_dim)
        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.cat_emb = nn.Embedding(num_categories, hidden_dim)
        self.gru = nn.GRU(hidden_dim * 3, hidden_dim, batch_first=True)

    def forward(self, poi, user, category):
        p = self.poi_emb(poi)
        u = self.user_emb(user).unsqueeze(1).expand(-1, p.size(1), -1)
        c = self.cat_emb(category)
        x = torch.cat([p, u, c], dim=-1)
        h, _ = self.gru(x)
        return h

class SequenceEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    
    def forward(self, x):
        return self.encoder(x)

class GuGen(BaseModel):
    def __init__(self, cfg: GuGenConfig):
        super().__init__()
        self.cfg = cfg
        self.history = HistoryEncoder(cfg.num_pois, cfg.num_users, cfg.num_categories, cfg.hidden_dim)
        self.sequence = SequenceEncoder(cfg.hidden_dim, cfg.num_heads, cfg.num_layers)
        self.head = nn.Linear(cfg.hidden_dim, cfg.num_pois)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, batch):
        h = self.history(batch["poi"], batch["user"], batch["category"])
        h = self.sequence(h)
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