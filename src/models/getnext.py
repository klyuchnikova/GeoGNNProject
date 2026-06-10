from dataclasses import dataclass
import torch
import torch.nn as nn
from .base import BaseModel

def build_rnn(kind: str, hidden_dim: int):
    kind = kind.lower()
    if kind == "rnn":
        return nn.RNN(hidden_dim, hidden_dim)
    if kind == "gru":
        return nn.GRU(hidden_dim, hidden_dim)
    if kind == "lstm":
        return nn.LSTM(hidden_dim, hidden_dim)
    raise ValueError(kind)

class GetNext(BaseModel):
    def __init__(self, cfg: GetNextConfig):
        super().__init__()
        self.cfg = cfg
        self.poi_emb = nn.Embedding(cfg.num_pois, cfg.hidden_dim)
        self.user_emb = nn.Embedding(cfg.num_users, cfg.hidden_dim)
        self.rnn = build_rnn(cfg.rnn_type, cfg.hidden_dim)
        in_dim = cfg.hidden_dim
        if cfg.use_user_embedding:
            in_dim += cfg.hidden_dim
        self.head = nn.Linear(in_dim, cfg.num_pois)
        self.criterion = nn.CrossEntropyLoss()

    def flashback(self, h, timestamps, coords):
        seq_len, batch_size, dim = h.shape
        out = torch.zeros_like(h)
        for i in range(seq_len):
            weights_sum = torch.zeros(batch_size, 1, device=h.device)
            for j in range(i + 1):
                dt = timestamps[i] - timestamps[j]
                ds = torch.norm(coords[i] - coords[j], dim=-1)
                wt = torch.exp(-dt.float())
                ws = torch.exp(-ds.float())
                w = (wt * ws).unsqueeze(-1)
                out[i] += w * h[j]
                weights_sum += w
            out[i] /= weights_sum
        return out

    def forward(self, batch):
        poi = batch["poi"]
        user = batch["user"]
        x = self.poi_emb(poi)
        h, _ = self.rnn(x)
        h = self.flashback(h, batch["timestamp"], batch["coord"])
        if self.cfg.use_user_embedding:
            u = self.user_emb(user)
            u = u.unsqueeze(0).expand(h.size(0), -1, -1)
            h = torch.cat([h, u], dim=-1)
        logits = self.head(h)
        return logits

    def loss(self, batch):
        logits = self(batch)
        target = batch["target"]
        return self.criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

    def predict(self, batch):
        logits = self(batch)
        return logits.argmax(dim=-1)
