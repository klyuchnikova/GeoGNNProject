from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import copy
import json
import random
import os

import numpy as np
import torch
import yaml


@dataclass
class DataConfig:
    dataset: str = "gowalla"
    raw_checkins: str = "data/raw/loc-gowalla_totalCheckins.txt.gz"
    raw_friendships: str | None = "data/raw/loc-gowalla_edges.txt.gz"
    raw_metadata: str | None = "data/raw/gowalla_spots_subset1.csv"
    city: str = "auto"
    candidate_cities: list[str] = field(default_factory=lambda: [
        "new_york", "los_angeles", "chicago", "san_francisco", "austin", "dallas", "seattle", "boston"
    ])
    min_checkins: int = 101
    min_poi_visits: int = 3
    max_users: int = 0
    sequence_length: int = 20
    sequence_stride: int = 20
    split_mode: str = "robust"  # robust: 70/10/20; paper: 80/0/20
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    output_dir: str = "data/processed"


@dataclass
class STKGConfig:
    spatial_radius_km: float = 3.0
    spatial_topk: int = 50
    max_spatial_pairs: int = 2_000_000
    include_friendship: bool = True
    deduplicate_triplets: bool = True
    output_dir: str = "data/kge"


@dataclass
class KGEConfig:
    embedding_dim: int = 64
    margin: float = 1.0
    learning_rate: float = 0.001
    batch_size: int = 4096
    epochs: int = 50
    negative_samples: int = 1
    corrupt_head_probability: float = 0.5
    validation_fraction: float = 0.02
    patience: int = 8
    checkpoint: str = "data/kge/transe_best.pt"
    device: str = "auto"


@dataclass
class GraphConfig:
    transition_topk: int = 100
    user_poi_topk: int = 100
    friend_topk: int = 20
    score_chunk_size: int = 1024
    output_dir: str = "data/graphs"


@dataclass
class ModelConfig:
    rnn: str = "gru"
    hidden_dim: int = 64
    lambda_t: float = 0.1
    lambda_s: float = 1000.0
    lambda_loc: float = 1.0
    lambda_user: float = 1.0
    use_spatial_graph: bool = True
    use_friend_graph: bool = True
    graph_weight_projection: bool = True
    coordinate_distance: str = "euclidean_degrees"
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    patience: int = 8
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    run_name: str = "graph_flashback"
    evaluate_test_after_training: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    stkg: STKGConfig = field(default_factory=STKGConfig)
    kge: KGEConfig = field(default_factory=KGEConfig)
    graphs: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    artifacts_dir: str = "artifacts"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    _merge_dataclass(cfg, values)
    if overrides:
        _merge_dataclass(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.data.dataset not in {"gowalla", "foursquare"}:
        raise ValueError("data.dataset must be 'gowalla' or 'foursquare'")
    if cfg.data.sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    if cfg.data.split_mode == "paper":
        cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.test_ratio = 0.8, 0.0, 0.2
    total = cfg.data.train_ratio + cfg.data.val_ratio + cfg.data.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to one, got {total}")
    if cfg.model.rnn not in {"rnn", "gru", "lstm"}:
        raise ValueError("model.rnn must be rnn/gru/lstm")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
