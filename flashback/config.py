from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import os
import random

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
        "new_york", "los_angeles", "chicago", "san_francisco",
        "austin", "dallas", "seattle", "boston",
    ])
    min_checkins: int = 101
    # Keep 1 for the paper-compatible pipeline. Filtering POIs with counts from
    # the complete timeline would leak validation/test information.
    min_poi_visits: int = 1
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
    # Rank-based spatial relation: each POI is connected to its k nearest POIs.
    spatial_mode: str = "rank"
    spatial_topk: int = 50
    spatial_symmetric: bool = True
    # Radius is retained only for optional diagnostics/backward compatibility.
    spatial_radius_km: float = 3.0
    max_spatial_pairs: int = 2_000_000
    include_friendship: bool = True
    deduplicate_triplets: bool = True
    output_dir: str = "data/kge"


@dataclass
class KGEConfig:
    embedding_dim: int = 100
    margin: float = 1.0
    p_norm: int = 1
    learning_rate: float = 0.001
    batch_size: int = 4096
    epochs: int = 100
    negative_samples: int = 1
    corrupt_head_probability: float = 0.5
    validation_fraction: float = 0.02
    patience: int = 15
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
    # These defaults follow the original Graph-Flashback Gowalla setup.
    rnn: str = "rnn"
    hidden_dim: int = 10
    lambda_t: float = 0.1
    lambda_s: float = 1000.0
    lambda_loc: float = 1.0
    lambda_user: float = 1.0
    # Spatial/friend relations remain inside STKG/TransE, but their additional
    # GCN branches are disabled in the main reproduction configuration.
    use_spatial_graph: bool = False
    use_friend_graph: bool = False
    graph_weight_projection: bool = False
    coordinate_distance: str = "euclidean_degrees"
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 200
    epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    gradient_clip: float = 5.0
    patience: int = 20
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    run_name: str = "graph_flashback"
    checkpoint_metric: str = "MRR"
    checkpoint_mode: str = "max"
    scheduler_milestones: list[int] = field(default_factory=lambda: [20, 40, 60, 80])
    scheduler_gamma: float = 0.2
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
    if cfg.data.sequence_stride < 1:
        raise ValueError("sequence_stride must be >= 1")
    if cfg.data.split_mode == "paper":
        cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.test_ratio = 0.8, 0.0, 0.2
    total = cfg.data.train_ratio + cfg.data.val_ratio + cfg.data.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to one, got {total}")
    if cfg.model.rnn not in {"rnn", "gru", "lstm"}:
        raise ValueError("model.rnn must be rnn/gru/lstm")
    if cfg.model.coordinate_distance not in {"euclidean_degrees", "haversine_km"}:
        raise ValueError("coordinate_distance must be euclidean_degrees or haversine_km")
    if cfg.stkg.spatial_mode not in {"rank", "radius"}:
        raise ValueError("stkg.spatial_mode must be rank or radius")
    if cfg.kge.p_norm not in {1, 2}:
        raise ValueError("kge.p_norm must be 1 or 2")
    if cfg.train.checkpoint_mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be min or max")
    if cfg.data.val_ratio == 0 and cfg.train.checkpoint_metric not in {"train_loss", "final_epoch"}:
        # A paper-mode run must not choose checkpoints on the test set.
        cfg.train.checkpoint_metric = "final_epoch"
        cfg.train.checkpoint_mode = "max"


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
    # Full deterministic mode can make sparse CUDA operations unavailable.
    torch.use_deterministic_algorithms(False)


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
