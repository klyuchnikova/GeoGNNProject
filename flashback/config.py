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
    input_format: str = "snap"  # snap | canonical_csv | foursquare
    preselected_city: bool = False
    raw_checkins: str = "data/raw/loc-gowalla_totalCheckins.txt.gz"
    raw_friendships: str | None = "data/raw/loc-gowalla_edges.txt.gz"
    raw_metadata: str | None = "data/raw/gowalla_spots_subset1.csv"
    city: str = "auto"
    candidate_cities: list[str] = field(default_factory=lambda: [
        "new_york", "los_angeles", "chicago", "san_francisco",
        "austin", "dallas", "seattle", "boston",
    ])
    # Iterative user/POI k-core thresholds. Paper-like mode uses 101/1;
    # tuned common-benchmark mode uses smaller user_k and poi_k=10.
    min_checkins: int = 101
    min_poi_visits: int = 1
    max_users: int = 0
    sequence_length: int = 20
    sequence_stride: int = 20
    # block_all predicts every next item inside non-overlapping blocks.
    # window_last gives every target a complete context window and is the
    # recommended high-quality mode.
    sequence_mode: str = "block_all"  # block_all | window_last
    split_mode: str = "robust"  # robust or paper
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    output_dir: str = "data/processed"


@dataclass
class STKGConfig:
    spatial_mode: str = "radius"  # rank | radius
    spatial_topk: int = 50
    spatial_symmetric: bool = True
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
    # natural reproduces the raw STKG distribution. relation_balanced stops
    # the spatial relation from occupying most optimization batches.
    sampling: str = "natural"  # natural | relation_balanced
    relation_weights: dict[str, float] = field(default_factory=lambda: {
        "visits": 0.35, "temporal": 0.35, "spatial": 0.25, "friend": 0.05,
    })
    checkpoint: str = "data/kge/transe_best.pt"
    device: str = "auto"


@dataclass
class GraphConfig:
    transition_topk: int = 100
    user_poi_topk: int = 100
    friend_topk: int = 20
    score_chunk_size: int = 1024
    minimum_score: float = 0.0
    row_relative_scores: bool = False
    output_dir: str = "data/graphs"


@dataclass
class ModelConfig:
    rnn: str = "rnn"
    hidden_dim: int = 10
    lambda_t: float = 0.1
    lambda_s: float = 1000.0
    learnable_decay: bool = False
    lambda_loc: float = 1.0
    lambda_user: float = 1.0
    use_transition_graph: bool = True
    use_preference_graph: bool = True
    use_spatial_graph: bool = False
    use_friend_graph: bool = False
    graph_weight_projection: bool = False
    graph_layers: int = 1
    graph_include_initial: bool = False
    coordinate_distance: str = "euclidean_degrees"
    use_category_embedding: bool = False
    use_time_embedding: bool = False
    # Optional train-only and context-only priors used by the tuned model.
    # They are disabled in the faithful configuration.
    personal_prior_weight: float = 0.0
    recent_prior_weight: float = 0.0
    global_prior_weight: float = 0.0
    geo_prior_weight: float = 0.0
    category_transition_weight: float = 0.0
    personal_prior_topk: int = 200
    recent_prior_tau: float = 5.0
    geo_prior_scale_km: float = 5.0
    learnable_prior_weights: bool = False
    use_repeat_gate: bool = False
    use_user_context: bool = False
    # If >0, bound only the neural residual; priors remain unbounded.
    bounded_residual_scale: float = 0.0
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 200
    epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    gradient_clip: float = 5.0
    label_smoothing: float = 0.0
    bpr_weight: float = 0.0
    repeat_gate_weight: float = 0.0
    hard_negatives: int = 8
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
    if cfg.data.input_format not in {"snap", "canonical_csv", "foursquare"}:
        raise ValueError("data.input_format must be snap, canonical_csv or foursquare")
    if cfg.data.sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    if cfg.data.sequence_stride < 1:
        raise ValueError("sequence_stride must be >= 1")
    if cfg.data.sequence_mode not in {"block_all", "window_last"}:
        raise ValueError("sequence_mode must be block_all or window_last")
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
    if cfg.kge.sampling not in {"natural", "relation_balanced"}:
        raise ValueError("kge.sampling must be natural or relation_balanced")
    if cfg.train.checkpoint_mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be min or max")
    if cfg.model.graph_layers < 0:
        raise ValueError("graph_layers must be >= 0")
    if cfg.train.hard_negatives < 1:
        raise ValueError("hard_negatives must be >= 1")
    if cfg.model.recent_prior_tau <= 0:
        raise ValueError("recent_prior_tau must be > 0")
    if cfg.model.geo_prior_scale_km <= 0:
        raise ValueError("geo_prior_scale_km must be > 0")
    for name in (
        "personal_prior_weight", "recent_prior_weight", "global_prior_weight",
        "geo_prior_weight", "category_transition_weight",
    ):
        if getattr(cfg.model, name) < 0:
            raise ValueError(f"{name} must be >= 0")
    if cfg.data.val_ratio == 0 and cfg.train.checkpoint_metric not in {"train_loss", "final_epoch"}:
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
    torch.use_deterministic_algorithms(False)


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
