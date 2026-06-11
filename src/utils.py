import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.filters import FilterConfig
from evaluate import Evaluator, print_metrics


@dataclass
class TrainConfig:
    model: str = "gugen"
    dataset: str = "foursquare"
    city: str = "NYC"
    data_root: str = "../input"
    seq_len: int = 20
    stride: int = 1
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1  # Changed default from 0.0 to 0.1
    gcn_layers: int = 2
    geo_dist_km: float = 0.5
    max_geo_neighbors: int = 10
    min_user_visits: int = 10
    min_poi_visits: int = 10
    use_kcore: bool = True
    user_k: int = 10
    poi_k: int = 10
    min_entropy: float | None = None
    max_dominant_ratio: float | None = None
    device: str = "cpu"
    seed: int = 42
    num_workers: int = 0
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    log_dir: str = "logs"
    lr_scheduler_factor: float = 0.1  # Changed from 0.01 to 0.1
    lr_scheduler_patience: int = 5
    exp_name: Optional[str] = None
    # NEW: Parameters for gugen_pos_graph
    use_time_embedding: bool = False
    use_positional_encoding: bool = False


def default_device() -> str:
    return str(resolve_device("cuda"))


def resolve_device(requested: str | None = None) -> torch.device:
    if requested in (None, "", "auto"):
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        logging.getLogger("geognn").warning(
            "Device '%s' requested but CUDA is not available. Using CPU.",
            requested,
        )
        return torch.device("cpu")

    try:
        # Critical fix for V100 + CUDA 12.1 + cuDNN 9.x
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        x = torch.randn(2, 3, device="cuda")
        gru = torch.nn.GRU(3, 4, batch_first=True).cuda()
        gru(x)
        torch.cuda.synchronize()

        # Re-enable for convolutions if needed
        # torch.backends.cudnn.enabled = True
        return torch.device(requested)
    except Exception as exc:
        logging.getLogger("geognn").warning(
            "Device '%s' is not usable (%s). Falling back to CPU.",
            requested,
            exc,
        )
        return torch.device("cpu")


def to_json_safe(obj):
    """Convert numpy/pandas scalars and mapping keys into JSON-serializable values."""
    if isinstance(obj, dict):
        return {to_json_safe(key): to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(value) for value in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("geognn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map raw IDs to contiguous 1-based integers (0 reserved for padding)."""
    df = df.copy()
    user_map = {
        int(uid): idx + 1 for idx, uid in enumerate(sorted(df["user_id"].unique()))
    }
    poi_map = {str(pid): idx + 1 for idx, pid in enumerate(sorted(df["poi_id"].unique()))}

    if not pd.api.types.is_integer_dtype(df["category"]):
        cat_map = {
            str(cid): idx + 1 for idx, cid in enumerate(sorted(df["category"].unique()))
        }
        df["category"] = df["category"].map(cat_map).astype(int)
    else:
        cat_map = {}

    df["user_id"] = df["user_id"].map(user_map)
    df["poi_id"] = df["poi_id"].map(poi_map)
    mappings = {"user": user_map, "poi": poi_map, "category": cat_map}
    return df, mappings


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in batch to device."""
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        elif isinstance(value, dict):
            result[key] = move_batch_to_device(value, device)
        elif isinstance(value, list):
            result[key] = [v.to(device) if torch.is_tensor(v) else v for v in value]
        else:
            result[key] = value
    return result


def train_epoch(model, dataloader, optimizer, device, epoch: int, logger: logging.Logger) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(dataloader, desc=f"Train epoch {epoch}", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
        total_batches += 1
        progress.set_postfix(loss=f"{loss_value:.4f}")

    avg_loss = total_loss / max(total_batches, 1)
    logger.info("Epoch %d train loss: %.4f", epoch, avg_loss)
    return avg_loss


def validate_epoch(model, dataloader, device, epoch: int, logger: logging.Logger, split: str) -> dict:
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(dataloader)
    logger.info("Epoch %d %s metrics: %s", epoch, split, metrics)
    print(f"\n{split.upper()} @ epoch {epoch}")
    print_metrics(metrics)
    return metrics


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: dict,
    cfg: TrainConfig,
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(cfg),
        },
        checkpoint_path,
    )


def save_results(
    cfg: TrainConfig,
    filter_cfg: FilterConfig,
    data_stats: dict,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    best_epoch: int,
    results_path: Path,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "model": cfg.model,
        "dataset": cfg.dataset,
        "city": cfg.city,
        "config": asdict(cfg),
        "filters": asdict(filter_cfg),
        "data_stats": data_stats,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "final_train_metrics": train_metrics,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2)


def run_name(cfg: TrainConfig) -> str:
    """Generate a unique name for this experiment run."""
    name = f"{cfg.model}_{cfg.dataset}_{cfg.city}"
    
    # Add graph indicator if applicable
    if cfg.model == "gugen_graph":
        name += f"_graph"
    elif cfg.model == "gugen_pos_graph":
        name += f"_pos_graph"
        if cfg.use_time_embedding:
            name += "_time"
        if cfg.use_positional_encoding:
            name += "_posenc"
    
    # Add experiment name if provided
    if cfg.exp_name:
        name += f"_{cfg.exp_name}"
    
    return name


def output_paths(cfg: TrainConfig) -> tuple[Path, Path, Path]:
    name = run_name(cfg)
    checkpoint_dir = Path(cfg.checkpoint_dir) / name
    results_dir = Path(cfg.results_dir) / name
    log_dir = Path(cfg.log_dir) / name
    return checkpoint_dir, results_dir, log_dir