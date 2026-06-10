from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import json
import logging
import shutil
import urllib.request

import numpy as np
import scipy.sparse as sp
import torch


def setup_logging(log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("flashback")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def download(url: str, destination: str | Path, expected_sha256: str | None = None) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        if expected_sha256 is None or sha256(destination) == expected_sha256:
            return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "GeoGNNProject/0.2"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)
    if expected_sha256 and sha256(destination) != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {destination}")
    return destination


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_maybe_gzip(path: str | Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=None if "b" in mode else "utf-8")
    return path.open(mode, encoding=None if "b" in mode else "utf-8")


def random_walk_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inverse[nonzero] = 1.0 / row_sum[nonzero]
    return sp.diags(inverse).dot(matrix).tocsr()


def scipy_to_torch_sparse(matrix: sp.spmatrix, device: torch.device | None = None) -> torch.Tensor:
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    tensor = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
    return tensor.to(device) if device is not None else tensor


def save_sparse(matrix: sp.spmatrix, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matrix.tocsr())


def load_sparse(path: str | Path) -> sp.csr_matrix:
    return sp.load_npz(path).tocsr()


def write_table(frame, path: str | Path, index: bool = False) -> Path:
    """Write Parquet when available, otherwise a pandas pickle at the same path.

    The matching read_table function auto-detects the fallback. Production/Kaggle
    requirements include pyarrow; the fallback keeps CPU smoke tests self-contained.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=index)
    except (ImportError, ModuleNotFoundError):
        frame.to_pickle(path)
    return path


def read_table(path: str | Path):
    import pandas as pd
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except (ImportError, ModuleNotFoundError, ValueError, OSError):
        return pd.read_pickle(path)
