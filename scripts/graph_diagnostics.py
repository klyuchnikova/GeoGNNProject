from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from flashback.config import dump_json, load_config
from flashback.utils import load_sparse, read_table


def _hit_at(rank_matrix: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray, ks: list[int]) -> dict[str, float]:
    out = {f"hit@{k}": 0 for k in ks}
    total = 0
    csr = rank_matrix.tocsr()
    for row, col in zip(rows.tolist(), cols.tolist()):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        indices = csr.indices[start:end]
        if len(indices) == 0:
            total += 1
            continue
        values = csr.data[start:end]
        order = np.argsort(-values, kind="stable")
        ranked = indices[order]
        total += 1
        for k in ks:
            if col in set(ranked[: min(k, len(ranked))].tolist()):
                out[f"hit@{k}"] += 1
    return {key: (value / total if total else 0.0) for key, value in out.items()} | {"targets": int(total)}


def diagnose(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    manifests = sorted(Path(cfg.data.output_dir).glob(f"{cfg.data.dataset}_*_manifest.json"))
    if not manifests:
        raise FileNotFoundError("Prepared manifest not found. Run prepare first.")
    manifest = json.loads(manifests[-1].read_text())
    city = manifest["city"]
    checkins = read_table(Path(cfg.data.output_dir) / f"{cfg.data.dataset}_{city}_checkins.parquet")
    graph_dir = Path(cfg.graphs.output_dir)
    transition = load_sparse(graph_dir / "poi_transition.npz")
    preference = load_sparse(graph_dir / "user_poi_preference.npz")
    ks = [1, 5, 10, 50, 100]

    rows_prev, cols_target, user_rows, target_cols = [], [], [], []
    for _, group in checkins.sort_values(["user_id", "timestamp"]).groupby("user_id", sort=False):
        pois = group["poi_id"].to_numpy(np.int64)
        users = group["user_id"].to_numpy(np.int64)
        splits = group["split"].astype(str).to_numpy()
        for pos in range(1, len(group)):
            if splits[pos] in {"validation", "test"}:
                rows_prev.append(pois[pos - 1])
                cols_target.append(pois[pos])
                user_rows.append(users[pos])
                target_cols.append(pois[pos])
    rows_prev = np.asarray(rows_prev, dtype=np.int64)
    cols_target = np.asarray(cols_target, dtype=np.int64)
    user_rows = np.asarray(user_rows, dtype=np.int64)
    target_cols = np.asarray(target_cols, dtype=np.int64)

    result = {
        "config": str(config_path),
        "graph_dir": str(graph_dir),
        "transition_true_next_neighbor": _hit_at(transition, rows_prev, cols_target, ks),
        "preference_true_target_neighbor": _hit_at(preference, user_rows, target_cols, ks),
        "transition_nnz": int(transition.nnz),
        "preference_nnz": int(preference.nnz),
    }
    dump_json(result, graph_dir / "graph_neighbor_diagnostics.json")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    diagnose(args.config)


if __name__ == "__main__":
    main()
