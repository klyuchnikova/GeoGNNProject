from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from flashback.config import ExperimentConfig, dump_json, resolve_device
from flashback.kge.transe import TransE
from flashback.kge.triplets import FRIEND, SPATIAL, TEMPORAL, VISITS
from flashback.utils import save_sparse


def _topk_relation(
    model,
    heads,
    targets,
    relation_id,
    k,
    chunk,
    device,
    exclude_self=False,
    minimum_score=0.0,
    row_relative_scores=False,
):
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    target_index = torch.as_tensor(targets, dtype=torch.long, device=device)
    target_embeddings = model.entity.weight[target_index]
    relation = model.relation.weight[relation_id].to(device)
    with torch.no_grad():
        for start in range(0, len(heads), chunk):
            head_ids = heads[start : start + chunk]
            head_index = torch.as_tensor(head_ids, dtype=torch.long, device=device)
            queries = model.entity.weight[head_index] + relation
            distances = torch.cdist(queries, target_embeddings, p=model.p_norm)
            if exclude_self:
                local = torch.arange(len(head_ids), device=device)
                target_positions = head_index - int(targets[0])
                valid = (target_positions >= 0) & (target_positions < len(targets))
                distances[local[valid], target_positions[valid]] = float("inf")
            keep = min(k, distances.shape[1])
            distance, index = torch.topk(distances, keep, largest=False, dim=1)
            if row_relative_scores:
                distance = distance - distance[:, :1]
            score = torch.exp(-distance)
            for local_row in range(len(head_ids)):
                local_score = score[local_row]
                local_index = index[local_row]
                if minimum_score > 0:
                    valid = local_score >= minimum_score
                    local_score = local_score[valid]
                    local_index = local_index[valid]
                rows.extend([start + local_row] * len(local_index))
                columns.extend(local_index.cpu().tolist())
                values.extend(local_score.cpu().tolist())
    return sp.csr_matrix(
        (np.asarray(values, np.float32), (rows, columns)),
        shape=(len(heads), len(targets)),
    )


def construct_graphs(cfg: ExperimentConfig, checkins_path: str | Path) -> dict[str, Path]:
    del checkins_path  # The learned graphs depend on train-only STKG/KGE artifacts.
    checkpoint = torch.load(cfg.kge.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["meta"]
    model = TransE(
        metadata["n_entities"],
        metadata["n_relations"],
        checkpoint["dim"],
        checkpoint.get("p_norm", cfg.kge.p_norm),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    device = resolve_device(cfg.kge.device)
    model.to(device)

    n_users, n_pois = metadata["n_users"], metadata["n_pois"]
    output = Path(cfg.graphs.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    poi_entities = np.arange(n_users, n_users + n_pois, dtype=np.int64)
    user_entities = np.arange(n_users, dtype=np.int64)

    matrices: dict[str, sp.csr_matrix] = {
        "transition": _topk_relation(
            model,
            poi_entities,
            poi_entities,
            TEMPORAL,
            cfg.graphs.transition_topk,
            cfg.graphs.score_chunk_size,
            device,
            True,
            cfg.graphs.minimum_score,
            cfg.graphs.row_relative_scores,
        ),
        "preference": _topk_relation(
            model,
            user_entities,
            poi_entities,
            VISITS,
            cfg.graphs.user_poi_topk,
            cfg.graphs.score_chunk_size,
            device,
            False,
            cfg.graphs.minimum_score,
            cfg.graphs.row_relative_scores,
        ),
    }
    if cfg.model.use_friend_graph:
        matrices["friends"] = _topk_relation(
            model,
            user_entities,
            user_entities,
            FRIEND,
            cfg.graphs.friend_topk,
            cfg.graphs.score_chunk_size,
            device,
            True,
            cfg.graphs.minimum_score,
            cfg.graphs.row_relative_scores,
        )
    if cfg.model.use_spatial_graph:
        matrices["spatial"] = _topk_relation(
            model,
            poi_entities,
            poi_entities,
            SPATIAL,
            cfg.stkg.spatial_topk,
            cfg.graphs.score_chunk_size,
            device,
            True,
            cfg.graphs.minimum_score,
            cfg.graphs.row_relative_scores,
        )

    filenames = {
        "transition": "poi_transition.npz",
        "preference": "user_poi_preference.npz",
        "friends": "user_friend.npz",
        "spatial": "poi_spatial.npz",
    }
    paths: dict[str, Path] = {}
    manifest = {"n_users": n_users, "n_pois": n_pois, "graphs": {}}
    for name, matrix in matrices.items():
        path = output / filenames[name]
        save_sparse(matrix, path)
        paths[name] = path
        manifest["graphs"][name] = {
            "path": str(path),
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
        }
    dump_json(manifest, output / "graph_manifest.json")
    return paths
