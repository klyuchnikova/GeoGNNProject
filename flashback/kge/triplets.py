from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree

from flashback.config import ExperimentConfig, dump_json
from flashback.utils import read_table

VISITS, TEMPORAL, SPATIAL, FRIEND = 0, 1, 2, 3
RELATION_NAMES = {VISITS: "visits", TEMPORAL: "temporal", SPATIAL: "spatial", FRIEND: "friend"}


@dataclass
class TripletArtifacts:
    triplets: Path
    metadata: Path


def _spatial_pairs_rank(ids: np.ndarray, radians: np.ndarray, topk: int, symmetric: bool):
    if len(ids) <= 1 or topk <= 0:
        return []
    k = min(len(ids), topk + 1)
    tree = BallTree(radians, metric="haversine")
    _, indices = tree.query(radians, k=k, return_distance=True, sort_results=True)
    pairs: set[tuple[int, int]] = set()
    for source_index, neighbors in enumerate(indices):
        source = int(ids[source_index])
        added = 0
        for neighbor_index in neighbors:
            target = int(ids[int(neighbor_index)])
            if target == source:
                continue
            pairs.add((source, target))
            if symmetric:
                pairs.add((target, source))
            added += 1
            if added >= topk:
                break
    return sorted(pairs)


def _spatial_pairs_radius(ids: np.ndarray, radians: np.ndarray, radius_km: float, topk: int, symmetric: bool):
    if len(ids) <= 1 or topk <= 0:
        return []
    tree = BallTree(radians, metric="haversine")
    indices, _ = tree.query_radius(
        radians,
        r=radius_km / 6371.0088,
        return_distance=True,
        sort_results=True,
    )
    pairs: set[tuple[int, int]] = set()
    for source_index, neighbors in enumerate(indices):
        source = int(ids[source_index])
        added = 0
        for neighbor_index in neighbors:
            target = int(ids[int(neighbor_index)])
            if target == source:
                continue
            pairs.add((source, target))
            if symmetric:
                pairs.add((target, source))
            added += 1
            if added >= topk:
                break
    return sorted(pairs)


def _stratified_validation_split(arr: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep every relation represented in train while creating a small KGE validation set."""
    if len(arr) <= 10 or fraction <= 0:
        return arr, np.empty((0, 3), dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for relation in np.unique(arr[:, 1]):
        subset = arr[arr[:, 1] == relation]
        order = rng.permutation(len(subset))
        subset = subset[order]
        n_val = min(max(1, int(len(subset) * fraction)), max(0, len(subset) - 1))
        val_parts.append(subset[:n_val])
        train_parts.append(subset[n_val:])
    train = np.concatenate(train_parts, axis=0)
    validation = np.concatenate(val_parts, axis=0) if val_parts else np.empty((0, 3), dtype=np.int64)
    return train[rng.permutation(len(train))], validation[rng.permutation(len(validation))]


def generate_triplets(
    cfg: ExperimentConfig,
    checkins_path: str | Path,
    friendships_path: str | Path | None = None,
) -> TripletArtifacts:
    out = Path(cfg.stkg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkins = read_table(checkins_path)
    train = checkins[checkins.split == "train"].sort_values(["user_id", "timestamp"])
    n_users = int(checkins.user_id.max()) + 1
    n_pois = int(checkins.poi_id.max()) + 1
    rows: list[tuple[int, int, int]] = []

    # Entity ids: users [0, U), POIs [U, U + L).
    rows.extend((int(r.user_id), VISITS, n_users + int(r.poi_id)) for r in train.itertuples())
    for _, group in train.groupby("user_id", sort=False):
        pois = group.poi_id.to_numpy(np.int64)
        rows.extend(
            (n_users + int(left), TEMPORAL, n_users + int(right))
            for left, right in zip(pois[:-1], pois[1:])
            if left != right
        )

    poi_coords = train.groupby("poi_id")[["latitude", "longitude"]].median().sort_index()
    poi_ids = poi_coords.index.to_numpy(np.int64)
    radians = np.radians(poi_coords[["latitude", "longitude"]].to_numpy())
    if cfg.stkg.spatial_mode == "rank":
        spatial_pairs = _spatial_pairs_rank(
            poi_ids, radians, cfg.stkg.spatial_topk, cfg.stkg.spatial_symmetric
        )
    else:
        spatial_pairs = _spatial_pairs_radius(
            poi_ids,
            radians,
            cfg.stkg.spatial_radius_km,
            cfg.stkg.spatial_topk,
            cfg.stkg.spatial_symmetric,
        )
    if len(spatial_pairs) > cfg.stkg.max_spatial_pairs:
        spatial_pairs = spatial_pairs[: cfg.stkg.max_spatial_pairs]
    rows.extend((n_users + source, SPATIAL, n_users + target) for source, target in spatial_pairs)

    if cfg.stkg.include_friendship and friendships_path and Path(friendships_path).exists():
        friendships = read_table(friendships_path)
        rows.extend((int(r.user_a), FRIEND, int(r.user_b)) for r in friendships.itertuples())
        rows.extend((int(r.user_b), FRIEND, int(r.user_a)) for r in friendships.itertuples())

    arr = np.asarray(rows, dtype=np.int64)
    if cfg.stkg.deduplicate_triplets and len(arr):
        arr = np.unique(arr, axis=0)
    train_arr, validation_arr = _stratified_validation_split(
        arr, cfg.kge.validation_fraction, cfg.train.seed
    )
    np.save(out / "triplets_train.npy", train_arr)
    np.save(out / "triplets_validation.npy", validation_arr)

    counts = {
        RELATION_NAMES[relation]: int((arr[:, 1] == relation).sum())
        for relation in RELATION_NAMES
    } if len(arr) else {}
    manifest = {
        "n_users": n_users,
        "n_pois": n_pois,
        "n_entities": n_users + n_pois,
        "n_relations": 4,
        "relation_names": RELATION_NAMES,
        "counts": counts,
        "train_triplets": int(len(train_arr)),
        "validation_triplets": int(len(validation_arr)),
        "spatial_mode": cfg.stkg.spatial_mode,
        "spatial_topk": cfg.stkg.spatial_topk,
        "spatial_symmetric": cfg.stkg.spatial_symmetric,
        "source": "train split only",
    }
    dump_json(manifest, out / "stkg_manifest.json")
    return TripletArtifacts(out / "triplets_train.npy", out / "stkg_manifest.json")
