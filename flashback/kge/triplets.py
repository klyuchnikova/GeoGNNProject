from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from flashback.config import ExperimentConfig, dump_json
from flashback.utils import read_table

VISITS, TEMPORAL, SPATIAL, FRIEND = 0, 1, 2, 3
RELATION_NAMES = {VISITS: "visits", TEMPORAL: "temporal", SPATIAL: "spatial", FRIEND: "friend"}

@dataclass
class TripletArtifacts:
    triplets: Path
    metadata: Path


def generate_triplets(cfg: ExperimentConfig, checkins_path: str | Path, friendships_path: str | Path | None = None) -> TripletArtifacts:
    out = Path(cfg.stkg.output_dir); out.mkdir(parents=True, exist_ok=True)
    checkins = read_table(checkins_path)
    train = checkins[checkins.split == "train"].sort_values(["user_id", "timestamp"])
    n_users = int(checkins.user_id.max()) + 1
    n_pois = int(checkins.poi_id.max()) + 1
    rows: list[tuple[int,int,int]] = []
    # entity ids: users [0,U), POIs [U,U+L)
    rows.extend((int(r.user_id), VISITS, n_users + int(r.poi_id)) for r in train.itertuples())
    for _, g in train.groupby("user_id", sort=False):
        p = g.poi_id.to_numpy(np.int64)
        rows.extend((n_users + int(a), TEMPORAL, n_users + int(b)) for a,b in zip(p[:-1], p[1:]) if a != b)
    poi_coords = train.groupby("poi_id")[["latitude","longitude"]].median().sort_index()
    ids = poi_coords.index.to_numpy(np.int64)
    if len(ids) > 1:
        rad = np.radians(poi_coords[["latitude","longitude"]].to_numpy())
        tree = BallTree(rad, metric="haversine")
        ind, dist = tree.query_radius(rad, r=cfg.stkg.spatial_radius_km / 6371.0088, return_distance=True, sort_results=True)
        pair_count = 0
        for source, neigh, ds in zip(ids, ind, dist):
            added = 0
            for j, d in zip(neigh, ds):
                target = ids[j]
                if target == source: continue
                rows.append((n_users + int(source), SPATIAL, n_users + int(target)))
                added += 1; pair_count += 1
                if added >= cfg.stkg.spatial_topk or pair_count >= cfg.stkg.max_spatial_pairs: break
            if pair_count >= cfg.stkg.max_spatial_pairs: break
    if cfg.stkg.include_friendship and friendships_path and Path(friendships_path).exists():
        fr = read_table(friendships_path)
        rows.extend((int(r.user_a), FRIEND, int(r.user_b)) for r in fr.itertuples())
        rows.extend((int(r.user_b), FRIEND, int(r.user_a)) for r in fr.itertuples())
    arr = np.asarray(rows, dtype=np.int64)
    if cfg.stkg.deduplicate_triplets and len(arr):
        arr = np.unique(arr, axis=0)
    rng = np.random.default_rng(cfg.train.seed)
    order = rng.permutation(len(arr)); arr = arr[order]
    n_val = max(1, int(len(arr) * cfg.kge.validation_fraction)) if len(arr) > 10 else 0
    val = arr[:n_val]; tr = arr[n_val:]
    np.save(out / "triplets_train.npy", tr)
    np.save(out / "triplets_validation.npy", val)
    counts = {RELATION_NAMES[r]: int((arr[:,1] == r).sum()) for r in RELATION_NAMES} if len(arr) else {}
    meta = {"n_users": n_users, "n_pois": n_pois, "n_entities": n_users+n_pois,
            "n_relations": 4, "relation_names": RELATION_NAMES, "counts": counts,
            "source": "train split only"}
    dump_json(meta, out / "stkg_manifest.json")
    return TripletArtifacts(out / "triplets_train.npy", out / "stkg_manifest.json")
