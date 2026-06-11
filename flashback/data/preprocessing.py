from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from flashback.config import ExperimentConfig, dump_json
from flashback.utils import write_table
from flashback.data.adapters import adapter_for
from flashback.data.cities import CITY_BOXES, rank_cities, select_city


@dataclass
class PreparedPaths:
    checkins: Path
    friendships: Path | None
    users: Path
    pois: Path
    legacy_txt: Path
    metadata: Path
    categories: Path | None = None


def _load_metadata(path: str | Path | None) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    id_candidates = ["spotid", "placeid", "location_id", "poi_id", "id"]
    id_col = next((c for c in id_candidates if c in frame.columns), None)
    if id_col is None:
        return None
    rename = {id_col: "raw_poi_id"}
    category_col = next((c for c in ["spot_categories", "spot_category", "category", "category_name"] if c in frame.columns), None)
    name_col = next((c for c in ["spotname", "spot_name", "name", "poi_name"] if c in frame.columns), None)
    if category_col:
        rename[category_col] = "metadata_category"
    if name_col:
        rename[name_col] = "metadata_poi_name"
    keep = [id_col] + ([category_col] if category_col else []) + ([name_col] if name_col else [])
    frame = frame[keep].rename(columns=rename).drop_duplicates("raw_poi_id")
    return frame


def _min_filter_once(frame: pd.DataFrame, min_checkins: int, min_poi_visits: int) -> pd.DataFrame:
    """Sequential user-min then POI-min filtering for filtration sweeps."""
    frame = frame.copy()
    if min_checkins > 1:
        user_counts = frame["raw_user_id"].value_counts()
        frame = frame[frame["raw_user_id"].isin(user_counts[user_counts >= min_checkins].index)]
    if min_poi_visits > 1:
        poi_counts = frame["raw_poi_id"].value_counts()
        frame = frame[frame["raw_poi_id"].isin(poi_counts[poi_counts >= min_poi_visits].index)]
    return frame


def _iterative_filter(frame: pd.DataFrame, user_k: int, poi_k: int) -> pd.DataFrame:
    frame = frame.copy()
    previous = -1
    while previous != len(frame):
        previous = len(frame)
        if user_k > 1:
            user_counts = frame["raw_user_id"].value_counts()
            frame = frame[frame["raw_user_id"].isin(user_counts[user_counts >= user_k].index)]
        if poi_k > 1:
            poi_counts = frame["raw_poi_id"].value_counts()
            frame = frame[frame["raw_poi_id"].isin(poi_counts[poi_counts >= poi_k].index)]
    return frame


def _apply_filtering(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    mode = cfg.data.filter_mode
    if mode == "none":
        return frame.copy()
    if mode == "min_only":
        return _min_filter_once(frame, cfg.data.min_checkins, cfg.data.min_poi_visits)
    if mode == "combined":
        frame = _min_filter_once(frame, cfg.data.min_checkins, cfg.data.min_poi_visits)
        return _iterative_filter(
            frame,
            cfg.data.user_k or cfg.data.min_checkins,
            cfg.data.poi_k or cfg.data.min_poi_visits,
        )
    if mode == "iterative_kcore":
        return _iterative_filter(
            frame,
            cfg.data.user_k or cfg.data.min_checkins,
            cfg.data.poi_k or cfg.data.min_poi_visits,
        )
    raise ValueError(f"Unknown filter mode: {mode}")


def _split_users(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("user_id", sort=False):
        group = group.sort_values("timestamp").copy()
        n = len(group)
        train_end = max(1, int(n * cfg.data.train_ratio))
        val_end = train_end + int(n * cfg.data.val_ratio)
        val_end = min(max(val_end, train_end), n)
        split = np.full(n, "test", dtype=object)
        split[:train_end] = "train"
        if val_end > train_end:
            split[train_end:val_end] = "validation"
        group["split"] = split
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _read_friendships(path: str | Path | None, user_map: dict[Any, int]) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    path = Path(path)
    first = path.open("rb").readline().decode("utf-8", errors="replace") if path.suffix != ".gz" else ""
    if path.name.endswith(".csv") or path.name.endswith(".csv.gz"):
        edges = pd.read_csv(path, compression="infer")
        edges.columns = [str(c).strip().lower() for c in edges.columns]
        aliases = {"user_a": "raw_user_a", "user_b": "raw_user_b", "source": "raw_user_a", "target": "raw_user_b"}
        edges = edges.rename(columns={c: aliases.get(c, c) for c in edges.columns})
        if not {"raw_user_a", "raw_user_b"}.issubset(edges.columns):
            if edges.shape[1] < 2:
                raise ValueError("Friendship CSV must contain two user columns")
            edges = edges.iloc[:, :2]
            edges.columns = ["raw_user_a", "raw_user_b"]
    else:
        edges = pd.read_csv(path, sep="\t", names=["raw_user_a", "raw_user_b"], header=None, compression="infer")
    edges = edges[edges["raw_user_a"].isin(user_map) & edges["raw_user_b"].isin(user_map)].copy()
    if edges.empty:
        return edges.assign(user_a=pd.Series(dtype=int), user_b=pd.Series(dtype=int))
    edges["user_a"] = edges["raw_user_a"].map(user_map).astype(int)
    edges["user_b"] = edges["raw_user_b"].map(user_map).astype(int)
    return edges[["raw_user_a", "raw_user_b", "user_a", "user_b"]].drop_duplicates()


def prepare_dataset(cfg: ExperimentConfig) -> PreparedPaths:
    output = Path(cfg.data.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ranking = rank_cities(
        cfg.data.dataset,
        cfg.data.raw_checkins,
        cfg.data.candidate_cities,
        cfg.data.min_checkins,
    ) if cfg.data.dataset == "gowalla" and cfg.data.candidate_cities and not cfg.data.preselected_city else pd.DataFrame()
    if not ranking.empty:
        ranking.to_csv(output / "city_ranking.csv", index=False)
    city = cfg.data.city
    if city == "auto":
        city = select_city(ranking) if cfg.data.dataset == "gowalla" else "all"
    if city != "all" and city not in CITY_BOXES:
        raise KeyError(f"Unknown city box {city}; available: {sorted(CITY_BOXES)}")

    adapter = adapter_for(cfg.data.dataset, cfg.data.input_format)
    chunks = []
    for chunk in adapter.iter_checkins(cfg.data.raw_checkins):
        if city != "all" and not cfg.data.preselected_city:
            chunk = chunk[CITY_BOXES[city].mask(chunk)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No check-ins found for city={city}")
    frame = pd.concat(chunks, ignore_index=True)
    metadata = _load_metadata(cfg.data.raw_metadata)
    if metadata is not None:
        frame = frame.merge(metadata, on="raw_poi_id", how="left")
        if "metadata_category" in frame:
            frame["category"] = frame["category"].fillna(frame["metadata_category"])
        if "metadata_poi_name" in frame:
            frame["poi_name"] = frame["poi_name"].fillna(frame["metadata_poi_name"])

    raw_rows_before_filter = len(frame)
    frame = _apply_filtering(frame, cfg)
    if cfg.data.max_users > 0:
        top_users = frame["raw_user_id"].value_counts().head(cfg.data.max_users).index
        frame = frame[frame["raw_user_id"].isin(top_users)]
    if frame.empty:
        raise ValueError("Filtering removed all check-ins")

    raw_users = sorted(frame["raw_user_id"].unique().tolist(), key=str)
    raw_pois = sorted(frame["raw_poi_id"].unique().tolist(), key=str)
    user_map = {value: idx for idx, value in enumerate(raw_users)}
    poi_map = {value: idx for idx, value in enumerate(raw_pois)}
    frame["user_id"] = frame["raw_user_id"].map(user_map).astype(int)
    frame["poi_id"] = frame["raw_poi_id"].map(poi_map).astype(int)

    # Category IDs are optional side information for the tuned Flashback model.
    # 0 is reserved for unknown/missing categories so the same code also works
    # for Gowalla subsets or Foursquare files without metadata.
    category_text = frame["category"].astype("string").fillna("__unknown__").str.strip()
    category_text = category_text.replace("", "__unknown__")
    categories = sorted([x for x in category_text.unique().tolist() if x != "__unknown__"], key=str)
    category_map = {"__unknown__": 0, **{value: idx + 1 for idx, value in enumerate(categories)}}
    frame["category_id"] = category_text.map(category_map).fillna(0).astype(int)

    frame = frame.sort_values(["user_id", "timestamp", "poi_id"]).reset_index(drop=True)
    frame = _split_users(frame, cfg)

    users = pd.DataFrame({"raw_user_id": raw_users, "user_id": range(len(raw_users))})
    poi_info = frame.sort_values("timestamp").groupby("poi_id", as_index=False).agg(
        raw_poi_id=("raw_poi_id", "first"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        category=("category", "first"),
        poi_name=("poi_name", "first"),
        checkins=("poi_id", "size"),
    )
    friendships = _read_friendships(cfg.data.raw_friendships, user_map)

    checkins_path = output / f"{cfg.data.dataset}_{city}_checkins.parquet"
    users_path = output / f"{cfg.data.dataset}_{city}_users.csv"
    pois_path = output / f"{cfg.data.dataset}_{city}_pois.csv"
    friendships_path = output / f"{cfg.data.dataset}_{city}_friendships.parquet" if friendships is not None else None
    legacy_path = output / f"checkins-{cfg.data.dataset}-{city}.txt"
    metadata_path = output / f"{cfg.data.dataset}_{city}_manifest.json"
    categories_path = output / f"{cfg.data.dataset}_{city}_categories.csv"
    write_table(frame, checkins_path, index=False)
    users.to_csv(users_path, index=False)
    poi_info.to_csv(pois_path, index=False)
    pd.DataFrame({"category": list(category_map.keys()), "category_id": list(category_map.values())}).to_csv(categories_path, index=False)
    if friendships_path is not None:
        write_table(friendships, friendships_path, index=False)

    # Original loader compatibility: user-contiguous and descending timestamps.
    legacy = frame.sort_values(["user_id", "timestamp"], ascending=[True, False])
    with legacy_path.open("w", encoding="utf-8") as handle:
        for row in legacy.itertuples(index=False):
            ts = pd.Timestamp(row.timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
            handle.write(f"{row.user_id}\t{ts}\t{row.latitude:.8f}\t{row.longitude:.8f}\t{row.poi_id}\n")

    split_counts = frame.groupby("split").size().to_dict()
    manifest = {
        "dataset": cfg.data.dataset,
        "city": city,
        "checkins": len(frame),
        "users": len(users),
        "pois": len(poi_info),
        "categories": len(category_map),
        "friend_edges": 0 if friendships is None else len(friendships),
        "split_counts": split_counts,
        "raw_rows_before_filter": int(raw_rows_before_filter),
        "filter_mode": cfg.data.filter_mode,
        "min_checkins": cfg.data.min_checkins,
        "min_poi_visits": cfg.data.min_poi_visits,
        "user_k": cfg.data.user_k or cfg.data.min_checkins,
        "poi_k": cfg.data.poi_k or cfg.data.min_poi_visits,
        "metadata_category_coverage": float(frame["category"].notna().mean()),
        "metadata_name_coverage": float(frame["poi_name"].notna().mean()),
        "timestamp_min": frame["timestamp"].min(),
        "timestamp_max": frame["timestamp"].max(),
        "config": cfg.to_dict(),
    }
    dump_json(manifest, metadata_path)
    return PreparedPaths(checkins_path, friendships_path, users_path, pois_path, legacy_path, metadata_path, categories_path)
