from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class PoiGraphs:
    """POI-level graphs built from training check-ins only."""

    transition_edge_index: torch.Tensor
    transition_edge_weight: torch.Tensor
    geo_edge_index: torch.Tensor
    geo_edge_weight: torch.Tensor
    poi_coords: torch.Tensor
    poi_category: torch.Tensor

    def to(self, device: torch.device) -> "PoiGraphs":
        return PoiGraphs(
            transition_edge_index=self.transition_edge_index.to(device),
            transition_edge_weight=self.transition_edge_weight.to(device),
            geo_edge_index=self.geo_edge_index.to(device),
            geo_edge_weight=self.geo_edge_weight.to(device),
            poi_coords=self.poi_coords.to(device),
            poi_category=self.poi_category.to(device),
        )


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _tensorize_edges(edges: dict[tuple[int, int], float]) -> tuple[torch.Tensor, torch.Tensor]:
    if not edges:
        empty = torch.zeros((2, 0), dtype=torch.long)
        return empty, torch.zeros(0, dtype=torch.float32)
    src = torch.tensor([k[0] for k in edges], dtype=torch.long)
    dst = torch.tensor([k[1] for k in edges], dtype=torch.long)
    weight = torch.tensor(list(edges.values()), dtype=torch.float32)
    return torch.stack([src, dst], dim=0), weight


def _build_transition_edges(train_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    edges: dict[tuple[int, int], float] = {}
    ordered = train_df.sort_values(["user_id", "timestamp"])
    for _, user_df in ordered.groupby("user_id", sort=False):
        pois = user_df["poi_id"].astype(int).tolist()
        for src, dst in zip(pois[:-1], pois[1:]):
            if src == dst:
                continue
            key = (src, dst)
            edges[key] = edges.get(key, 0.0) + 1.0

    return _tensorize_edges(edges)


def _build_geo_edges(
    poi_coords: np.ndarray,
    poi_ids: np.ndarray,
    geo_dist_km: float,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Connect each POI only to its nearest spatial neighbors (sparse kNN graph)."""
    n = len(poi_ids)
    if n <= 1:
        empty = torch.zeros((2, 0), dtype=torch.long)
        return empty, torch.zeros(0, dtype=torch.float32)

    lats = poi_coords[:, 0]
    lons = poi_coords[:, 1]
    edges: dict[tuple[int, int], float] = {}
    k = max(1, max_neighbors)

    for i in range(n):
        dists = _haversine_km(lats[i], lons[i], lats, lons)
        dists[i] = np.inf
        order = np.argsort(dists)
        added = 0
        for j in order:
            if added >= k:
                break
            dist = dists[j]
            if dist > geo_dist_km:
                break
            src = int(poi_ids[i])
            dst = int(poi_ids[j])
            weight = float(1.0 / (dist + 1e-3))
            edges[(src, dst)] = max(edges.get((src, dst), 0.0), weight)
            edges[(dst, src)] = max(edges.get((dst, src), 0.0), weight)
            added += 1

    return _tensorize_edges(edges)


def build_poi_graphs(
    train_df: pd.DataFrame,
    num_pois: int,
    num_categories: int,
    geo_dist_km: float = 0.5,
    max_geo_neighbors: int = 10,
) -> PoiGraphs:
    """Build transition and geo POI graphs using training split only."""
    poi_meta = (
        train_df.groupby("poi_id")
        .agg(lat=("lat", "mean"), lon=("lon", "mean"), category=("category", "first"))
        .reset_index()
    )
    poi_meta["poi_id"] = poi_meta["poi_id"].astype(int)
    poi_meta["category"] = poi_meta["category"].astype(int)

    poi_coords = torch.zeros(num_pois, 2, dtype=torch.float32)
    poi_category = torch.zeros(num_pois, dtype=torch.long)
    for row in poi_meta.itertuples(index=False):
        pid = int(row.poi_id)
        poi_coords[pid, 0] = float(row.lat)
        poi_coords[pid, 1] = float(row.lon)
        poi_category[pid] = int(row.category)

    active = poi_meta["poi_id"].astype(int).values
    active_coords = poi_coords[active].numpy().copy()

    trans_edge_index, trans_edge_weight = _build_transition_edges(train_df)
    geo_edge_index, geo_edge_weight = _build_geo_edges(
        active_coords,
        active,
        geo_dist_km=geo_dist_km,
        max_neighbors=max_geo_neighbors,
    )

    if geo_edge_index.numel() == 0 and trans_edge_index.numel() > 0:
        geo_edge_index = trans_edge_index.clone()
        geo_edge_weight = trans_edge_weight.clone()

    return PoiGraphs(
        transition_edge_index=trans_edge_index,
        transition_edge_weight=trans_edge_weight,
        geo_edge_index=geo_edge_index,
        geo_edge_weight=geo_edge_weight,
        poi_coords=poi_coords,
        poi_category=poi_category,
    )
