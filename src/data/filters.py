from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

class BaseFilter(ABC):
    @abstractmethod
    def apply(self, df):
        pass

class MinUserVisitsFilter(BaseFilter):
    def __init__(self, k):
        self.k = k
    def apply(self, df):
        valid = df.groupby("user_id").size()[lambda x: x >= self.k].index
        return df[df.user_id.isin(valid)]

class MinPoiVisitsFilter(BaseFilter):
    def __init__(self, k):
        self.k = k
    def apply(self, df):
        valid = df.groupby("poi_id").size()[lambda x: x >= self.k].index
        return df[df.poi_id.isin(valid)]

class IterativeKCoreFilter(BaseFilter):
    def __init__(self, user_k=10, poi_k=10):
        self.user_k, self.poi_k = user_k, poi_k
    def apply(self, df):
        prev_size = -1
        while prev_size != len(df):
            prev_size = len(df)
            valid_users = df.groupby("user_id").size()[lambda x: x >= self.user_k].index
            df = df[df.user_id.isin(valid_users)]
            valid_pois = df.groupby("poi_id").size()[lambda x: x >= self.poi_k].index
            df = df[df.poi_id.isin(valid_pois)]
        return df

class UserEntropyFilter(BaseFilter):
    def __init__(self, min_entropy):
        self.min_entropy = min_entropy
    def apply(self, df):
        keep = [uid for uid, g in df.groupby("user_id") 
                if -np.sum((p := g.poi_id.value_counts(normalize=True).values) * np.log(p + 1e-12)) >= self.min_entropy]
        return df[df.user_id.isin(keep)]

class GeoBoundingBoxFilter(BaseFilter):
    """Keep check-ins inside a lat/lon bounding box (useful for Gowalla city subsets)."""

    def __init__(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float):
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

    def apply(self, df):
        mask = (
            (df["lat"] >= self.min_lat)
            & (df["lat"] <= self.max_lat)
            & (df["lon"] >= self.min_lon)
            & (df["lon"] <= self.max_lon)
        )
        return df[mask]


class DominantPoiFilter(BaseFilter):
    def __init__(self, max_ratio=0.8):
        self.max_ratio = max_ratio
    def apply(self, df):
        keep = [uid for uid, g in df.groupby("user_id") 
                if g.poi_id.value_counts(normalize=True).max() <= self.max_ratio]
        return df[df.user_id.isin(keep)]


@dataclass
class FilterConfig:
    min_user_visits: int = 10
    min_poi_visits: int = 10
    use_kcore: bool = True
    user_k: int = 10
    poi_k: int = 10
    min_entropy: float | None = None
    max_dominant_ratio: float | None = None
    bbox: tuple[float, float, float, float] | None = None


def filter_pipeline(df: pd.DataFrame, cfg: FilterConfig) -> pd.DataFrame:
    """Apply the configured filtration steps in a fixed order."""
    filters: list[BaseFilter] = []

    if cfg.bbox is not None:
        min_lat, max_lat, min_lon, max_lon = cfg.bbox
        filters.append(GeoBoundingBoxFilter(min_lat, max_lat, min_lon, max_lon))
    if cfg.min_user_visits > 0:
        filters.append(MinUserVisitsFilter(cfg.min_user_visits))
    if cfg.min_poi_visits > 0:
        filters.append(MinPoiVisitsFilter(cfg.min_poi_visits))
    if cfg.use_kcore:
        filters.append(IterativeKCoreFilter(cfg.user_k, cfg.poi_k))
    if cfg.min_entropy is not None:
        filters.append(UserEntropyFilter(cfg.min_entropy))
    if cfg.max_dominant_ratio is not None:
        filters.append(DominantPoiFilter(cfg.max_dominant_ratio))

    for filt in filters:
        df = filt.apply(df)
    return df.reset_index(drop=True)