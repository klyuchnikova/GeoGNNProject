from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from flashback.data.adapters import adapter_for


@dataclass(frozen=True)
class CityBox:
    key: str
    label: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        return frame["latitude"].between(self.min_lat, self.max_lat) & frame["longitude"].between(self.min_lon, self.max_lon)


# Deliberately broad metro boxes. They are used for reproducible preselection; optional metadata can refine them.
CITY_BOXES: dict[str, CityBox] = {
    "new_york": CityBox("new_york", "New York metro", 40.45, 41.10, -74.35, -73.55),
    "los_angeles": CityBox("los_angeles", "Los Angeles metro", 33.55, 34.45, -118.85, -117.55),
    "chicago": CityBox("chicago", "Chicago metro", 41.55, 42.15, -88.10, -87.30),
    "san_francisco": CityBox("san_francisco", "San Francisco Bay Area", 37.20, 38.15, -122.75, -121.65),
    "austin": CityBox("austin", "Austin metro", 29.95, 30.65, -98.10, -97.35),
    "dallas": CityBox("dallas", "Dallas–Fort Worth", 32.35, 33.35, -97.60, -96.30),
    "seattle": CityBox("seattle", "Seattle metro", 47.25, 47.95, -122.65, -121.95),
    "boston": CityBox("boston", "Boston metro", 42.10, 42.65, -71.45, -70.75),
    "tokyo": CityBox("tokyo", "Tokyo", 35.45, 35.95, 139.35, 140.05),
}


def rank_cities(
    dataset: str,
    checkins_path: str | Path,
    candidate_keys: list[str],
    min_checkins: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    adapter = adapter_for(dataset)
    counters: dict[str, dict[str, object]] = {
        key: {"checkins": 0, "users": {}, "pois": set(), "min_time": None, "max_time": None}
        for key in candidate_keys
    }
    for chunk in adapter.iter_checkins(checkins_path, chunksize=chunksize):
        for key in candidate_keys:
            box = CITY_BOXES[key]
            part = chunk.loc[box.mask(chunk), ["raw_user_id", "raw_poi_id", "timestamp"]]
            if part.empty:
                continue
            state = counters[key]
            state["checkins"] = int(state["checkins"]) + len(part)
            counts = part["raw_user_id"].value_counts()
            users = state["users"]
            assert isinstance(users, dict)
            for user, count in counts.items():
                users[user] = users.get(user, 0) + int(count)
            pois = state["pois"]
            assert isinstance(pois, set)
            pois.update(part["raw_poi_id"].unique().tolist())
            local_min, local_max = part["timestamp"].min(), part["timestamp"].max()
            state["min_time"] = local_min if state["min_time"] is None else min(state["min_time"], local_min)
            state["max_time"] = local_max if state["max_time"] is None else max(state["max_time"], local_max)

    rows = []
    for key, state in counters.items():
        users = state["users"]
        assert isinstance(users, dict)
        eligible = [count for count in users.values() if count >= min_checkins]
        active_checkins = sum(eligible)
        span_days = 0
        if state["min_time"] is not None and state["max_time"] is not None:
            span_days = max(1, int((state["max_time"] - state["min_time"]).total_seconds() / 86400))
        # Eligibility dominates; POI diversity and temporal coverage break ties.
        score = np.log1p(active_checkins) * np.log1p(len(eligible)) * np.log1p(len(state["pois"])) * np.log1p(span_days)
        rows.append({
            "city": key,
            "label": CITY_BOXES[key].label,
            "checkins": int(state["checkins"]),
            "users": len(users),
            "eligible_users": len(eligible),
            "eligible_checkins": int(active_checkins),
            "pois": len(state["pois"]),
            "time_span_days": span_days,
            "score": float(score),
        })
    result = pd.DataFrame(rows).sort_values(
        ["score", "eligible_users", "eligible_checkins", "pois"], ascending=False
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def select_city(ranking: pd.DataFrame) -> str:
    viable = ranking[(ranking["eligible_users"] >= 20) & (ranking["pois"] >= 100)]
    if viable.empty:
        viable = ranking
    if viable.empty or int(viable.iloc[0]["eligible_users"]) == 0:
        raise ValueError("No viable city found. Lower min_checkins or provide a wider city box.")
    return str(viable.iloc[0]["city"])
