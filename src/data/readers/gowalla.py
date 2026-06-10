from pathlib import Path

import pandas as pd

from .base import BaseReader


class GowallaReader(BaseReader):
    """Load Gowalla check-ins from the SNAP tab-separated format."""

    def __init__(self, root_dir: str | Path, filename: str = "loc-gowalla_totalCheckins.txt"):
        self.root_dir = Path(root_dir)
        self.filename = filename

    def load(self) -> pd.DataFrame:
        path = self.root_dir / self.filename
        if not path.exists():
            raise FileNotFoundError(f"Gowalla data not found: {path}")

        raw = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["user_id", "timestamp", "lat", "lon", "poi_id"],
        )
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw["poi_id"] = raw["poi_id"].astype(str)
        raw["user_id"] = raw["user_id"].astype(int)

        df = pd.DataFrame(
            {
                "user_id": raw["user_id"],
                "poi_id": raw["poi_id"],
                "timestamp": raw["timestamp"].astype("int64") // 10**9,
                "lat": raw["lat"].astype(float),
                "lon": raw["lon"].astype(float),
                "category": 0,
                "poi_name": "",
            }
        )
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        return self.validate(df)
