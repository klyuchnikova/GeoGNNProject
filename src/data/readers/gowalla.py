from pathlib import Path
import pandas as pd
import gzip
from .base import BaseReader

class GowallaReader(BaseReader):
    def __init__(self, root_dir: str | Path, city: str = "austin"):
        self.root_dir = Path(root_dir)
        self.city = city.lower()

    def load(self) -> pd.DataFrame:
        path = self.root_dir / f"gowalla_{self.city}_checkins.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Gowalla data not found: {path}")

        with gzip.open(path, 'rt', encoding='utf-8') as f:
            raw = pd.read_csv(f)

        df = pd.DataFrame({
            "user_id": raw["raw_user_id"].astype(int),
            "poi_id": raw["raw_poi_id"].astype(str),
            "timestamp": pd.to_datetime(raw["timestamp"], utc=True).astype("int64") // 10**9,
            "lat": raw["latitude"].astype(float),
            "lon": raw["longitude"].astype(float),
            "category": raw.get("category", 0).fillna(0).astype(str),
            "poi_name": raw.get("poi_name", ""),
        })
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        return self.validate(df)
