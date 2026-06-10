from pathlib import Path

import pandas as pd

from .base import BaseReader

CITY_FILES = {
    "NYC": "dataset_TSMC2014_NYC.csv",
    "TKY": "dataset_TSMC2014_TKY.csv",
}


class FoursquareReader(BaseReader):
    """Load TSMC2014 Foursquare check-ins into the unified schema."""

    def __init__(self, root_dir: str | Path, city: str = "NYC"):
        self.root_dir = Path(root_dir)
        self.city = city.upper()
        if self.city not in CITY_FILES:
            raise ValueError(
                f"Unknown city '{city}'. Supported: {sorted(CITY_FILES)}"
            )

    def load(self) -> pd.DataFrame:
        path = self.root_dir / CITY_FILES[self.city]
        if not path.exists():
            raise FileNotFoundError(f"Foursquare data not found: {path}")

        raw = pd.read_csv(path)
        raw["utcTimestamp"] = pd.to_datetime(
            raw["utcTimestamp"],
            format="%a %b %d %H:%M:%S %z %Y",
            utc=True,
        )

        df = pd.DataFrame(
            {
                "user_id": raw["userId"].astype(int),
                "poi_id": raw["venueId"].astype(str),
                "timestamp": raw["utcTimestamp"].astype("int64") // 10**9,
                "lat": raw["latitude"].astype(float),
                "lon": raw["longitude"].astype(float),
                "category": raw["venueCategoryId"].astype(str),
                "poi_name": raw["venueCategory"].astype(str),
            }
        )
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        return self.validate(df)
