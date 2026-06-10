from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd

@dataclass
class Checkin:
    user_id: int
    poi_id: int

    timestamp: int

    lat: float
    lon: float

    category_id: int | None
    category_name: str | None

    poi_name: str | None

class BaseReader(ABC):

    REQUIRED_COLUMNS = [
        "user_id",
        "poi_id",
        "timestamp",
        "lat",
        "lon",
        "category",
        "poi_name",
    ]

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

    def validate(self, df):
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df
