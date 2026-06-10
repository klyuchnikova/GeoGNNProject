from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from itertools import chain

import numpy as np
import pandas as pd

from flashback.utils import open_maybe_gzip

CANONICAL_COLUMNS = [
    "raw_user_id", "timestamp", "latitude", "longitude", "raw_poi_id", "category", "poi_name"
]


class DatasetAdapter:
    def iter_checkins(self, path: str | Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
        raise NotImplementedError


class GowallaSnapAdapter(DatasetAdapter):
    """Reads the official SNAP Gowalla check-in file.

    Expected columns: user, ISO UTC timestamp, latitude, longitude, location id.
    """

    def iter_checkins(self, path: str | Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
        names = ["raw_user_id", "timestamp", "latitude", "longitude", "raw_poi_id"]
        for chunk in pd.read_csv(
            path,
            sep="\t",
            names=names,
            header=None,
            compression="infer",
            chunksize=chunksize,
            dtype={"raw_user_id": "int64", "raw_poi_id": "int64"},
        ):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
            chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
            chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
            chunk["category"] = pd.NA
            chunk["poi_name"] = pd.NA
            yield _clean_canonical(chunk)


class FoursquareTSMCAdapter(DatasetAdapter):
    """Reads TSMC2014 NYC/Tokyo CSV or TXT files with flexible headers."""

    def iter_checkins(self, path: str | Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
        path = Path(path)
        first_line = path.open("rb").readline().decode("latin-1", errors="replace")
        sep = "\t" if first_line.count("\t") >= first_line.count(",") else ","
        last_error = None
        for encoding in ("utf-8", "latin-1"):
            try:
                iterator = pd.read_csv(path, sep=sep, chunksize=chunksize, encoding=encoding)
                first = next(iter(iterator))
                for chunk in chain([first], iterator):
                    yield self._normalize(chunk)
                return
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"Could not parse Foursquare file {path}: {last_error}")

    @staticmethod
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        aliases = {
            "userid": "raw_user_id", "user_id": "raw_user_id", "user": "raw_user_id",
            "venueid": "raw_poi_id", "venue_id": "raw_poi_id", "placeid": "raw_poi_id", "poi_id": "raw_poi_id",
            "utctimestamp": "timestamp", "datetime": "timestamp", "time": "timestamp",
            "lat": "latitude", "lng": "longitude", "lon": "longitude",
            "venuecategory": "category", "spot_categ": "category", "spot_categories": "category",
            "venuename": "poi_name", "spotname": "poi_name", "name": "poi_name",
        }
        frame = frame.rename(columns={c: aliases.get(c, c) for c in frame.columns})
        required = {"raw_user_id", "raw_poi_id", "timestamp", "latitude", "longitude"}
        missing = required - set(frame.columns)
        if missing and frame.shape[1] >= 8:
            # Standard TSMC order if headers were lost.
            cols = list(frame.columns)
            frame = frame.iloc[:, :8]
            frame.columns = [
                "raw_user_id", "raw_poi_id", "venue_category_id", "category",
                "latitude", "longitude", "timezone_offset", "timestamp",
            ]
            missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Foursquare file misses columns: {sorted(missing)}")
        if "category" not in frame:
            frame["category"] = pd.NA
        if "poi_name" not in frame:
            frame["poi_name"] = pd.NA
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return _clean_canonical(frame[CANONICAL_COLUMNS])


def adapter_for(name: str) -> DatasetAdapter:
    if name == "gowalla":
        return GowallaSnapAdapter()
    if name == "foursquare":
        return FoursquareTSMCAdapter()
    raise ValueError(f"Unsupported dataset adapter: {name}")


def _clean_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.dropna(subset=["raw_user_id", "raw_poi_id", "timestamp", "latitude", "longitude"])
    frame = frame[frame["latitude"].between(-90, 90) & frame["longitude"].between(-180, 180)]
    frame = frame.drop_duplicates(subset=["raw_user_id", "raw_poi_id", "timestamp", "latitude", "longitude"])
    return frame
