from __future__ import annotations
from pathlib import Path
import json, urllib.request
from flashback.utils import download

SNAP_CHECKINS = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
SNAP_EDGES = "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"
FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/22126586"


def download_gowalla(raw_dir: str | Path = "data/raw", with_metadata: bool = True) -> dict[str, str]:
    raw = Path(raw_dir); raw.mkdir(parents=True, exist_ok=True)
    paths = {
        "checkins": str(download(SNAP_CHECKINS, raw / "loc-gowalla_totalCheckins.txt.gz")),
        "friendships": str(download(SNAP_EDGES, raw / "loc-gowalla_edges.txt.gz")),
    }
    if with_metadata:
        req = urllib.request.Request(FIGSHARE_ARTICLE_API, headers={"User-Agent": "GeoGNNProject/0.2"})
        with urllib.request.urlopen(req, timeout=120) as response:
            article = json.load(response)
        candidates = [f for f in article.get("files", []) if f.get("name") == "gowalla_spots_subset1.csv"]
        if candidates:
            paths["metadata"] = str(download(candidates[0]["download_url"], raw / candidates[0]["name"]))
    return paths


def download_foursquare_kaggle(dataset_slug: str, raw_dir: str | Path = "data/raw/foursquare") -> Path:
    """Download through Kaggle CLI (requires credentials or a Kaggle notebook session)."""
    import subprocess
    raw = Path(raw_dir); raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(raw), "--unzip"], check=True)
    return raw
