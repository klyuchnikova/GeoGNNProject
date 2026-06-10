from pathlib import Path

import pandas as pd
import yaml

from flashback.config import load_config
from flashback.data.preprocessing import prepare_dataset
from flashback.utils import read_table


def test_preselected_canonical_gowalla(tmp_path: Path):
    data = pd.DataFrame({
        "raw_user_id": [1] * 6 + [2] * 6,
        "timestamp": pd.date_range("2020-01-01", periods=12, freq="h", tz="UTC"),
        "latitude": [30.2] * 12,
        "longitude": [-97.7] * 12,
        "raw_poi_id": [10, 11, 10, 12, 10, 11] * 2,
        "category": ["Coffee", "Office", "Coffee", "Park", "Coffee", "Office"] * 2,
    })
    checkins = tmp_path / "austin.csv.gz"
    data.to_csv(checkins, index=False, compression="gzip")
    friends = tmp_path / "friends.csv.gz"
    pd.DataFrame({"raw_user_a": [1], "raw_user_b": [2]}).to_csv(friends, index=False, compression="gzip")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "data": {
            "dataset": "gowalla",
            "input_format": "canonical_csv",
            "preselected_city": True,
            "raw_checkins": str(checkins),
            "raw_friendships": str(friends),
            "raw_metadata": None,
            "city": "austin",
            "candidate_cities": [],
            "min_checkins": 2,
            "min_poi_visits": 1,
            "sequence_length": 2,
            "sequence_stride": 1,
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "output_dir": str(tmp_path / "processed"),
        }
    }), encoding="utf-8")
    cfg = load_config(cfg_path)
    paths = prepare_dataset(cfg)
    prepared = read_table(paths.checkins)
    assert len(prepared) == 12
    assert prepared["user_id"].nunique() == 2
    assert prepared["poi_id"].nunique() == 3
    assert prepared["category_id"].nunique() >= 3
    assert paths.friendships is not None and paths.friendships.exists()
