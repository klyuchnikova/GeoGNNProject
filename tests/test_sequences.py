import pandas as pd
from flashback.data.sequences import NextPoiSequenceDataset


def frame():
    n = 12
    return pd.DataFrame({
        "user_id": [0] * n,
        "poi_id": list(range(n)),
        "category_id": [1] * n,
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
        "latitude": [1.0] * n,
        "longitude": [2.0] * n,
        "split": ["train"] * 7 + ["validation"] * 2 + ["test"] * 3,
    })


def test_validation_can_use_train_context():
    ds = NextPoiSequenceDataset(frame(), "validation", 4, 4, "block_all")
    assert sum(int(x["target_mask"].sum()) for x in ds) == 2


def test_targets_not_duplicated():
    ds = NextPoiSequenceDataset(frame(), "test", 4, 2, "block_all")
    assert sum(int(x["target_mask"].sum()) for x in ds) == 3


def test_window_last_has_full_context_and_one_target():
    ds = NextPoiSequenceDataset(frame(), "test", 4, 1, "window_last")
    assert len(ds) == 3
    sample = ds[0]
    assert sample["locations"].shape[0] == 4
    assert sample["targets"].shape[0] == 1
    assert sample["target_mask"].tolist() == [True]
