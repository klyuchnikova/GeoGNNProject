from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from flashback.utils import read_table
from torch.utils.data import Dataset


class NextPoiSequenceDataset(Dataset):
    """Fixed-length chronological blocks with split-aware target masks.

    Inputs may contain earlier-split context, while loss/metrics are applied only
    to targets whose row belongs to the requested split. Blocks are non-overlapping
    by default, preventing duplicate evaluation targets.
    """

    def __init__(self, checkins: pd.DataFrame, split: str, sequence_length: int, stride: int):
        self.samples: list[tuple[np.ndarray, ...]] = []
        stride = max(1, int(stride))
        for user_id, group in checkins.groupby("user_id", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            loc = group["poi_id"].to_numpy(np.int64)
            ts = pd.to_datetime(group["timestamp"], utc=True).astype("int64").to_numpy(np.float64) / 1e9
            coords = group[["latitude", "longitude"]].to_numpy(np.float32)
            split_labels = group["split"].astype(str).to_numpy()
            n = len(group)
            if n < 2:
                continue
            starts = list(range(0, max(1, n - 1), stride))
            last_start = max(0, n - sequence_length - 1)
            if last_start not in starts:
                starts.append(last_start)
            seen_targets: set[int] = set()
            for start in sorted(set(starts)):
                end = min(start + sequence_length, n - 1)
                length = end - start
                if length <= 0:
                    continue
                target_positions = np.arange(start + 1, end + 1)
                mask = split_labels[target_positions] == split
                # With overlapping windows, count each target once.
                for j, pos in enumerate(target_positions):
                    if pos in seen_targets:
                        mask[j] = False
                if not mask.any():
                    continue
                seen_targets.update(target_positions[mask].tolist())
                pad = sequence_length - length
                in_loc = np.pad(loc[start:end], (0, pad), constant_values=0)
                in_ts = np.pad(ts[start:end], (0, pad), constant_values=0.0)
                in_coords = np.pad(coords[start:end], ((0, pad), (0, 0)), constant_values=0.0)
                targets = np.pad(loc[start + 1:end + 1], (0, pad), constant_values=0)
                target_mask = np.pad(mask.astype(np.bool_), (0, pad), constant_values=False)
                valid_input = np.zeros(sequence_length, dtype=np.bool_)
                valid_input[:length] = True
                self.samples.append((in_loc, in_ts, in_coords, targets, target_mask, valid_input, int(user_id)))
        if not self.samples:
            raise ValueError(f"No {split} targets. Check filtering, split ratios, and sequence_length.")

    @classmethod
    def from_parquet(cls, path: str | Path, split: str, sequence_length: int, stride: int):
        return cls(read_table(path), split, sequence_length, stride)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        loc, ts, coords, target, target_mask, valid_input, user = self.samples[index]
        return {
            "locations": torch.from_numpy(loc),
            "timestamps": torch.from_numpy(ts),
            "coordinates": torch.from_numpy(coords),
            "targets": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask),
            "valid_input": torch.from_numpy(valid_input),
            "user_id": torch.tensor(user, dtype=torch.long),
        }
