from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from flashback.utils import read_table
from torch.utils.data import Dataset


class NextPoiSequenceDataset(Dataset):
    """Chronological next-POI samples with split-aware context.

    ``block_all`` matches the original fixed-block formulation: all next steps
    inside a block can contribute to loss/metrics.

    ``window_last`` is the recommended tuned mode: each target is predicted
    from a complete rolling context window and appears exactly once. It removes
    the artificial zero-state boundary every 20 events without leaking future
    observations.
    """

    def __init__(
        self,
        checkins: pd.DataFrame,
        split: str,
        sequence_length: int,
        stride: int,
        mode: str = "block_all",
        history_topk: int = 100,
    ):
        self.samples: list[tuple[np.ndarray, ...]] = []
        self.n_pois = int(checkins["poi_id"].max()) + 1
        self.history_topk = max(1, min(int(history_topk), self.n_pois))
        stride = max(1, int(stride))
        for user_id, group in checkins.groupby("user_id", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            loc = group["poi_id"].to_numpy(np.int64)
            cat = (
                group["category_id"].fillna(0).to_numpy(np.int64)
                if "category_id" in group
                else np.zeros(len(group), dtype=np.int64)
            )
            ts = pd.to_datetime(group["timestamp"], utc=True).astype("int64").to_numpy(np.float64) / 1e9
            coords = group[["latitude", "longitude"]].to_numpy(np.float32)
            split_labels = group["split"].astype(str).to_numpy()
            if len(group) < 2:
                continue
            if mode == "window_last":
                self._append_window_samples(
                    int(user_id), loc, cat, ts, coords, split_labels,
                    split, sequence_length, stride,
                )
            elif mode == "block_all":
                self._append_block_samples(
                    int(user_id), loc, cat, ts, coords, split_labels,
                    split, sequence_length, stride,
                )
            else:
                raise ValueError(f"Unknown sequence mode: {mode}")
        if not self.samples:
            raise ValueError(f"No {split} targets. Check filtering, split ratios, and sequence settings.")

    def _history_prior(self, counts: np.ndarray, output_steps: int = 1) -> tuple[np.ndarray, np.ndarray]:
        topk = self.history_topk
        indices = np.zeros((output_steps, topk), dtype=np.int64)
        values = np.zeros((output_steps, topk), dtype=np.float32)
        if counts.sum() <= 0:
            return indices, values
        score = np.log1p(counts.astype(np.float32))
        positive = np.flatnonzero(score > 0)
        if positive.size == 0:
            return indices, values
        keep = min(topk, positive.size)
        selected = positive[np.argpartition(score[positive], -keep)[-keep:]]
        selected = selected[np.argsort(score[selected])[::-1]]
        selected_values = score[selected]
        selected_values = selected_values / max(float(selected_values.max()), 1e-12)
        indices[:, :keep] = selected[None, :]
        values[:, :keep] = selected_values[None, :]
        return indices, values

    def _append_window_samples(
        self, user_id, loc, cat, ts, coords, split_labels,
        split, sequence_length, stride,
    ):
        # Every evaluated target has exactly sequence_length preceding events.
        # This is the same target construction used by strong sequential POI
        # baselines and avoids losing context at arbitrary block boundaries.
        candidate_positions = [
            pos for pos in range(sequence_length, len(loc))
            if split_labels[pos] == split
        ]
        counts = np.zeros(self.n_pois, dtype=np.float32)
        next_candidate = 0
        for target_pos in candidate_positions[::stride]:
            while next_candidate < target_pos:
                counts[loc[next_candidate]] += 1.0
                next_candidate += 1
            start = target_pos - sequence_length
            in_loc = loc[start:target_pos].copy()
            in_cat = cat[start:target_pos].copy()
            in_ts = ts[start:target_pos].copy()
            in_coords = coords[start:target_pos].copy()
            targets = np.asarray([loc[target_pos]], dtype=np.int64)
            target_mask = np.asarray([True], dtype=np.bool_)
            valid_input = np.ones(sequence_length, dtype=np.bool_)
            history_index, history_value = self._history_prior(counts, output_steps=1)
            self.samples.append(
                (
                    in_loc,
                    in_cat,
                    in_ts,
                    in_coords,
                    targets,
                    target_mask,
                    valid_input,
                    history_index,
                    history_value,
                    user_id,
                )
            )

    def _append_block_samples(
        self, user_id, loc, cat, ts, coords, split_labels,
        split, sequence_length, stride,
    ):
        n = len(loc)
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
            for j, pos in enumerate(target_positions):
                if pos in seen_targets:
                    mask[j] = False
            if not mask.any():
                continue
            seen_targets.update(target_positions[mask].tolist())
            history_indices = []
            history_values = []
            for pos in target_positions:
                counts = np.bincount(loc[:pos], minlength=self.n_pois).astype(np.float32)
                index, value = self._history_prior(counts, output_steps=1)
                history_indices.append(index[0])
                history_values.append(value[0])
            pad = sequence_length - length
            in_loc = np.pad(loc[start:end], (0, pad), constant_values=0)
            in_cat = np.pad(cat[start:end], (0, pad), constant_values=0)
            in_ts = np.pad(ts[start:end], (0, pad), constant_values=0.0)
            in_coords = np.pad(coords[start:end], ((0, pad), (0, 0)), constant_values=0.0)
            targets = np.pad(loc[start + 1:end + 1], (0, pad), constant_values=0)
            target_mask = np.pad(mask.astype(np.bool_), (0, pad), constant_values=False)
            valid_input = np.zeros(sequence_length, dtype=np.bool_)
            valid_input[:length] = True
            history_index = np.pad(
                np.stack(history_indices),
                ((0, pad), (0, 0)),
                constant_values=0,
            )
            history_value = np.pad(
                np.stack(history_values),
                ((0, pad), (0, 0)),
                constant_values=0.0,
            )
            self.samples.append(
                (
                    in_loc,
                    in_cat,
                    in_ts,
                    in_coords,
                    targets,
                    target_mask,
                    valid_input,
                    history_index,
                    history_value,
                    user_id,
                )
            )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        split: str,
        sequence_length: int,
        stride: int,
        mode: str = "block_all",
        history_topk: int = 100,
    ):
        return cls(read_table(path), split, sequence_length, stride, mode, history_topk)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        (
            loc,
            cat,
            ts,
            coords,
            target,
            target_mask,
            valid_input,
            history_index,
            history_value,
            user,
        ) = self.samples[index]
        return {
            "locations": torch.from_numpy(loc),
            "category_ids": torch.from_numpy(cat),
            "timestamps": torch.from_numpy(ts),
            "coordinates": torch.from_numpy(coords),
            "targets": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask),
            "valid_input": torch.from_numpy(valid_input),
            "history_prior_index": torch.from_numpy(history_index),
            "history_prior_value": torch.from_numpy(history_value),
            "user_id": torch.tensor(user, dtype=torch.long),
        }
