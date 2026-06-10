from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from flashback.data.sequences import NextPoiSequenceDataset
from flashback.evaluation.metrics import RankingAccumulator
from flashback.utils import read_table


def evaluate_popularity(
    checkins_path,
    sequence_length=20,
    stride=20,
    batch_size=128,
    personal=False,
    mode="block_all",
):
    frame = read_table(checkins_path)
    train = frame[frame.split == "train"]
    n_pois = int(frame.poi_id.max()) + 1
    global_scores = torch.bincount(
        torch.tensor(train.poi_id.to_numpy()), minlength=n_pois
    ).float()
    personal_scores = {
        int(user): torch.bincount(
            torch.tensor(group.poi_id.to_numpy()), minlength=n_pois
        ).float()
        for user, group in train.groupby("user_id")
    }
    dataset = NextPoiSequenceDataset(
        frame, "test", sequence_length, stride, mode
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    accumulator = RankingAccumulator()
    for batch in loader:
        logits = torch.stack([
            personal_scores.get(int(user), global_scores) if personal else global_scores
            for user in batch["user_id"]
        ])
        logits = logits[:, None, :].expand(-1, batch["targets"].shape[1], -1)
        accumulator.update(logits, batch["targets"], batch["target_mask"], batch["user_id"])
    return accumulator.compute()
