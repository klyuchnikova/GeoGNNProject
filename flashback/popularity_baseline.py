from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from flashback.data.sequences import NextPoiSequenceDataset
from flashback.evaluation.metrics import RankingAccumulator
from flashback.utils import read_table


def evaluate_popularity(checkins_path, sequence_length=20, stride=20, batch_size=128, personal=False):
    frame=read_table(checkins_path); train=frame[frame.split=="train"]
    n_pois=int(frame.poi_id.max())+1
    global_scores=torch.bincount(torch.tensor(train.poi_id.to_numpy()),minlength=n_pois).float()
    personal_scores={int(u):torch.bincount(torch.tensor(g.poi_id.to_numpy()),minlength=n_pois).float() for u,g in train.groupby("user_id")}
    ds=NextPoiSequenceDataset(frame,"test",sequence_length,stride); loader=DataLoader(ds,batch_size=batch_size)
    acc=RankingAccumulator()
    for b in loader:
        logits=torch.stack([personal_scores.get(int(u),global_scores) if personal else global_scores for u in b["user_id"]])
        logits=logits[:,None,:].expand(-1,sequence_length,-1)
        acc.update(logits,b["targets"],b["target_mask"],b["user_id"])
    return acc.compute()
