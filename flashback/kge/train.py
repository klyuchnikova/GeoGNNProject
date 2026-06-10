from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from flashback.config import ExperimentConfig, resolve_device, seed_everything, dump_json
from flashback.kge.transe import TransE
from flashback.kge.triplets import VISITS, TEMPORAL, SPATIAL, FRIEND


def _negative(batch: torch.Tensor, n_users: int, n_pois: int, p_head: float) -> torch.Tensor:
    neg = batch.clone(); device = batch.device
    choose_head = torch.rand(len(batch), device=device) < p_head
    rel = batch[:,1]
    # Relation-aware domains preserve entity types.
    for r in (VISITS, TEMPORAL, SPATIAL, FRIEND):
        mask = rel == r
        if not mask.any(): continue
        hm = mask & choose_head; tm = mask & ~choose_head
        if r == VISITS:
            if hm.any(): neg[hm,0] = torch.randint(0, n_users, (int(hm.sum()),), device=device)
            if tm.any(): neg[tm,2] = n_users + torch.randint(0, n_pois, (int(tm.sum()),), device=device)
        elif r in (TEMPORAL, SPATIAL):
            if hm.any(): neg[hm,0] = n_users + torch.randint(0, n_pois, (int(hm.sum()),), device=device)
            if tm.any(): neg[tm,2] = n_users + torch.randint(0, n_pois, (int(tm.sum()),), device=device)
        else:
            if hm.any(): neg[hm,0] = torch.randint(0, n_users, (int(hm.sum()),), device=device)
            if tm.any(): neg[tm,2] = torch.randint(0, n_users, (int(tm.sum()),), device=device)
    return neg


def train_transe(cfg: ExperimentConfig) -> Path:
    seed_everything(cfg.train.seed); out = Path(cfg.stkg.output_dir)
    meta = json.loads((out/"stkg_manifest.json").read_text())
    tr = torch.from_numpy(np.load(out/"triplets_train.npy")).long()
    va = torch.from_numpy(np.load(out/"triplets_validation.npy")).long()
    device = resolve_device(cfg.kge.device)
    model = TransE(meta["n_entities"], meta["n_relations"], cfg.kge.embedding_dim, cfg.kge.p_norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.kge.learning_rate)
    loader = DataLoader(TensorDataset(tr), batch_size=cfg.kge.batch_size, shuffle=True)
    va_device = va.to(device)
    va_negative = _negative(va_device, meta["n_users"], meta["n_pois"], cfg.kge.corrupt_head_probability) if len(va) else None
    best, bad, history = float("inf"), 0, []
    ckpt = Path(cfg.kge.checkpoint); ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.kge.epochs+1):
        model.train(); total=n=0
        for (batch,) in loader:
            batch=batch.to(device)
            positives=batch.repeat_interleave(max(1,cfg.kge.negative_samples),dim=0)
            neg=_negative(positives, meta["n_users"], meta["n_pois"], cfg.kge.corrupt_head_probability)
            loss=torch.relu(cfg.kge.margin + model(positives) - model(neg)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); model.normalize_entities()
            total += float(loss.detach())*len(batch); n += len(batch)
        model.eval()
        with torch.no_grad():
            if len(va):
                vb=va_device; vn=va_negative
                val=float(torch.relu(cfg.kge.margin + model(vb)-model(vn)).mean())
            else: val=total/max(1,n)
        row={"epoch":epoch,"train_loss":total/max(1,n),"validation_loss":val}; history.append(row)
        if val < best-1e-6:
            best=val; bad=0
            torch.save({"state_dict":model.state_dict(),"meta":meta,"dim":cfg.kge.embedding_dim,"p_norm":cfg.kge.p_norm,"epoch":epoch,"config":cfg.to_dict()}, ckpt)
        else:
            bad += 1
            if bad >= cfg.kge.patience: break
    dump_json(history, out/"transe_history.json")
    return ckpt
