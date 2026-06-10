from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from flashback.config import ExperimentConfig, resolve_device, seed_everything, dump_json
from flashback.kge.transe import TransE
from flashback.kge.triplets import VISITS, TEMPORAL, SPATIAL, FRIEND

RELATION_NAMES = {VISITS: "visits", TEMPORAL: "temporal", SPATIAL: "spatial", FRIEND: "friend"}


def _negative(batch: torch.Tensor, n_users: int, n_pois: int, p_head: float) -> torch.Tensor:
    neg = batch.clone()
    device = batch.device
    choose_head = torch.rand(len(batch), device=device) < p_head
    rel = batch[:, 1]
    for relation in (VISITS, TEMPORAL, SPATIAL, FRIEND):
        mask = rel == relation
        if not mask.any():
            continue
        head_mask = mask & choose_head
        tail_mask = mask & ~choose_head
        if relation == VISITS:
            if head_mask.any():
                neg[head_mask, 0] = torch.randint(0, n_users, (int(head_mask.sum()),), device=device)
            if tail_mask.any():
                neg[tail_mask, 2] = n_users + torch.randint(0, n_pois, (int(tail_mask.sum()),), device=device)
        elif relation in (TEMPORAL, SPATIAL):
            if head_mask.any():
                neg[head_mask, 0] = n_users + torch.randint(0, n_pois, (int(head_mask.sum()),), device=device)
            if tail_mask.any():
                neg[tail_mask, 2] = n_users + torch.randint(0, n_pois, (int(tail_mask.sum()),), device=device)
        else:
            if head_mask.any():
                neg[head_mask, 0] = torch.randint(0, n_users, (int(head_mask.sum()),), device=device)
            if tail_mask.any():
                neg[tail_mask, 2] = torch.randint(0, n_users, (int(tail_mask.sum()),), device=device)
    return neg


def _loader(cfg: ExperimentConfig, triplets: torch.Tensor) -> DataLoader:
    dataset = TensorDataset(triplets)
    if cfg.kge.sampling == "natural":
        return DataLoader(dataset, batch_size=cfg.kge.batch_size, shuffle=True)

    relations = triplets[:, 1].numpy()
    counts = {int(r): max(1, int((relations == r).sum())) for r in np.unique(relations)}
    target = {
        relation: float(cfg.kge.relation_weights.get(name, 0.0))
        for relation, name in RELATION_NAMES.items()
    }
    weights = np.asarray([
        target.get(int(relation), 0.0) / counts.get(int(relation), 1)
        for relation in relations
    ], dtype=np.float64)
    if not np.any(weights > 0):
        raise ValueError("relation_balanced sampling needs positive relation_weights")
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=len(triplets),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=cfg.kge.batch_size, sampler=sampler)


@torch.no_grad()
def _relation_diagnostics(model, validation, negative, margin):
    diagnostics = {}
    for relation, name in RELATION_NAMES.items():
        mask = validation[:, 1] == relation
        if not mask.any():
            continue
        positive_distance = model(validation[mask])
        negative_distance = model(negative[mask])
        diagnostics[name] = {
            "count": int(mask.sum()),
            "margin_loss": float(torch.relu(margin + positive_distance - negative_distance).mean()),
            "pairwise_accuracy": float((positive_distance < negative_distance).float().mean()),
            "positive_distance": float(positive_distance.mean()),
            "negative_distance": float(negative_distance.mean()),
        }
    return diagnostics


def train_transe(cfg: ExperimentConfig) -> Path:
    seed_everything(cfg.train.seed)
    output = Path(cfg.stkg.output_dir)
    metadata = json.loads((output / "stkg_manifest.json").read_text())
    train_triplets = torch.from_numpy(np.load(output / "triplets_train.npy")).long()
    validation_triplets = torch.from_numpy(np.load(output / "triplets_validation.npy")).long()
    device = resolve_device(cfg.kge.device)
    model = TransE(
        metadata["n_entities"], metadata["n_relations"],
        cfg.kge.embedding_dim, cfg.kge.p_norm,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.kge.learning_rate)
    loader = _loader(cfg, train_triplets)
    validation = validation_triplets.to(device)
    validation_negative = (
        _negative(validation, metadata["n_users"], metadata["n_pois"], cfg.kge.corrupt_head_probability)
        if len(validation) else None
    )
    best = float("inf")
    bad_epochs = 0
    history = []
    checkpoint = Path(cfg.kge.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.kge.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            positives = batch.repeat_interleave(max(1, cfg.kge.negative_samples), dim=0)
            negatives = _negative(
                positives, metadata["n_users"], metadata["n_pois"],
                cfg.kge.corrupt_head_probability,
            )
            loss = torch.relu(
                cfg.kge.margin + model(positives) - model(negatives)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.normalize_entities()
            total += float(loss.detach()) * len(batch)
            count += len(batch)

        model.eval()
        with torch.no_grad():
            if len(validation):
                validation_loss = float(torch.relu(
                    cfg.kge.margin + model(validation) - model(validation_negative)
                ).mean())
            else:
                validation_loss = total / max(1, count)
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, count),
            "validation_loss": validation_loss,
        }
        history.append(row)
        if validation_loss < best - 1e-6:
            best = validation_loss
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "meta": metadata,
                "dim": cfg.kge.embedding_dim,
                "p_norm": cfg.kge.p_norm,
                "epoch": epoch,
                "config": cfg.to_dict(),
            }, checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.kge.patience:
                break

    dump_json(history, output / "transe_history.json")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    diagnostics = {
        "best_epoch": saved["epoch"],
        "best_validation_loss": best,
        "sampling": cfg.kge.sampling,
        "negative_samples": cfg.kge.negative_samples,
        "relations": _relation_diagnostics(
            model, validation, validation_negative, cfg.kge.margin
        ) if len(validation) else {},
    }
    dump_json(diagnostics, output / "transe_diagnostics.json")
    return checkpoint
