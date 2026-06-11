from __future__ import annotations

import torch

from flashback.evaluation.metrics import RankingAccumulator, exact_ranks


@torch.no_grad()
def evaluate(model, loader, device, save_predictions=False):
    model.eval()
    accumulator = RankingAccumulator()
    records = []
    loss_sum = 0.0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits, _, _ = model(
            batch["locations"],
            batch["timestamps"],
            batch["coordinates"],
            batch["user_id"],
            batch["valid_input"],
            batch.get("category_ids"),
            batch.get("history_prior_index"),
            batch.get("history_prior_value"),
        )
        mask = batch["target_mask"]
        flat_logits = logits[mask]
        flat_targets = batch["targets"][mask]
        if flat_targets.numel():
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction="sum")
            loss_sum += float(loss)
            count += flat_targets.numel()
            accumulator.update(logits, batch["targets"], mask, batch["user_id"])
            if save_predictions:
                ranks = exact_ranks(logits, batch["targets"])
                for b, user in enumerate(batch["user_id"].tolist()):
                    for step in torch.where(mask[b])[0].tolist():
                        records.append({
                            "user_id": user,
                            "step": step,
                            "target_poi": int(batch["targets"][b, step]),
                            "rank": int(ranks[b, step]),
                            "top1": int(logits[b, step].argmax()),
                        })
    result = accumulator.compute()
    result["loss"] = loss_sum / max(1, count)
    return result, records
