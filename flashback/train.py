from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from flashback.config import dump_json, load_config, resolve_device, seed_everything
from flashback.data.sequences import NextPoiSequenceDataset
from flashback.evaluation.evaluator import evaluate
from flashback.model.graph_flashback import load_model
from flashback.popularity_baseline import evaluate_popularity
from flashback.utils import setup_logging, write_table


def _loader(path, split, cfg, shuffle=False):
    dataset = NextPoiSequenceDataset.from_parquet(
        path, split, cfg.data.sequence_length, cfg.data.sequence_stride
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
    )


def _is_better(value: float, best: float, mode: str) -> bool:
    return value > best + 1e-12 if mode == "max" else value < best - 1e-12


def train_graph_flashback(cfg, checkins_path):
    seed_everything(cfg.train.seed)
    logger = setup_logging()
    device = resolve_device(cfg.train.device)
    model = load_model(cfg, checkins_path).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(cfg.train.scheduler_milestones),
        gamma=cfg.train.scheduler_gamma,
    )

    train_loader = _loader(checkins_path, "train", cfg, shuffle=True)
    validation_loader = (
        _loader(checkins_path, "validation", cfg) if cfg.data.val_ratio > 0 else None
    )

    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{cfg.train.run_name}_best.pt"
    history: list[dict] = []
    best = float("-inf") if cfg.train.checkpoint_mode == "max" else float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0
        target_count = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            mask = batch["target_mask"]
            logits, _, _ = model(
                batch["locations"],
                batch["timestamps"],
                batch["coordinates"],
                batch["user_id"],
                batch["valid_input"],
            )
            loss = torch.nn.functional.cross_entropy(
                logits[mask], batch["targets"][mask]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * int(mask.sum())
            target_count += int(mask.sum())

        train_loss = total_loss / max(1, target_count)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        validation = None
        if validation_loader is not None:
            validation, _ = evaluate(model, validation_loader, device)
            row.update({f"validation_{key}": value for key, value in validation.items()})

        metric_name = cfg.train.checkpoint_metric
        if metric_name == "final_epoch":
            metric_value = float(epoch)
        elif metric_name == "train_loss":
            metric_value = train_loss
        else:
            if validation is None or metric_name not in validation:
                raise KeyError(
                    f"Checkpoint metric {metric_name!r} is unavailable. "
                    "Use train_loss/final_epoch for an 80/20 paper run."
                )
            metric_value = float(validation[metric_name])

        history.append(row)
        improved = _is_better(metric_value, best, cfg.train.checkpoint_mode)
        if improved or metric_name == "final_epoch":
            best = metric_value
            bad_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "epoch": epoch,
                    "checkpoint_metric": metric_name,
                    "checkpoint_value": metric_value,
                },
                checkpoint,
            )
        else:
            bad_epochs += 1

        logger.info(
            "epoch=%d train_loss=%.5f validation_MRR=%s",
            epoch,
            train_loss,
            "-" if validation is None else f"{validation['MRR']:.5f}",
        )
        scheduler.step()
        if validation_loader is not None and bad_epochs >= cfg.train.patience:
            logger.info("Early stopping after %d epochs without improvement", bad_epochs)
            break

    result_dir = Path(cfg.artifacts_dir) / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        result_dir / f"{cfg.train.run_name}_history.csv", index=False
    )

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    results = {
        "best_epoch": saved["epoch"],
        "checkpoint_metric": saved["checkpoint_metric"],
        "checkpoint_value": saved["checkpoint_value"],
    }
    if validation_loader is not None:
        results["validation"] = evaluate(model, validation_loader, device)[0]

    if cfg.train.evaluate_test_after_training:
        test_loader = _loader(checkins_path, "test", cfg)
        test, predictions = evaluate(model, test_loader, device, save_predictions=True)
        results["test"] = test
        prediction_dir = Path(cfg.artifacts_dir) / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        write_table(
            pd.DataFrame(predictions),
            prediction_dir / f"{cfg.train.run_name}_test.parquet",
            index=False,
        )

    dump_json(results, result_dir / f"{cfg.train.run_name}_metrics.json")
    for personal, name in [(False, "global_popularity"), (True, "personal_popularity")]:
        baseline = evaluate_popularity(
            checkins_path,
            cfg.data.sequence_length,
            cfg.data.sequence_stride,
            cfg.train.batch_size,
            personal,
        )
        dump_json({"test": baseline}, result_dir / f"{name}_metrics.json")
    return checkpoint, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkins", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkins = args.checkins or next(
        Path(cfg.data.output_dir).glob(f"{cfg.data.dataset}_*_checkins.parquet")
    )
    train_graph_flashback(cfg, checkins)


if __name__ == "__main__":
    main()
