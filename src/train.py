import argparse
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from data import (
    FoursquareReader,
    GowallaReader,
    PoiDataset,
    SequenceBuilder,
    TemporalSplitter,
    build_poi_graphs,
    collate_fn,
    filter_pipeline,
)
from data.filters import FilterConfig
from evaluate import Evaluator, print_metrics
from models.gugen import GuGen, GuGenConfig
from models.gugen_graph import GuGenGraph, GuGenGraphConfig
from models.lstm import LstmConfig, LstmNextPOI
from utils import (
    TrainConfig,
    output_paths,
    remap_ids,
    resolve_device,
    save_checkpoint,
    save_results,
    set_seed,
    setup_logging,
    train_epoch,
    validate_epoch,
)


def load_dataframe(cfg: TrainConfig):
    if cfg.dataset == "foursquare":
        reader = FoursquareReader(cfg.data_root, city=cfg.city)
    elif cfg.dataset == "gowalla":
        reader = GowallaReader(cfg.data_root, city=cfg.city)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    return reader.load()


def build_dataloaders(cfg: TrainConfig, filter_cfg: FilterConfig, logger):
    df = load_dataframe(cfg)
    raw_stats = {
        "raw_rows": len(df),
        "raw_users": df["user_id"].nunique(),
        "raw_pois": df["poi_id"].nunique(),
    }
    logger.info("Loaded raw data: %s", raw_stats)

    df = filter_pipeline(df, filter_cfg)
    df, mappings = remap_ids(df)
    filtered_stats = {
        "filtered_rows": len(df),
        "filtered_users": df["user_id"].nunique(),
        "filtered_pois": df["poi_id"].nunique(),
        "filtered_categories": int(df["category"].nunique()),
    }
    logger.info("After filtering: %s", filtered_stats)

    splitter = TemporalSplitter()
    train_df, val_df, test_df = splitter.split(df)

    builder = SequenceBuilder(seq_len=cfg.seq_len, stride=cfg.stride)
    train_samples = builder.build(train_df)
    val_samples = builder.build(val_df)
    test_samples = builder.build(test_df)

    if not train_samples:
        raise ValueError(
            "No training sequences were built. Try relaxing filters or reducing seq_len."
        )

    sample_stats = {
        "train_sequences": len(train_samples),
        "val_sequences": len(val_samples),
        "test_sequences": len(test_samples),
    }
    logger.info("Built sequences: %s", sample_stats)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "collate_fn": collate_fn,
        "num_workers": cfg.num_workers,
    }
    train_loader = DataLoader(PoiDataset(train_samples), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(PoiDataset(val_samples), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(PoiDataset(test_samples), shuffle=False, **loader_kwargs)

    vocab = {
        "num_users": int(df["user_id"].max()) + 1,
        "num_pois": int(df["poi_id"].max()) + 1,
        "num_categories": int(df["category"].max()) + 1,
    }
    data_stats = {**raw_stats, **filtered_stats, **sample_stats, **vocab, "mappings": mappings}
    return train_loader, val_loader, test_loader, vocab, data_stats, train_df


def build_model(cfg: TrainConfig, vocab: dict, graphs=None):
    if cfg.model == "gugen":
        model_cfg = GuGenConfig(
            num_pois=vocab["num_pois"],
            num_users=vocab["num_users"],
            num_categories=vocab["num_categories"],
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
        )
        return GuGen(model_cfg)

    if cfg.model == "lstm":
        model_cfg = LstmConfig(
            num_pois=vocab["num_pois"],
            num_users=vocab["num_users"],
            num_categories=vocab["num_categories"],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        return LstmNextPOI(model_cfg)

    if cfg.model == "gugen_graph":
        if graphs is None:
            raise ValueError("gugen_graph requires POI graphs built from the training split")
        model_cfg = GuGenGraphConfig(
            num_pois=vocab["num_pois"],
            num_users=vocab["num_users"],
            num_categories=vocab["num_categories"],
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            gcn_layers=cfg.gcn_layers,
            dropout=cfg.dropout,
        )
        return GuGenGraph(model_cfg, graphs)

    raise ValueError(f"Unsupported model: {cfg.model}")


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    checkpoint_dir, results_dir, log_dir = output_paths(cfg)
    logger = setup_logging(log_dir / "train.log")
    logger.info("Using device: %s", device)

    filter_cfg = FilterConfig(
        min_user_visits=cfg.min_user_visits,
        min_poi_visits=cfg.min_poi_visits,
        use_kcore=cfg.use_kcore,
        user_k=cfg.user_k,
        poi_k=cfg.poi_k,
        min_entropy=cfg.min_entropy,
        max_dominant_ratio=cfg.max_dominant_ratio,
    )

    logger.info("Starting training with config: %s", asdict(cfg))
    train_loader, val_loader, test_loader, vocab, data_stats, train_df = build_dataloaders(
        cfg, filter_cfg, logger
    )

    graphs = None
    if cfg.model == "gugen_graph":
        graphs = build_poi_graphs(
            train_df,
            num_pois=vocab["num_pois"],
            num_categories=vocab["num_categories"],
            geo_dist_km=cfg.geo_dist_km,
            max_geo_neighbors=cfg.max_geo_neighbors,
        )
        graph_stats = {
            "transition_edges": int(graphs.transition_edge_index.size(1)),
            "geo_edges": int(graphs.geo_edge_index.size(1)),
            "geo_dist_km": cfg.geo_dist_km,
            "gcn_layers": cfg.gcn_layers,
        }
        data_stats["graph_stats"] = graph_stats
        logger.info("Built POI graphs from training split: %s", graph_stats)

    model = build_model(cfg, vocab, graphs=graphs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * cfg.lr_scheduler_factor,
    )

    best_val_acc1 = -1.0
    best_epoch = -1
    best_val_metrics = {}
    best_test_metrics = {}
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_epoch(model, train_loader, optimizer, device, epoch, logger)
        val_metrics = validate_epoch(model, val_loader, device, epoch, logger, split="val")
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d - Learning rate: %.6f", epoch, current_lr)

        if val_metrics["acc1"] > best_val_acc1:
            best_val_acc1 = val_metrics["acc1"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_test_metrics = validate_epoch(
                model, test_loader, device, epoch, logger, split="test"
            )
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val": val_metrics, "test": best_test_metrics},
                cfg,
                checkpoint_dir / "best.pt",
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.lr_scheduler_patience:
                break

    if best_epoch < 0:
        best_test_metrics = validate_epoch(
            model, test_loader, device, cfg.epochs, logger, split="test"
        )
        best_val_metrics = validate_epoch(
            model, val_loader, device, cfg.epochs, logger, split="val"
        )
        best_epoch = cfg.epochs
        save_checkpoint(
            model,
            optimizer,
            best_epoch,
            {"val": best_val_metrics, "test": best_test_metrics},
            cfg,
            checkpoint_dir / "best.pt",
        )

    train_evaluator = Evaluator(model, device)
    final_train_metrics = train_evaluator.evaluate(train_loader)
    logger.info("Final train metrics: %s", final_train_metrics)

    save_results(
        cfg,
        filter_cfg,
        data_stats,
        final_train_metrics,
        best_val_metrics,
        best_test_metrics,
        best_epoch,
        results_dir / "metrics.json",
    )
    save_checkpoint(
        model,
        optimizer,
        cfg.epochs,
        {"val": best_val_metrics, "test": best_test_metrics},
        cfg,
        checkpoint_dir / "last.pt",
    )

    print("\nBest results")
    print(f"Best epoch: {best_epoch}")
    print_metrics(best_test_metrics)
    logger.info("Training finished. Best epoch=%d", best_epoch)
    return best_test_metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train POI recommendation models")
    parser.add_argument("--model", default="gugen", choices=["gugen", "gugen_graph", "lstm"])
    parser.add_argument("--dataset", default="foursquare", choices=["foursquare", "gowalla"])
    parser.add_argument("--city", default="NYC")
    parser.add_argument("--data-root", default="../input")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gcn-layers", type=int, default=2)
    parser.add_argument("--geo-dist-km", type=float, default=0.5)
    parser.add_argument("--max-geo-neighbors", type=int, default=10)
    parser.add_argument("--min-user-visits", type=int, default=10)
    parser.add_argument("--min-poi-visits", type=int, default=10)
    parser.add_argument("--no-kcore", action="store_true")
    parser.add_argument("--user-k", type=int, default=10)
    parser.add_argument("--poi-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--exp-name", default=None)
    args = parser.parse_args()

    device = str(resolve_device(args.device))
    return TrainConfig(
        model=args.model,
        dataset=args.dataset,
        city=args.city,
        data_root=args.data_root,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        gcn_layers=args.gcn_layers,
        geo_dist_km=args.geo_dist_km,
        max_geo_neighbors=args.max_geo_neighbors,
        min_user_visits=args.min_user_visits,
        min_poi_visits=args.min_poi_visits,
        use_kcore=not args.no_kcore,
        user_k=args.user_k,
        poi_k=args.poi_k,
        device=device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        log_dir=args.log_dir,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        exp_name=args.exp_name,
        weight_decay=args.weight_decay,
    )


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
