from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashback.config import load_config
from flashback.data.preprocessing import prepare_dataset, PreparedPaths
from flashback.kge.triplets import generate_triplets
from flashback.kge.train import train_transe
from flashback.kge.construct_graphs import construct_graphs
from flashback.train import train_graph_flashback
from flashback.analysis.visualizations import create_visualizations


def _prepared(cfg):
    manifests = sorted(
        Path(cfg.data.output_dir).glob(f"{cfg.data.dataset}_*_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError("Prepared manifest not found; run stage prepare")
    manifest = json.loads(manifests[0].read_text())
    city = manifest["city"]
    base = Path(cfg.data.output_dir)
    friend = base / f"{cfg.data.dataset}_{city}_friendships.parquet"
    categories = base / f"{cfg.data.dataset}_{city}_categories.csv"
    return PreparedPaths(
        base / f"{cfg.data.dataset}_{city}_checkins.parquet",
        friend if friend.exists() else None,
        base / f"{cfg.data.dataset}_{city}_users.csv",
        base / f"{cfg.data.dataset}_{city}_pois.csv",
        base / f"checkins-{cfg.data.dataset}-{city}.txt",
        manifests[0],
        categories if categories.exists() else None,
    )


def run(cfg, stage):
    prepared = None
    if stage in {"prepare", "all"}:
        if not Path(cfg.data.raw_checkins).exists():
            raise FileNotFoundError(
                f"Austin input not found: {cfg.data.raw_checkins}. "
                "Attach/extract gowalla_austin_enriched_195k.zip first."
            )
        prepared = prepare_dataset(cfg)
    if stage in {"stkg", "kge", "graphs", "train", "analyze"}:
        prepared = _prepared(cfg)
    if stage in {"stkg", "all"}:
        generate_triplets(cfg, prepared.checkins, prepared.friendships)
    if stage in {"kge", "all"}:
        train_transe(cfg)
    if stage in {"graphs", "all"}:
        construct_graphs(cfg, prepared.checkins)
    if stage in {"train", "all"}:
        train_graph_flashback(cfg, prepared.checkins)
    if stage in {"analyze", "all"}:
        create_visualizations(cfg.artifacts_dir, cfg.data.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["prepare", "stkg", "kge", "graphs", "train", "analyze", "all"],
        default="all",
    )
    args = parser.parse_args()
    run(load_config(args.config), args.stage)


if __name__ == "__main__":
    main()
