from __future__ import annotations
import argparse,json
from pathlib import Path
from flashback.config import load_config
from flashback.data.downloads import download_gowalla
from flashback.data.preprocessing import prepare_dataset,PreparedPaths
from flashback.kge.triplets import generate_triplets
from flashback.kge.train import train_transe
from flashback.kge.construct_graphs import construct_graphs
from flashback.train import train_graph_flashback
from flashback.analysis.visualizations import create_visualizations


def _prepared(cfg):
    manifests=sorted(Path(cfg.data.output_dir).glob(f"{cfg.data.dataset}_*_manifest.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    if not manifests: raise FileNotFoundError("Prepared manifest not found; run stage prepare")
    m=json.loads(manifests[0].read_text()); city=m["city"]; base=Path(cfg.data.output_dir)
    friend=base/f"{cfg.data.dataset}_{city}_friendships.parquet"
    return PreparedPaths(base/f"{cfg.data.dataset}_{city}_checkins.parquet",friend if friend.exists() else None,
        base/f"{cfg.data.dataset}_{city}_users.csv",base/f"{cfg.data.dataset}_{city}_pois.csv",
        base/f"checkins-{cfg.data.dataset}-{city}.txt",manifests[0])

def run(cfg,stage):
    if stage in {"download","all"}:
        need_download = stage == "download" or not Path(cfg.data.raw_checkins).exists()
        if cfg.data.dataset=="gowalla" and need_download:
            paths=download_gowalla("data/raw",with_metadata=True);cfg.data.raw_checkins=paths["checkins"];cfg.data.raw_friendships=paths["friendships"];cfg.data.raw_metadata=paths.get("metadata")
        elif cfg.data.dataset != "gowalla" and stage=="download": raise RuntimeError("Use scripts/download_foursquare.py or mount a Kaggle dataset, then set data.raw_checkins")
    prepared=None
    if stage in {"prepare","all"}: prepared=prepare_dataset(cfg)
    if stage in {"stkg","kge","graphs","train","evaluate","analyze"}: prepared=_prepared(cfg)
    if stage in {"stkg","all"}: generate_triplets(cfg,prepared.checkins,prepared.friendships)
    if stage in {"kge","all"}: train_transe(cfg)
    if stage in {"graphs","all"}: construct_graphs(cfg,prepared.checkins)
    if stage in {"train","all"}: train_graph_flashback(cfg,prepared.checkins)
    if stage in {"analyze","all"}: create_visualizations(cfg.artifacts_dir,cfg.data.output_dir)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--stage",choices=["download","prepare","stkg","kge","graphs","train","analyze","all"],default="all");a=p.parse_args();run(load_config(a.config),a.stage)
if __name__=="__main__":main()
