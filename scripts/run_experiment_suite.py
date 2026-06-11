from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashback.config import load_config

METRICS = ["Acc@1", "Acc@5", "Acc@10", "MAP@5", "MAP@10", "MRR"]


def deep_merge(base, update):
    result = deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def run_cmd(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True, env=env)


def ready(cfg, stage):
    if stage == "prepare":
        return any(Path(cfg.data.output_dir).glob(f"{cfg.data.dataset}_*_manifest.json"))
    if stage == "stkg":
        return (Path(cfg.stkg.output_dir) / "stkg_manifest.json").exists() and (Path(cfg.stkg.output_dir) / "triplets_train.npy").exists()
    if stage == "kge":
        return Path(cfg.kge.checkpoint).exists() and (Path(cfg.stkg.output_dir) / "transe_diagnostics.json").exists()
    if stage == "graphs":
        return (Path(cfg.graphs.output_dir) / "graph_manifest.json").exists()
    return False


def group_config(suite, group_name):
    group = suite["groups"][group_name]
    raw = yaml.safe_load((ROOT / group["base_config"]).read_text(encoding="utf-8"))
    return deep_merge(raw, group.get("overrides", {}))


def write_group_config(suite, group_name):
    raw = group_config(suite, group_name)
    out = ROOT / "runs" / "generated_configs" / f"_shared_{group_name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def generate_config(name, suite, group_name, overrides):
    raw = group_config(suite, group_name)
    raw = deep_merge(raw, overrides)
    raw.setdefault("train", {})["run_name"] = name
    raw["train"]["checkpoint_dir"] = f"runs/experiments/{name}/checkpoints"
    raw["artifacts_dir"] = f"runs/experiments/{name}/artifacts"
    out = ROOT / "runs" / "generated_configs" / f"{name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def collect(suite, selected, status):
    rows = []
    for name in selected:
        meta = suite["experiments"][name]
        p = ROOT / "runs" / "experiments" / name / "artifacts" / "results" / f"{name}_metrics.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        row = {
            "experiment": name,
            "label": meta.get("label", name),
            "source": "experiment",
            "family": meta.get("family", ""),
            "best_epoch": data.get("best_epoch"),
        }
        row.update({m: data.get("test", {}).get(m) for m in METRICS})
        for macro_name in ["macro_user_Acc@1", "macro_user_Acc@5", "macro_user_Acc@10", "macro_user_MAP@5", "macro_user_MAP@10", "macro_user_MRR"]:
            row[macro_name] = data.get("test", {}).get(macro_name)
        rows.append(row)

    for name in selected:
        result_dir = ROOT / "runs" / "experiments" / name / "artifacts" / "results"
        if not result_dir.exists():
            continue
        for baseline, label in [("global_popularity", "Global popularity"), ("personal_popularity", "Personal popularity")]:
            p = result_dir / f"{baseline}_metrics.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8")).get("test", {})
                row = {"experiment": baseline, "label": label, "source": "baseline", "family": "baseline", "best_epoch": None}
                row.update({m: d.get(m) for m in METRICS})
                rows.append(row)
        break

    for label, vals in suite.get("published_reference", {}).items():
        row = {
            "experiment": label.lower().replace(" ", "_").replace("/", "_"),
            "label": label,
            "source": "published_reference",
            "family": "published_reference",
            "best_epoch": None,
        }
        row.update({m: vals.get(m) for m in METRICS})
        rows.append(row)

    summary = ROOT / "runs" / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(summary / "experiment_metrics.csv", index=False)
    (summary / "experiment_metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (summary / "run_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    if not frame.empty:
        frame[frame["family"].isin(["common_protocol", "baseline", "published_reference"])].to_csv(summary / "common_protocol_results.csv", index=False)
        frame[frame["family"].eq("filtered_protocol")].to_csv(summary / "filtered_protocol_results.csv", index=False)
        frame[frame["family"].eq("foursquare_nyc")].to_csv(summary / "foursquare_nyc_results.csv", index=False)

    diag_rows = []
    for diag_path in sorted((ROOT / "runs" / "shared").glob("*/graphs/graph_neighbor_diagnostics.json")):
        try:
            d = json.loads(diag_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        group = diag_path.parts[-3]
        row = {"group": group, "graph_dir": d.get("graph_dir", ""), "transition_nnz": d.get("transition_nnz"), "preference_nnz": d.get("preference_nnz")}
        for prefix, key in [("transition", "transition_true_next_neighbor"), ("preference", "preference_true_target_neighbor")]:
            for metric, value in (d.get(key) or {}).items():
                row[f"{prefix}_{metric}"] = value
        diag_rows.append(row)
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(summary / "graph_diagnostics_summary.csv", index=False)

    return frame


def run_suite(suite_path: Path, profile: str, install_cmd: list[str] | None, force_assets: bool, force_train: bool):
    suite = yaml.safe_load((ROOT / suite_path).read_text(encoding="utf-8"))
    selected = suite["profiles"][profile]
    status = {"suite": str(suite_path), "profile": profile, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments": {}}

    if install_cmd:
        run_cmd([sys.executable] + install_cmd)

    groups = []
    for name in selected:
        group = suite["experiments"][name]["group"]
        if group not in groups:
            groups.append(group)

    for group in groups:
        group_path = write_group_config(suite, group)
        cfg = load_config(group_path)
        for stage in ["prepare", "stkg", "kge", "graphs"]:
            if force_assets or not ready(cfg, stage):
                run_cmd([sys.executable, "-m", "flashback.pipeline", "--config", group_path.relative_to(ROOT), "--stage", stage])
            else:
                print(f"[resume] {group}: {stage} already complete")
        diag = Path(cfg.graphs.output_dir) / "graph_neighbor_diagnostics.json"
        if force_assets or not diag.exists():
            run_cmd([sys.executable, "scripts/graph_diagnostics.py", "--config", group_path.relative_to(ROOT)])

    for name in selected:
        meta = suite["experiments"][name]
        generated = generate_config(name, suite, meta["group"], meta.get("overrides", {}))
        metrics = ROOT / "runs" / "experiments" / name / "artifacts" / "results" / f"{name}_metrics.json"
        started = time.time()
        try:
            if force_train or not metrics.exists():
                run_cmd([sys.executable, "-m", "flashback.pipeline", "--config", generated.relative_to(ROOT), "--stage", "train"])
                run_cmd([sys.executable, "-m", "flashback.pipeline", "--config", generated.relative_to(ROOT), "--stage", "analyze"])
            else:
                print(f"[resume] {name}: metrics already exist")
            status["experiments"][name] = {"status": "complete", "seconds": round(time.time() - started, 2)}
        except Exception as exc:
            status["experiments"][name] = {"status": "failed", "error": repr(exc), "seconds": round(time.time() - started, 2)}
            collect(suite, selected, status)
            raise
        collect(suite, selected, status)

    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    frame = collect(suite, selected, status)
    print("\nFinal summary:")
    if not frame.empty:
        print(frame[["label"] + METRICS].to_string(index=False))
    print("\nPackage results with: python scripts/package_results.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--profile", default="report")
    parser.add_argument("--install-cmd", nargs="*", default=None)
    parser.add_argument("--force-assets", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    run_suite(args.suite, args.profile, args.install_cmd, args.force_assets, args.force_train)


if __name__ == "__main__":
    main()
