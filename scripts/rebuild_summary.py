from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["Acc@1", "Acc@5", "Acc@10", "MAP@5", "MAP@10", "MRR"]


def read_suites() -> tuple[dict, dict]:
    labels, families = {}, {}
    for suite_path in (ROOT / "experiments").glob("*.yaml"):
        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
        for name, meta in (suite.get("experiments") or {}).items():
            labels[name] = meta.get("label", name)
            families[name] = meta.get("family", "")
    return labels, families


def main() -> None:
    labels, families = read_suites()
    rows = []
    for metrics_path in sorted((ROOT / "runs" / "experiments").glob("*/artifacts/results/*_metrics.json")):
        exp = metrics_path.parents[2].name
        if exp in {"global_popularity", "personal_popularity"}:
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = data.get("test", {})
        row = {
            "experiment": exp,
            "label": labels.get(exp, exp),
            "family": families.get(exp, ""),
            "source": "experiment",
            "best_epoch": data.get("best_epoch"),
        }
        row.update({m: test.get(m) for m in METRICS})
        rows.append(row)

        result_dir = metrics_path.parent
        for baseline, label in [("global_popularity", "Global popularity"), ("personal_popularity", "Personal popularity")]:
            p = result_dir / f"{baseline}_metrics.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8")).get("test", {})
                b = {"experiment": f"{baseline}_{exp}", "label": f"{label} ({exp})", "family": "baseline", "source": "baseline", "best_epoch": None}
                b.update({m: d.get(m) for m in METRICS})
                rows.append(b)

    summary = ROOT / "runs" / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No experiment metrics found under runs/experiments")
    df = df.sort_values("MRR", ascending=False)
    df.to_csv(summary / "all_experiments_rebuilt.csv", index=False)
    df[df["family"].eq("filtered_protocol")].to_csv(summary / "gowalla_filtered_results.csv", index=False)
    df[df["family"].eq("common_protocol")].to_csv(summary / "gowalla_common_results.csv", index=False)
    df[df["family"].eq("foursquare_nyc")].to_csv(summary / "foursquare_nyc_results.csv", index=False)
    df[df["family"].eq("baseline_model")].to_csv(summary / "plain_baselines.csv", index=False)
    df[df["family"].eq("original_like")].to_csv(summary / "original_like_results.csv", index=False)
    print(df[["experiment", "label"] + METRICS].to_string(index=False))


if __name__ == "__main__":
    main()
