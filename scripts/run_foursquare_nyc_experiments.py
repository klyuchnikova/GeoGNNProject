from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="report", choices=["report", "best"])
    parser.add_argument("--force-assets", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    cmd = [
        sys.executable, "scripts/run_experiment_suite.py",
        "--suite", "experiments/foursquare_nyc_suite.yaml",
        "--profile", args.profile,
        "--install-cmd", "scripts/download_foursquare_nyc.py",
    ]
    if args.force_assets:
        cmd.append("--force-assets")
    if args.force_train:
        cmd.append("--force-train")
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
