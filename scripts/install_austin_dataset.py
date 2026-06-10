"""Install the portable Austin Gowalla package into ``data/input``.

The tuned experiment requires only ``gowalla_austin_checkins.csv.gz``. The
friendship file is copied when present because the faithful STKG configuration
can use it, but it is optional for the recommended run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

PACKAGE_NAME = "gowalla_austin_enriched_195k.zip"
CHECKINS_NAME = "gowalla_austin_checkins.csv.gz"
FRIENDS_NAME = "gowalla_austin_friendships.csv.gz"


def _search_roots() -> list[Path]:
    return [Path.cwd(), Path("/kaggle/input"), Path("/kaggle/working"), Path.home() / "Downloads"]


def _find_archive() -> Path | None:
    for root in _search_roots():
        if root.exists():
            found = next(root.rglob(PACKAGE_NAME), None)
            if found:
                return found
    return None


def install(archive: Path | None, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(archive) if archive is not None else _find_archive()

    if archive is not None:
        if not archive.exists():
            raise FileNotFoundError(archive)
        with zipfile.ZipFile(archive) as zf:
            if CHECKINS_NAME not in zf.namelist():
                raise ValueError(f"{archive} does not contain {CHECKINS_NAME}")
            for name in [CHECKINS_NAME, FRIENDS_NAME, "manifest.json"]:
                if name in zf.namelist():
                    with zf.open(name) as source, (output / name).open("wb") as destination:
                        shutil.copyfileobj(source, destination)
    else:
        found: dict[str, Path] = {}
        for root in _search_roots():
            if not root.exists():
                continue
            for name in [CHECKINS_NAME, FRIENDS_NAME, "manifest.json"]:
                path = next(root.rglob(name), None)
                if path:
                    found[name] = path
        if CHECKINS_NAME not in found:
            raise FileNotFoundError(
                f"Attach {PACKAGE_NAME} to Kaggle or provide --archive explicitly."
            )
        for name, source in found.items():
            shutil.copy2(source, output / name)

    checkins = output / CHECKINS_NAME
    required = {"raw_user_id", "timestamp", "latitude", "longitude", "raw_poi_id", "category"}
    columns = set(pd.read_csv(checkins, nrows=5).columns)
    missing = required - columns
    if missing:
        raise ValueError(f"Austin check-in file misses columns: {sorted(missing)}")

    friends = output / FRIENDS_NAME
    if friends.exists():
        friend_columns = set(pd.read_csv(friends, nrows=5).columns)
        if not {"raw_user_a", "raw_user_b"}.issubset(friend_columns):
            raise ValueError("Friendship file must contain raw_user_a and raw_user_b")

    manifest = output / "manifest.json"
    if manifest.exists():
        print(json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    print(f"Installed check-ins: {checkins}")
    print(f"Optional friendships: {friends if friends.exists() else 'not installed'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/input"))
    args = parser.parse_args()
    install(args.archive, args.output)


if __name__ == "__main__":
    main()
