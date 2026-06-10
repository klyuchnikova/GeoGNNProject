"""Install the portable Austin Gowalla package into ``data/input``.

Kaggle may expose an uploaded ZIP either as the original archive or as an
already-extracted dataset directory. This installer supports both layouts and
accepts compressed or plain CSV files.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

PACKAGE_NAME = "gowalla_austin_enriched_195k.zip"
CHECKINS_GZ = "gowalla_austin_checkins.csv.gz"
CHECKINS_CSV = "gowalla_austin_checkins.csv"
FRIENDS_GZ = "gowalla_austin_friendships.csv.gz"
FRIENDS_CSV = "gowalla_austin_friendships.csv"


def _search_roots() -> list[Path]:
    return [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.cwd(),
        Path.home() / "Downloads",
    ]


def _find_first(names: list[str], roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            found = next(root.rglob(name), None)
            if found is not None:
                return found
    return None


def _copy_or_gzip(source: Path, destination_gz: Path) -> None:
    destination_gz.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination_gz.resolve():
        return
    if source.name.endswith(".gz"):
        shutil.copy2(source, destination_gz)
        return
    with source.open("rb") as src, gzip.open(destination_gz, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def _install_from_zip(archive: Path, output: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        checkins_name = CHECKINS_GZ if CHECKINS_GZ in names else CHECKINS_CSV if CHECKINS_CSV in names else None
        if checkins_name is None:
            raise ValueError(f"{archive} contains neither {CHECKINS_GZ} nor {CHECKINS_CSV}")

        temp_dir = output / ".install_tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        try:
            zf.extract(checkins_name, temp_dir)
            _copy_or_gzip(temp_dir / checkins_name, output / CHECKINS_GZ)

            friends_name = FRIENDS_GZ if FRIENDS_GZ in names else FRIENDS_CSV if FRIENDS_CSV in names else None
            if friends_name:
                zf.extract(friends_name, temp_dir)
                _copy_or_gzip(temp_dir / friends_name, output / FRIENDS_GZ)

            if "manifest.json" in names:
                zf.extract("manifest.json", temp_dir)
                shutil.copy2(temp_dir / "manifest.json", output / "manifest.json")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def install(source: Path | None, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    roots = _search_roots()

    if source is not None:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file() and zipfile.is_zipfile(source):
            _install_from_zip(source, output)
        elif source.is_dir():
            roots = [source]
            checkins = _find_first([CHECKINS_GZ, CHECKINS_CSV], roots)
            if checkins is None:
                raise FileNotFoundError(f"No Austin check-in CSV found under {source}")
            _copy_or_gzip(checkins, output / CHECKINS_GZ)
            friends = _find_first([FRIENDS_GZ, FRIENDS_CSV], roots)
            if friends:
                _copy_or_gzip(friends, output / FRIENDS_GZ)
            manifest = _find_first(["manifest.json"], roots)
            if manifest and manifest.resolve() != (output / "manifest.json").resolve():
                shutil.copy2(manifest, output / "manifest.json")
        else:
            _copy_or_gzip(source, output / CHECKINS_GZ)
    else:
        archive = _find_first([PACKAGE_NAME], roots)
        if archive is not None:
            _install_from_zip(archive, output)
        else:
            checkins = _find_first([CHECKINS_GZ, CHECKINS_CSV], roots)
            if checkins is None:
                raise FileNotFoundError(
                    "Austin input was not found. Attach the Kaggle dataset containing "
                    f"{CHECKINS_CSV} or {CHECKINS_GZ}, or provide --source explicitly."
                )
            print(f"Found check-ins: {checkins}")
            _copy_or_gzip(checkins, output / CHECKINS_GZ)
            friends = _find_first([FRIENDS_GZ, FRIENDS_CSV], roots)
            if friends:
                print(f"Found optional friendships: {friends}")
                _copy_or_gzip(friends, output / FRIENDS_GZ)
            manifest = _find_first(["manifest.json"], roots)
            if manifest and manifest.resolve() != (output / "manifest.json").resolve():
                shutil.copy2(manifest, output / "manifest.json")

    checkins = output / CHECKINS_GZ
    required = {"raw_user_id", "timestamp", "latitude", "longitude", "raw_poi_id", "category"}
    columns = set(pd.read_csv(checkins, nrows=5).columns)
    missing = required - columns
    if missing:
        raise ValueError(f"Austin check-in file misses columns: {sorted(missing)}")

    friends = output / FRIENDS_GZ
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
    parser.add_argument("--source", type=Path, default=None, help="ZIP, extracted dataset directory, or check-in CSV")
    parser.add_argument("--archive", type=Path, default=None, help="Deprecated alias for --source")
    parser.add_argument("--output", type=Path, default=Path("data/input"))
    args = parser.parse_args()
    install(args.source or args.archive, args.output)


if __name__ == "__main__":
    main()
