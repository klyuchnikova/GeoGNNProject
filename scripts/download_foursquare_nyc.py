"""Install the public Foursquare NYC/Tokyo check-in dataset for Kaggle runs.

The script first looks for an attached Kaggle input containing the standard
``dataset_TSMC2014_NYC.txt`` file. If not found, it downloads the public Kaggle
archive ``chetanism/foursquare-nyc-and-tokyo-checkin-dataset`` and extracts the
NYC check-ins into ``data/input/foursquare_nyc_checkins.txt``.
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

KAGGLE_DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/chetanism/foursquare-nyc-and-tokyo-checkin-dataset"
NYC_NAMES = [
    "dataset_TSMC2014_NYC.txt",
    "dataset_tsmc2014_nyc.txt",
    "dataset_TSMC2014_NYC.csv",
    "dataset_tsmc2014_nyc.csv",
    "foursquare_nyc_checkins.txt",
    "foursquare_nyc_checkins.csv",
]


def search_file(roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        for name in NYC_NAMES:
            found = next(root.rglob(name), None)
            if found is not None and found.is_file():
                return found
    return None


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading Foursquare NYC/Tokyo archive to {path} ...")
    urllib.request.urlretrieve(KAGGLE_DATASET_URL, path)


def extract_nyc(archive: Path, work_dir: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        nyc_member = None
        for member in members:
            lower = Path(member).name.lower()
            if lower in {name.lower() for name in NYC_NAMES} or ("nyc" in lower and lower.endswith((".txt", ".csv"))):
                nyc_member = member
                break
        if nyc_member is None:
            raise FileNotFoundError(f"NYC check-in file not found inside {archive}; members={members[:20]}")
        work_dir.mkdir(parents=True, exist_ok=True)
        zf.extract(nyc_member, work_dir)
        return work_dir / nyc_member


def install(source: Path | None, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "foursquare_nyc_checkins.txt"
    roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd(), Path.home() / "Downloads"]

    if source is not None:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file() and zipfile.is_zipfile(source):
            found = extract_nyc(source, output / ".foursquare_extract")
        elif source.is_dir():
            found = search_file([source])
            if found is None:
                raise FileNotFoundError(f"No Foursquare NYC check-in file found under {source}")
        else:
            found = source
    else:
        found = search_file(roots)
        if found is None:
            archive = output / "foursquare_nyc_tokyo_dataset.zip"
            download_archive(archive)
            found = extract_nyc(archive, output / ".foursquare_extract")

    if found.resolve() != target.resolve():
        shutil.copy2(found, target)

    # Quick validation. The adapter can handle no-header TSMC files, but here we
    # only check that the file is readable and has at least the standard columns.
    sample = pd.read_csv(target, sep="\t", header=None, nrows=3, engine="python")
    if sample.shape[1] < 8:
        sample = pd.read_csv(target, sep=",", header=None, nrows=3, engine="python")
    if sample.shape[1] < 5:
        raise ValueError(f"Foursquare NYC file looks too narrow: {sample.shape[1]} columns")
    print(f"Installed Foursquare NYC check-ins: {target}")
    print(f"Rows preview: {len(sample)} rows, {sample.shape[1]} columns")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None, help="optional archive, extracted directory, or NYC check-in file")
    parser.add_argument("--output", type=Path, default=Path("data/input"))
    args = parser.parse_args()
    install(args.source, args.output)


if __name__ == "__main__":
    main()
