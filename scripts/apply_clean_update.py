"""Remove obsolete overlays while preserving the report and EDA notebooks."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
obsolete_files = [
    "README_OLD.md",
    "MERGE_INSTRUCTIONS.md",
    "IMPLEMENTATION_REPORT.md",
    "MANIFEST.sha256",
    "Makefile",
    "notebooks/flashback_results_analysis.ipynb",
    "scripts/download_gowalla.py",
    "scripts/download_foursquare.py",
    "scripts/rank_gowalla_cities.py",
    "scripts/export_austin_dataset.py",
    "flashback/data/downloads.py",
    "configs/foursquare_nyc.yaml",
    "configs/foursquare_tky.yaml",
]
obsolete_dirs = ["docs", "examples", "flashback/legacy", ".pytest_cache", ".pytest_tmp"]
for relative in obsolete_files:
    (root / relative).unlink(missing_ok=True)
for relative in obsolete_dirs:
    path = root / relative
    if path.exists():
        shutil.rmtree(path)
for path in root.rglob("__pycache__"):
    shutil.rmtree(path, ignore_errors=True)
for path in root.rglob("*.pyc"):
    path.unlink(missing_ok=True)
# Remove generated directories only when they are empty or contain only .gitkeep.
for relative in ["artifacts", "checkpoints", "data/processed", "data/kge", "data/graphs"]:
    path = root / relative
    if path.exists() and not any(
        item for item in path.rglob("*") if item.is_file() and item.name != ".gitkeep"
    ):
        shutil.rmtree(path)
print("Clean update applied; report and EDA notebooks were preserved.")
