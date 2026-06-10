"""Remove files from the previous oversized overlay without touching EDA/report files."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
protected = {
    "GeoGNNProject.pdf",
    "foursquaregraphs-eda.ipynb",
    "gowalla_validated_metadata_eda.ipynb",
    "LICENSE",
}
obsolete_files = [
    "README_OLD.md", "MERGE_INSTRUCTIONS.md", "IMPLEMENTATION_REPORT.md",
    "MANIFEST.sha256", "Makefile", "notebooks/flashback_results_analysis.ipynb",
    "scripts/download_gowalla.py", "scripts/download_foursquare.py",
    "scripts/rank_gowalla_cities.py",
]
obsolete_dirs = ["docs", "examples", "flashback/legacy", ".pytest_cache"]
for relative in obsolete_files:
    if relative not in protected:
        (root / relative).unlink(missing_ok=True)
for relative in obsolete_dirs:
    path = root / relative
    if path.exists():
        shutil.rmtree(path)
for path in root.rglob("__pycache__"):
    shutil.rmtree(path, ignore_errors=True)
for path in root.rglob("*.pyc"):
    path.unlink(missing_ok=True)
# Remove only empty generated directories; do not destroy completed experiments.
for relative in ["artifacts/city_selection", "artifacts/results", "artifacts/figures", "artifacts/predictions"]:
    path = root / relative
    if path.exists() and not any(p for p in path.rglob("*") if p.is_file() and p.name != ".gitkeep"):
        shutil.rmtree(path)
print("Clean update applied. EDA notebooks and GeoGNNProject.pdf were preserved.")
