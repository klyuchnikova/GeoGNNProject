from pathlib import Path
required = [
    "GeoGNNProject.pdf",
    "foursquaregraphs-eda.ipynb",
    "gowalla_validated_metadata_eda.ipynb",
    "README.md",
    "flashback/pipeline.py",
    "configs/gowalla_auto.yaml",
    "notebooks/kaggle_graph_flashback.ipynb",
]
missing = [path for path in required if not Path(path).exists()]
if missing:
    raise SystemExit("Missing required project files: " + ", ".join(missing))
print("Project verification passed")
