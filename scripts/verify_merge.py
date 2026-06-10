from pathlib import Path
required=["GeoGNNProject.pdf","foursquaregraphs-eda.ipynb","gowalla_validated_metadata_eda.ipynb","flashback/pipeline.py","docs/KAGGLE_RUN.md"]
missing=[p for p in required if not Path(p).exists()]
if missing: raise SystemExit("Missing after merge: "+", ".join(missing))
print("Merge verification passed")
