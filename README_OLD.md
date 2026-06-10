# GeoGNNProject

Research project on next-POI recommendation under spatial noise and temporal
drift.

This branch adds a runnable Graph-Flashback baseline adapted from
https://github.com/kevin-xuan/Graph-Flashback.

Key files:

- `flashback/` - model runtime, training loop, graph builder, and evaluation.
- `flashback/popularity_baseline.py` - global popularity sanity baseline.
- `scripts/prepare_gowalla_city.py` - creates a city-level Gowalla subset.
- `scripts/run_flashback_austin_improved.sh` - Kaggle-friendly improved
  Flashback run: baseline, temporal/user-location/spatial graph build, and
  GRU training.
- `docs/kaggle_flashback_run.md` - Kaggle commands for preparing data and
  launching the model.
- `gowalla_validated_metadata_eda.ipynb` and `foursquaregraphs-eda.ipynb` -
  exploratory notebooks used in the project report.

Default Gowalla subset for the first run: Austin.
