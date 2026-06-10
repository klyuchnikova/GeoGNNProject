# GeoGNNProject — full Graph-Flashback pipeline

This overlay preserves the existing project report and EDA notebooks and adds a reproducible, train-only **STKG → TransE → Graph-Flashback** implementation for Gowalla and Foursquare.

## Main run

```bash
python -m pip install -r requirements.txt
python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage all
```

The Gowalla run downloads the official SNAP check-ins and friendship graph, optionally joins Figshare spot metadata, objectively ranks candidate metro areas, selects a viable city before model training, creates chronological splits, builds a train-only STKG, trains TransE, derives sparse KGE graphs, trains Graph-Flashback, evaluates it and creates figures.

Required metrics: `Acc@1/5/10`, `MAP@5/10`, and `MRR`. Both micro and macro-user results are written to `artifacts/results`.

See [docs/KAGGLE_RUN.md](docs/KAGGLE_RUN.md), [docs/METHODOLOGY.md](docs/METHODOLOGY.md), and [MERGE_INSTRUCTIONS.md](MERGE_INSTRUCTIONS.md).

## Stages

```bash
python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage download
python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage prepare
python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage stkg
python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage kge
python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage graphs
python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage train
python -m flashback.pipeline --config configs/gowalla_auto.yaml --stage analyze
```

## Foursquare

Download/mount TSMC2014 and switch only the config:

```bash
python scripts/download_foursquare.py
python -m flashback.pipeline --config configs/foursquare_nyc.yaml --stage all
```

TSMC does not provide the SNAP-style friendship graph, so the Foursquare presets disable only the social relation; the model, STKG, transition graph, user–POI graph and evaluation code are unchanged.

## Verification

```bash
pytest -q
python scripts/make_synthetic_data.py
python -m flashback.pipeline --config configs/gowalla_smoke.yaml --stage all
```
