# GeoGNNProject — offline Austin Graph-Flashback experiment suite

This branch contains a clean Austin-only implementation of Graph-Flashback and an enhanced ranking-oriented variant. The full experiment suite can run in Kaggle **after the repository has been cloned**, with Internet disabled. The only attached input is `gowalla_austin_enriched_195k.zip`.

The original project materials remain in the repository root and are not replaced by this overlay:

- `GeoGNNProject.pdf`
- `gowalla_validated_metadata_eda.ipynb`
- `foursquaregraphs-eda.ipynb`


## Install this overlay into `flashback-branch`

From the repository root in PowerShell:

```powershell
git checkout flashback-branch
Expand-Archive -LiteralPath "C:\path\GeoGNNProject_flashback_austin_offline_v7.zip" -DestinationPath "." -Force
python .\scripts\apply_clean_update.py
python .\scripts\verify_merge.py
git add -A
git commit -m "Add offline Austin Graph-Flashback experiment suite"
git push origin flashback-branch
```

The cleanup script preserves the PDF and both original EDA notebooks. The Austin dataset is not committed to Git.

## Dataset

Attach `gowalla_austin_enriched_195k.zip` as a Kaggle Dataset. It contains Austin Gowalla check-ins enriched with POI categories and an optional Austin friendship edge list. Foursquare is not mixed into this dataset.

Required check-in columns:

`raw_user_id, timestamp, latitude, longitude, raw_poi_id, category`

## Models and experiments

The suite keeps the original Graph-Flashback core:

`train-only STKG → TransE → KGE transition/preference graphs → graph propagation → recurrent encoder → temporal/spatial/preference flashback → next-POI ranking`

Two shared graph constructions are tested:

- **radius STKG**: 3 km then top-50, used by the strongest tuned configuration;
- **rank STKG**: symmetric top-50 nearest POIs, matching the rank-based scheme described in the paper.

The `full` profile runs:

1. enhanced Graph-Flashback with seeds 42, 7 and 2026;
2. enhanced Flashback without KGE graphs;
3. transition-graph-only ablation;
4. preference-graph-only ablation;
5. full model without train-only ranking priors;
6. full model without BPR;
7. tuned model with rank-based STKG;
8. faithful Austin Graph-Flashback with RNN-10 and rank-based STKG.

All tuned radius ablations reuse one prepared dataset, one TransE model and one pair of sparse graphs. The run is resumable: completed assets and experiment metrics are skipped.

## Kaggle offline run

### One online step

Clone `flashback-branch` while Internet is enabled, then attach the Austin Dataset. After that Internet can be disabled. GPU and Internet are independent Kaggle settings.

### Offline cells

```python
%cd /kaggle/working/GeoGNNProject
!python scripts/check_offline_environment.py
!python scripts/install_austin_dataset.py
!python scripts/verify_merge.py
```

Optional smoke test:

```python
!python scripts/make_synthetic_data.py
!python -m flashback.pipeline --config configs/smoke.yaml --stage all
!python -m pytest -q --basetemp=/kaggle/working/pytest_tmp
```

Full experiment suite:

```python
!python scripts/run_austin_experiments.py --profile full
```

If the session stops, run the same command again. Existing stages and completed experiments are reused.

Package results:

```python
!python scripts/package_results.py
```

The ZIP is written to:

`/kaggle/working/graph_flashback_austin_experiments.zip`

A browser download from the Kaggle notebook does not require outbound Internet access from the kernel. You can click the file in the Output panel or use the final notebook cell. Saving a notebook version also preserves `/kaggle/working` outputs for later download.

## Local VS Code / PowerShell

You need only:

1. this repository/code overlay;
2. `gowalla_austin_enriched_195k.zip`;
3. a Python environment with packages from `requirements.txt` already installed.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local_offline.ps1 `
  -Profile full `
  -Dataset "C:\path\gowalla_austin_enriched_195k.zip"
```

For a completely offline new Windows environment, download platform-specific wheels once while online:

```powershell
python -m pip download -r requirements.txt -d wheelhouse
```

Then install offline on the target machine:

```powershell
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
```

Do not reuse a Linux/Kaggle wheelhouse on Windows.

## Outputs

- `runs/shared/`: prepared data, STKG, TransE and sparse graph assets;
- `runs/experiments/<name>/`: model history, metrics, predictions and figures;
- `runs/summary/experiment_metrics.csv`: all experiments and published reference rows;
- `runs/summary/seed_summary.csv`: mean and standard deviation across the three full tuned seeds;
- `runs/summary/mrr_comparison.png`: compact comparison plot.

Metrics: `Acc@1/5/10`, `MAP@5/10`, `MRR`, plus macro-user variants.

Published full-Gowalla numbers are included only as non-comparable references; Austin uses a different subset and protocol.
