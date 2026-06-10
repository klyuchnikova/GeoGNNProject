# GeoGNNProject — clean Austin Graph-Flashback experiment suite

This branch contains an Austin-only Graph-Flashback implementation plus a tidy experiment runner modelled after the `gugen` branch style: a small set of named profiles, explicit filtration settings, reusable shared assets and resumable training.

The original project materials stay in the repository root and are preserved by the cleanup script:

- `GeoGNNProject.pdf`
- `gowalla_validated_metadata_eda.ipynb`
- `foursquaregraphs-eda.ipynb`

## What changed in v8

The old unsuccessful one-off runs are not part of the suite. The kept and added experiments are:

- three-seed tuned model for `mean ± std`;
- KGE graph ablations: no graph, transition-only, preference-only, full graph;
- rank-based STKG check;
- tuned friendship ablation;
- RNN/GRU/LSTM recurrent-cell comparison;
- GUGEN-style filtration sweep: min-only, iterative k-core, combined min+k-core;
- original-like Austin Graph-Flashback reference;
- graph-neighbor diagnostics showing whether the true next POI is present in KGE graph neighbors.

## Dataset

Attach `gowalla_austin_enriched_195k.zip` as a Kaggle Dataset. It contains only Austin Gowalla data:

- `gowalla_austin_checkins.csv(.gz)`
- `gowalla_austin_friendships.csv(.gz)`
- `manifest.json`
- `README.md`

Required check-in columns:

`raw_user_id, timestamp, latitude, longitude, raw_poi_id, category`

Foursquare is not mixed into this dataset.

## Experiment profiles

Profiles are defined in `experiments/austin_suite.yaml`.

| Profile | Purpose |
|---|---|
| `core` | Compact report-ready run: full model, graph ablations, friendship, original-like. |
| `full` | Main run: core + 3 seeds + cell sweep + selected filter sweep. |
| `filter_sweep` | Filtration sensitivity experiments in the GUGEN style. |
| `cells` | RNN vs GRU vs LSTM on the common Austin split. |
| `graph_ablation` | KGE graph and friendship ablations on the common split. |
| `original_like` | Rank-STKG Graph-Flashback with RNN/GRU/LSTM cells and paper-like settings. |

The `full` profile currently runs 15 experiments:

1. `tuned_full_seed42`
2. `tuned_full_seed7`
3. `tuned_full_seed2026`
4. `tuned_no_kge_graphs`
5. `tuned_transition_only`
6. `tuned_preference_only`
7. `tuned_no_priors`
8. `tuned_rank_scheme`
9. `tuned_with_friendship`
10. `tuned_lstm`
11. `tuned_rnn`
12. `filter_min20_20`
13. `filter_kcore20_20`
14. `filter_combined10_10`
15. `faithful_rank`

The `filter_sweep` profile additionally includes `filter_no_filter_2_2`, `filter_min10_10`, `filter_kcore10_10` and `filter_kcore30_20`.

## Install this overlay into `flashback-branch`

From the repository root in PowerShell:

```powershell
git checkout flashback-branch
Expand-Archive -LiteralPath "C:\path\GeoGNNProject_flashback_austin_clean_v8.zip" -DestinationPath "." -Force
python .\scripts\apply_clean_update.py
python .\scripts\verify_merge.py
git add -A
git commit -m "Clean Austin Graph-Flashback experiments"
git push origin flashback-branch
```

The Austin dataset is not committed to Git.

## Kaggle offline run

### One online step

Clone the branch while Internet is enabled:

```python
%cd /kaggle/working
!rm -rf GeoGNNProject
!git clone --depth 1 --branch flashback-branch --single-branch https://github.com/klyuchnikova/GeoGNNProject.git
%cd /kaggle/working/GeoGNNProject
```

Attach `gowalla_austin_enriched_195k.zip` as an Input Dataset. After cloning and attaching the dataset, Internet can be disabled. Kaggle GPU continues to work because the accelerator setting is independent of Internet access.

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

Main full run:

```python
!python scripts/run_austin_experiments.py --profile full
```

Smaller runs:

```python
!python scripts/run_austin_experiments.py --profile core
!python scripts/run_austin_experiments.py --profile filter_sweep
!python scripts/run_austin_experiments.py --profile cells
!python scripts/run_austin_experiments.py --profile graph_ablation
!python scripts/run_austin_experiments.py --profile original_like
```

The runner is resumable. If the session stops, repeat the same command; completed shared assets and completed experiment metrics are skipped.

## Package and download results

```python
!python scripts/package_results.py
```

The archive is written to:

`/kaggle/working/graph_flashback_austin_experiments.zip`

A browser download from a Kaggle notebook does not require outbound Internet access from the Python kernel. Click the file in the Output panel or display a link:

```python
from IPython.display import FileLink, display
display(FileLink('/kaggle/working/graph_flashback_austin_experiments.zip'))
```

## Local VS Code / PowerShell

You need only:

1. this repository/code overlay;
2. `gowalla_austin_enriched_195k.zip`;
3. a Python environment with packages from `requirements.txt` installed.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_offline.ps1 `
  -Profile full `
  -Dataset "C:\path\gowalla_austin_enriched_195k.zip"
```

For a fully offline new Windows machine, download platform-specific wheels once while online:

```powershell
python -m pip download -r requirements.txt -d wheelhouse
```

Then install offline:

```powershell
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
```

Do not reuse a Linux/Kaggle wheelhouse on Windows.

## Outputs

- `runs/shared/`: prepared data, STKG, TransE and sparse graph assets;
- `runs/shared/*/graphs/graph_neighbor_diagnostics.json`: KGE graph hit-rate diagnostics;
- `runs/experiments/<name>/`: histories, metrics, predictions and figures;
- `runs/summary/experiment_metrics.csv`: all experiment rows plus baselines and published references;
- `runs/summary/common_protocol_results.csv`: fixed Austin split experiments plus baselines and references;
- `runs/summary/per_filter_results.csv`: filter-sweep rows;
- `runs/summary/graph_diagnostics_summary.csv`: true-next-POI hit-rate inside KGE graph neighbors;
- `runs/summary/seed_summary.csv`: mean and standard deviation across the three full tuned seeds;
- `runs/summary/mrr_comparison.png`: compact MRR plot.

Metrics: `Acc@1/5/10`, `MAP@5/10`, `MRR`, plus macro-user variants inside detailed JSON files.

Published full-Gowalla values are included only as references. They are not directly comparable to Austin-only experiments.
