# GeoGNNProject — Graph-Flashback for the Austin Gowalla subset

This branch contains a clean, reproducible implementation of Graph-Flashback for next-POI recommendation. The default experiment uses the portable Austin-only Gowalla package instead of downloading and joining the complete global Gowalla dataset inside Kaggle.

The repository keeps the original project materials in the root:

- `GeoGNNProject.pdf`
- `gowalla_validated_metadata_eda.ipynb`
- `foursquaregraphs-eda.ipynb`

The model code is in `flashback/`; generated datasets, graphs, checkpoints and results are intentionally excluded from Git.

## Dataset expected by the project

Attach or download:

```text
gowalla_austin_enriched_195k.zip
```

The package contains:

```text
gowalla_austin_checkins.csv.gz
gowalla_austin_friendships.csv.gz   # optional for the faithful STKG run
manifest.json
README.md
```

The recommended tuned experiment requires only the check-in file. Its columns are:

| Column | Meaning | Used by the tuned model |
|---|---|---|
| `raw_user_id` | original Gowalla user identifier | user embedding, user–POI graph, personal prior |
| `timestamp` | UTC check-in time | chronological split, temporal flashback, time features |
| `latitude` | POI latitude | spatial flashback and candidate-distance prior |
| `longitude` | POI longitude | spatial flashback and candidate-distance prior |
| `raw_poi_id` | original Gowalla POI identifier | POI embedding and prediction target |
| `category` | category joined from Gowalla spot metadata | category embedding and category-transition prior |

The package was formed as:

```text
SNAP Gowalla check-ins
→ select the Austin metro area by coordinates
→ left join Gowalla spot metadata by the original POI id
→ retain original user/POI ids and categories
→ export Austin-only files
```

Foursquare records are not mixed into this dataset.

## What is implemented

### Graph-Flashback core

The faithful path retains the main structure of the original repository:

```text
train check-ins
→ spatial-temporal knowledge graph
→ TransE embeddings
→ KGE-derived POI transition graph
→ KGE-derived user–POI preference graph
→ graph-propagated POI embeddings
→ RNN/GRU sequence encoder
→ temporal × spatial × preference flashback aggregation
→ user-conditioned next-POI ranking
```

The original implementation and paper are available at:

- https://github.com/kevin-xuan/Graph-Flashback
- https://doi.org/10.1145/3534678.3539383

### Tuned improvements

`configs/gowalla_auto.yaml` is designed for the strongest Austin result while keeping the Graph-Flashback core. It adds:

- iterative user/POI k-core filtering (`30/10`);
- chronological `80/10/10` train/validation/test split;
- rolling 20-event windows with stride 1, so every target has a full recent context;
- GRU with hidden size 128;
- POI category, hour-of-week and time-gap embeddings;
- user embedding added to recurrent inputs;
- two LightGCN-style propagation steps on the transition graph;
- relation-balanced TransE training with four negatives per positive;
- train-only personal and global popularity priors;
- recent-history, candidate-distance and category-transition priors;
- a learned repeat/explore gate that controls repeat-oriented priors;
- cross-entropy plus auxiliary BPR ranking loss and hard negatives;
- checkpoint selection by validation MRR.

These additions are inspired by course topics on LightGCN, ranking objectives, negative sampling, temporal context and graph construction. This tuned mode is an enhanced Graph-Flashback experiment, not a verbatim reproduction of the paper protocol.

A key implementation correction is that user-preference similarity is computed against graph-propagated POI embeddings only. Category/time context embeddings are not incorrectly compared with the user–POI graph representation.

## Configurations

### `configs/gowalla_auto.yaml`

Recommended tuned Austin experiment. Use this first.

### `configs/gowalla_faithful.yaml`

Austin adaptation closer to the original Graph-Flashback architecture:

- RNN hidden size 10;
- fixed-block sequences;
- no category/time context;
- no popularity or candidate priors;
- no BPR auxiliary loss;
- one graph-propagation step.

It has a validation split for model selection.

### `configs/gowalla_paper.yaml`

Paper-style `80/20` run after all decisions have been made. It trains for the configured number of epochs and evaluates test only at the end. Do not use this configuration for hyperparameter selection.

### `configs/gowalla_smoke.yaml`

Small synthetic CPU test of the complete pipeline.

## Local installation on Windows PowerShell

From the root of `GeoGNNProject`:

```powershell
git checkout flashback-branch

Expand-Archive `
  -LiteralPath "C:\Users\marga\Downloads\GeoGNNProject_flashback_final_v6.zip" `
  -DestinationPath "." `
  -Force

python .\scripts\apply_clean_update.py
python -m pip install -r requirements.txt

python .\scripts\install_austin_dataset.py `
  --archive "C:\Users\marga\Downloads\gowalla_austin_enriched_195k.zip"

python .\scripts\verify_merge.py

New-Item -ItemType Directory -Force ".pytest_tmp"
python -m pytest -q --basetemp="$PWD\.pytest_tmp"
Remove-Item -Recurse -Force ".pytest_tmp"
```

Then commit the cleaned project:

```powershell
git add -A
git commit -m "Add final tuned Graph-Flashback pipeline"
git push origin flashback-branch
```

The Austin dataset is under `data/`, which is ignored by Git.

## Kaggle: complete run

Enable a T4 GPU and attach the Kaggle Dataset containing `gowalla_austin_enriched_195k.zip`.

### 1. Clone the exact branch

```python
%cd /kaggle/working
!rm -rf GeoGNNProject
!git clone --depth 1 --branch flashback-branch --single-branch \
    https://github.com/klyuchnikova/GeoGNNProject.git
%cd /kaggle/working/GeoGNNProject
!git branch --show-current
!git log -1 --oneline
```

### 2. Install dependencies without replacing Kaggle PyTorch

```python
!grep -v '^torch' requirements.txt > /tmp/requirements-kaggle.txt
!python -m pip install -q -r /tmp/requirements-kaggle.txt
```

### 3. Install the attached Austin package

```python
!python scripts/install_austin_dataset.py
!python scripts/verify_merge.py
```

Check the GPU:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### 4. Smoke-test

```python
!python scripts/make_synthetic_data.py
!python -m flashback.pipeline \
    --config configs/gowalla_smoke.yaml \
    --stage all
!python -m pytest -q --basetemp=/kaggle/working/pytest_tmp
```

### 5. Tuned Austin run

Run stages separately so that an interrupted Kaggle session is easier to diagnose:

```python
CONFIG = "configs/gowalla_auto.yaml"
```

```python
!python -m flashback.pipeline --config $CONFIG --stage prepare
!python -m flashback.pipeline --config $CONFIG --stage stkg
!python -m flashback.pipeline --config $CONFIG --stage kge
!python -m flashback.pipeline --config $CONFIG --stage graphs
!python -m flashback.pipeline --config $CONFIG --stage train
!python -m flashback.pipeline --config $CONFIG --stage analyze
```

## Metrics

The evaluator reports:

- `Acc@1`, `Acc@5`, `Acc@10`;
- `MAP@5`, `MAP@10`;
- `MRR`;
- macro-user variants;
- global-popularity and personal-popularity baselines.

For one relevant next POI, `MAP@K` is the mean reciprocal rank truncated at K.

The original Graph-Flashback paper reports Gowalla results on a much larger processed global dataset, not on this Austin subset. Therefore article numbers are a reference target, not a guaranteed directly comparable result. The first acceptance criterion for the tuned Austin model is that it beats personal popularity on the same split, especially in `MRR`, `Acc@5` and `Acc@10`.

## Generated outputs

After a run:

```text
data/processed/       remapped Austin table and split manifests
data/kge/             STKG triplets, TransE checkpoint and diagnostics
data/graphs/          sparse transition and preference graphs
checkpoints/          best Graph-Flashback checkpoint
artifacts/results/    metrics and learning history
artifacts/predictions target ranks and top-1 predictions
artifacts/figures/    learning curves and metric plots
```

These directories are created only when needed; no empty generated folders are shipped in the archive.

## Recommended experiment order

1. Run `gowalla_auto.yaml` and compare with personal popularity.
2. Run `gowalla_faithful.yaml` as the architectural reference.
3. Inspect TransE relation diagnostics and learning curves.
4. If the tuned model wins on validation, freeze all choices.
5. Run `gowalla_paper.yaml` once for the final paper-style test.
6. Report exact configuration, dataset counts, seed and `mean ± std` for multiple seeds when compute permits.

## Reproducibility notes

- All model-selection decisions use validation metrics.
- Test is evaluated after loading the selected checkpoint.
- STKG, TransE priors and popularity priors use train events only.
- POI coordinates and categories are static metadata, not future interaction labels.
- The tuned k-core is applied before the temporal split to define a stable closed vocabulary; this differs from a strict cold-start protocol and must be disclosed in the report.
