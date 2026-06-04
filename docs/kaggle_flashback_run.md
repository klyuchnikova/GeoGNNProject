# Kaggle Runbook: Graph-Flashback on Gowalla Austin

This runbook is meant for a Kaggle notebook. The repository contains code only;
large Gowalla data, generated graphs, and logs are produced inside Kaggle.

## 1. Clone This Branch

```bash
git clone -b flashback-branch https://github.com/klyuchnikova/GeoGNNProject.git
cd GeoGNNProject
pip install -q -r flashback/requirements.txt
```

If the repository is private or Kaggle internet is disabled, upload the repo as
a Kaggle dataset or use the Kaggle Git integration, then `cd` into the project.

## 2. Get Gowalla SNAP Check-ins

Option A: attach a Kaggle dataset that contains:

```text
loc-gowalla_totalCheckins.txt.gz
```

Then set:

```bash
export GOWALLA_RAW=/kaggle/input/<your-dataset>/loc-gowalla_totalCheckins.txt.gz
```

Option B: if Kaggle internet is enabled:

```bash
mkdir -p data
wget -nc -P data https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
export GOWALLA_RAW=data/loc-gowalla_totalCheckins.txt.gz
```

## 3. Build the City Subset

Use Austin first. It is the safest first slice for Gowalla because the dataset
has dense activity around Austin/Texas, while the full global Gowalla is too
large for a first model run.

```bash
python scripts/prepare_gowalla_city.py \
  --input "$GOWALLA_RAW" \
  --city austin \
  --output data/checkins-gowalla-austin.txt \
  --min-checkins 101
```

Austin bbox used by default:

```text
lat: 30.10 .. 30.45
lng: -97.95 .. -97.55
```

If too few users remain, try `--min-checkins 51 --sequence-length 10` in the
training step, or use a larger custom bbox via:

```bash
python scripts/prepare_gowalla_city.py \
  --input "$GOWALLA_RAW" \
  --bbox 30.00 30.55 -98.10 -97.40 \
  --output data/checkins-gowalla-austin.txt \
  --min-checkins 101
```

## 4. Build Graphs Required by Flashback

This branch includes a lightweight graph builder so the model can run without
pretrained KGE artifacts:

```bash
python flashback/build_graphs.py \
  --dataset data/checkins-gowalla-austin.txt \
  --output-dir data/graphs \
  --dataset-name gowalla_austin \
  --min-checkins 101 \
  --transition-topk 100 \
  --user-loc-topk 100
```

It produces:

```text
data/graphs/gowalla_austin_transition_top100.pkl
data/graphs/gowalla_austin_user_loc_top100.pkl
```

## 5. Smoke Test

Use this first to verify the Kaggle environment:

```bash
python flashback/train.py \
  --dataset data/checkins-gowalla-austin.txt \
  --trans_loc_file data/graphs/gowalla_austin_transition_top100.pkl \
  --trans_interact_file data/graphs/gowalla_austin_user_loc_top100.pkl \
  --log_file results/flashback_austin_smoke \
  --epochs 1 \
  --batch-size 64 \
  --gpu -1
```

## 6. GPU Training

```bash
python flashback/train.py \
  --dataset data/checkins-gowalla-austin.txt \
  --trans_loc_file data/graphs/gowalla_austin_transition_top100.pkl \
  --trans_interact_file data/graphs/gowalla_austin_user_loc_top100.pkl \
  --log_file results/flashback_austin \
  --epochs 100 \
  --batch-size 200 \
  --validate-epoch 5 \
  --gpu 0
```

The evaluation log reports:

```text
Acc@1
Acc@5
Acc@10
MAP@5
MAP@10
MRR
```

## Notes

- For a paper-level Graph-Flashback reproduction, use the original KGE pipeline
  and pretrained graph artifacts from
  https://github.com/kevin-xuan/Graph-Flashback.
- For our robustness experiments, this branch is enough to start with a
  reproducible city-level baseline and then add perturbation/drift runs.
