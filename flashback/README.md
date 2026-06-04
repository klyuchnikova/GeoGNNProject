# Graph-Flashback Runtime

This directory contains a runnable adaptation of
[kevin-xuan/Graph-Flashback](https://github.com/kevin-xuan/Graph-Flashback)
for the GeoGNN robustness project.

The code keeps the original Flashback-style training loop, but adds:

- configurable `sequence_length`, `min_checkins`, and `max_users`;
- evaluation metrics required for our experiments:
  `Acc@1`, `Acc@5`, `Acc@10`, `MAP@5`, `MAP@10`, and `MRR`;
- `popularity_baseline.py`, a global popularity baseline using the same split
  and metrics as the model;
- `build_graphs.py`, a lightweight graph builder that creates the required
  POI transition graph, user-location graph, and spatial POI graph from the
  prepared dataset.

## Expected Data Format

The training file must be tab-separated, with one check-in per line:

```text
user_id<TAB>timestamp<TAB>latitude<TAB>longitude<TAB>location_id
```

For compatibility with the original loader, rows for each user must be
contiguous and sorted by timestamp descending. The city preparation script in
`../scripts/prepare_gowalla_city.py` writes exactly this format.

## Minimal Run

```bash
python scripts/prepare_gowalla_city.py \
  --input /kaggle/input/<dataset>/loc-gowalla_totalCheckins.txt.gz \
  --city austin \
  --output data/checkins-gowalla-austin.txt \
  --min-checkins 101

python flashback/build_graphs.py \
  --dataset data/checkins-gowalla-austin.txt \
  --output-dir data/graphs \
  --dataset-name gowalla_austin \
  --min-checkins 101 \
  --transition-topk 100 \
  --user-loc-topk 100 \
  --spatial-topk 50 \
  --spatial-radius-km 3.0

python flashback/popularity_baseline.py \
  --dataset data/checkins-gowalla-austin.txt \
  --min-checkins 101 \
  --batch-size 200

python flashback/train.py \
  --dataset data/checkins-gowalla-austin.txt \
  --trans_loc_file data/graphs/gowalla_austin_transition_top100.pkl \
  --trans_interact_file data/graphs/gowalla_austin_user_loc_top100.pkl \
  --trans_loc_spatial_file data/graphs/gowalla_austin_spatial_top50.pkl \
  --use_spatial_graph \
  --log_file results/flashback_austin_gru_h32_lr003_spatial \
  --epochs 10 \
  --batch-size 200 \
  --validate-epoch 1 \
  --rnn gru \
  --hidden-dim 32 \
  --lr 0.003 \
  --gpu 0
```

Use `--gpu -1` for CPU smoke tests.

## KGE Graphs

The graph builder above is a city-level approximation, not the original KGE
pipeline. For closest paper-style reproduction, use graph `.pkl` files generated
from pretrained KGE embeddings on the exact same remapped dataset ids.
