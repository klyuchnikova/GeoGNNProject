# Graph-Flashback Runtime

This directory contains a runnable adaptation of
[kevin-xuan/Graph-Flashback](https://github.com/kevin-xuan/Graph-Flashback)
for the GeoGNN robustness project.

The code keeps the original Flashback-style training loop, but adds:

- configurable `sequence_length`, `min_checkins`, and `max_users`;
- evaluation metrics required for our experiments:
  `Acc@1`, `Acc@5`, `Acc@10`, `MAP@5`, `MAP@10`, and `MRR`;
- `build_graphs.py`, a lightweight graph builder that creates the required
  POI transition graph and user-location graph from the prepared dataset.

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
  --min-checkins 101

python flashback/train.py \
  --dataset data/checkins-gowalla-austin.txt \
  --trans_loc_file data/graphs/gowalla_austin_transition_top100.pkl \
  --trans_interact_file data/graphs/gowalla_austin_user_loc_top100.pkl \
  --log_file results/flashback_austin \
  --gpu 0
```

Use `--gpu -1` for CPU smoke tests.
