#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-data/checkins-gowalla-austin.txt}"
GRAPH_DIR="${GRAPH_DIR:-data/graphs}"
DATASET_NAME="${DATASET_NAME:-gowalla_austin}"
GPU="${GPU:-0}"

python flashback/popularity_baseline.py \
  --dataset "$DATASET" \
  --min-checkins 101 \
  --batch-size 200

python flashback/build_graphs.py \
  --dataset "$DATASET" \
  --output-dir "$GRAPH_DIR" \
  --dataset-name "$DATASET_NAME" \
  --min-checkins 101 \
  --transition-topk 100 \
  --user-loc-topk 100 \
  --spatial-topk 50 \
  --spatial-radius-km 3.0

python flashback/train.py \
  --dataset "$DATASET" \
  --trans_loc_file "$GRAPH_DIR/${DATASET_NAME}_transition_top100.pkl" \
  --trans_interact_file "$GRAPH_DIR/${DATASET_NAME}_user_loc_top100.pkl" \
  --trans_loc_spatial_file "$GRAPH_DIR/${DATASET_NAME}_spatial_top50.pkl" \
  --use_spatial_graph \
  --log_file results/flashback_austin_gru_h32_lr003_spatial \
  --epochs 10 \
  --batch-size 200 \
  --validate-epoch 1 \
  --rnn gru \
  --hidden-dim 32 \
  --lr 0.003 \
  --gpu "$GPU"
