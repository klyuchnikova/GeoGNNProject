#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dl_inference
cd src

BASE_ARGS="--model gugen --seq-len 20 --batch-size 512 --epochs 20 --lr 0.0005 --weight-decay 1e-5 --hidden-dim 256 --num-layers 3 --num-heads 8 --device cuda --seed 42 --lr-scheduler-factor 0.1 --lr-scheduler-patience 3"

declare -A DATASET_CITIES
DATASET_CITIES["foursquare"]="NYC TKY"
DATASET_CITIES["gowalla"]="austin"

FILTERS=(
  # 1. No filtration (baseline)
  "no_filter|--min-user-visits 2 --min-poi-visits 2 --no-kcore"
  
  # 2. Weak filtration (default)
  "min_10_10|--min-user-visits 10 --min-poi-visits 10 --no-kcore"
  
  # 3. Strong Min Visits filtration
  "min_20_20|--min-user-visits 20 --min-poi-visits 20 --no-kcore"
  
  # 4. K-Core only (no min-visits)
  "kcore_10_10|--min-user-visits 2 --min-poi-visits 2 --user-k 10 --poi-k 10"
  
  # 5. Strong K-Core
  "kcore_20_20|--min-user-visits 2 --min-poi-visits 2 --user-k 20 --poi-k 20"
  
  # 6. Combined (Min Visits + K-Core)
  "combined_10_10|--min-user-visits 10 --min-poi-visits 10 --user-k 10 --poi-k 10"
)

for DATASET in foursquare gowalla; do
  echo "=========================================="
  echo "Running experiments on $DATASET"
  echo "=========================================="

  CITIES=${DATASET_CITIES[$DATASET]}

  for CITY in $CITIES; do
    echo "----------------------------------------"
    echo "Dataset: $DATASET, City: $CITY"
    echo "----------------------------------------"
    for filter_config in "${FILTERS[@]}"; do
        EXP_NAME="${filter_config%%|*}"
        FILTER_ARGS="${filter_config##*|}"
 
        echo "----------------------------------------"
        echo "Experiment: $EXP_NAME"
        echo "Filters: $FILTER_ARGS"
        echo "----------------------------------------"

        CMD="python train.py \
        --dataset $DATASET \
        --city $CITY \
        --data-root ../input \
        --results-dir ../results \
        --checkpoint-dir ../checkpoints \
        $BASE_ARGS \
        $FILTER_ARGS \
        --exp-name $EXP_NAME"
        
        echo "Running: $CMD"
        $CMD

        if [ $? -eq 0 ]; then
        echo "✓ Completed: $EXP_NAME on $DATASET"
        else
        echo "✗ Failed: $EXP_NAME on $DATASET"
        fi

        echo ""
        done
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="