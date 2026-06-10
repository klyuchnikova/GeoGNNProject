#!/bin/bash
set -euo pipefail

PROFILE="${1:-full}"

echo "=========================================="
echo "Graph-Flashback Austin experiments"
echo "Profile: ${PROFILE}"
echo "=========================================="

python scripts/check_offline_environment.py
python scripts/install_austin_dataset.py
python scripts/verify_merge.py
python scripts/run_austin_experiments.py --profile "${PROFILE}"
python scripts/package_results.py

echo "=========================================="
echo "Completed profile: ${PROFILE}"
echo "Results: graph_flashback_austin_experiments.zip"
echo "=========================================="
