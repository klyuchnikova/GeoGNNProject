import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Summarize experiment results")
parser.add_argument(
    "--model",
    default="gugen",
    choices=["gugen", "gugen_graph", "lstm"],
    help="Model prefix to scan under results/",
)
args = parser.parse_args()
model_prefix = args.model

print("Experiment | ACC@1 | ACC@5 | ACC@10 | MAP@5 | MAP@10 | MRR")
print("---------|-------|-------|--------|-------|--------|-----")

for exp_dir in sorted(Path("../results").glob(f"{model_prefix}_*")):
    if not exp_dir.is_dir():
        continue
    
    metrics_file = exp_dir / "metrics.json"
    if not metrics_file.exists():
        continue
    
    exp_name = exp_dir.name
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    # Check different possible locations for test metrics
    if 'test_metrics' in data:
        test_metrics = data['test_metrics']
    elif 'test' in data:
        test_metrics = data['test']
    else:
        test_metrics = data  # fallback to root
    
    acc1 = test_metrics.get('acc1', 0)
    acc5 = test_metrics.get('acc5', 0)
    acc10 = test_metrics.get('acc10', 0)
    map5 = test_metrics.get('map5', 0)  # Fixed this line
    map10 = test_metrics.get('map10', 0)  # Fixed this line
    mrr = test_metrics.get('mrr', 0)
    
    print(f"{exp_name} | {acc1:.4f} | {acc5:.4f} | {acc10:.4f} | {map5:.4f} | {map10:.4f} | {mrr:.4f}")

# Parse all results
results = {}

for exp_dir in Path("../results").glob(f"{model_prefix}_*"):
    if not exp_dir.is_dir():
        continue
    
    metrics_file = exp_dir / "metrics.json"
    if not metrics_file.exists():
        continue
    
    exp_name = exp_dir.name
    # Parse: {model}_foursquare_NYC_no_filter
    parts = exp_name.split('_')
    if len(parts) < 4:
        continue

    dataset = parts[1]
    city = parts[2]
    filter_type = '_'.join(parts[3:])
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    # Check different possible locations
    if 'test_metrics' in data:
        test_metrics = data['test_metrics']
    elif 'test' in data:
        test_metrics = data['test']
    else:
        test_metrics = data
    
    key = f"{dataset}_{city}"
    
    if key not in results:
        results[key] = {}
    
    results[key][filter_type] = test_metrics.get('acc1', 0)

# Print table
print("\n=== ACC@1 Comparison ===\n")
headers = ['Dataset/City', 'no_filter', 'min_10_10', 'min_20_20', 'kcore_10_10', 'kcore_20_20', 'combined_10_10']
print(f"{headers[0]:<20}", end="")
for h in headers[1:]:
    print(f"{h:>12}", end="")
print()

for dataset_city in sorted(results.keys()):
    print(f"{dataset_city:<20}", end="")
    for filter_type in headers[1:]:
        val = results[dataset_city].get(filter_type, 0)
        print(f"{val:>12.4f}", end="")
    print()

# Find best per dataset
print("\n=== Best Filter per Dataset ===\n")
for dataset_city in sorted(results.keys()):
    if results[dataset_city]:
        best_filter = max(results[dataset_city].items(), key=lambda x: x[1])
        print(f"{dataset_city:<20} Best: {best_filter[0]} (ACC@1={best_filter[1]:.4f})")