# POI Recommendation Experiments: Filtration Methods Comparison

## Experiment Setup

- **Model**: GuGen (Transformer-based)
- **Datasets**: Foursquare (NYC, TKY) & Gowalla (Austin)
- **Sequence Length**: 20
- **Epochs**: 20
- **Base Configuration**: hidden_dim=256, layers=3, heads=8, lr=0.0005

## Results Summary

### Best ACC@1 per Dataset

| Dataset | Best Filter | ACC@1 | ACC@5 | ACC@10 | MRR |
|---------|------------|-------|-------|--------|-----|
| Foursquare NYC | min_20_20 | 18.85% | 56.34% | 73.22% | 0.3548 |
| Foursquare TKY | kcore_20_20 | 22.45% | 54.96% | 67.07% | 0.3694 |
| Gowalla Austin | kcore_20_20 | 14.27% | 30.26% | 38.26% | 0.2227 |

### Full

| Model | Dataset | Filter | ACC@1 | ACC@5 | ACC@10 | MRR |
|-------|---------|--------|-------|-------|--------|-----|
| **GuGen** | NYC | **min,20,20** | **18.85%** | 56.34% | 73.22% | 0.3548 |
| GuGen | NYC | k,20,20 | 18.30% | 55.34% | 73.15% | 0.3456 |
| GuGen | NYC | min,10,10 | 14.81% | 43.70% | 62.04% | 0.2885 |
| GuGen | NYC | k,10,10 | 16.80% | 48.86% | 65.47% | 0.3137 |
| GuGen | NYC | combined,10,10 | 16.80% | 48.86% | 65.47% | 0.3137 |
| GuGen | NYC | none | 13.38% | 36.73% | 48.11% | 0.2424 |
| **GuGen** | TKY | **k,20,20** | **22.45%** | 54.96% | 67.07% | 0.3694 |
| GuGen | TKY | min,20,20 | 21.37% | 54.56% | 67.38% | 0.3626 |
| GuGen | TKY | min,10,10 | 19.49% | 48.49% | 60.27% | 0.3266 |
| GuGen | TKY | k,10,10 | 19.49% | 48.49% | 60.27% | 0.3266 |
| GuGen | TKY | combined,10,10 | 19.49% | 48.49% | 60.27% | 0.3266 |
| GuGen | TKY | none | 16.46% | 41.36% | 51.42% | 0.2783 |
| **GuGen** | Gowalla | **k,20,20** | **14.27%** | 30.26% | 38.26% | 0.2227 |
| GuGen | Gowalla | min,20,20 | 14.27% | 30.26% | 38.26% | 0.2227 |
| GuGen | Gowalla | min,10,10 | 12.72% | 25.75% | 33.09% | 0.1950 |
| GuGen | Gowalla | k,10,10 | 12.72% | 25.75% | 33.09% | 0.1950 |
| GuGen | Gowalla | combined,10,10 | 12.72% | 25.75% | 33.09% | 0.1950 |
| GuGen | Gowalla | none | 9.11% | 19.99% | 25.83% | 0.1476 |
| **LSTM** | NYC | **min,20,20** | **15.07%** | 47.44% | 66.65% | 0.3024 |
| LSTM | NYC | k,20,20 | 14.44% | 48.21% | 66.37% | 0.2987 |
| LSTM | NYC | min,10,10 | 13.56% | 41.20% | 57.84% | 0.2695 |
| LSTM | NYC | k,10,10 | 13.56% | 41.20% | 57.84% | 0.2695 |
| LSTM | NYC | combined,10,10 | 13.56% | 41.20% | 57.84% | 0.2695 |
| LSTM | NYC | none | 11.61% | 32.51% | 45.84% | 0.2219 |
| **LSTM** | TKY | **min,20,20** | **19.73%** | 53.18% | 66.36% | 0.3466 |
| LSTM | TKY | k,20,20 | 19.65% | 52.88% | 66.09% | 0.3447 |
| LSTM | TKY | min,10,10 | 17.87% | 47.89% | 60.83% | 0.3159 |
| LSTM | TKY | k,10,10 | 17.87% | 47.89% | 60.83% | 0.3159 |
| LSTM | TKY | none | 15.75% | 40.81% | 51.29% | 0.2727 |

## Key Findings

GuGen outperforms LSTM across all datasets
- NYC: +3.78% absolute improvement (18.85% vs 15.07%)
- TKY: +2.72% improvement (22.45% vs 19.73%)