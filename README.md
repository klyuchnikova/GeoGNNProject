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

### Full Comparison Table (ACC@1)

| Dataset/City | no_filter | min_10_10 | min_20_20 | kcore_10_10 | kcore_20_20 | combined_10_10 |
|-------------|-----------|-----------|-----------|-------------|-------------|----------------|
| Foursquare NYC | 13.38% | 14.81% | **18.85%** | 16.80% | 18.30% | 16.80% |
| Foursquare TKY | 16.46% | 19.49% | 21.37% | 19.49% | **22.45%** | 19.49% |
| Gowalla Austin | 9.11% | 12.72% | 14.27% | 12.72% | **14.27%** | 12.72% |

## Key Findings

Filtration Significantly Improves Performance. And stronger filtration - better results.
- NYC: +5.47% absolute improvement (13.38% → 18.85%)
- TKY: +5.99% improvement (16.46% → 22.45%)
- Austin: +5.16% improvement (9.11% → 14.27%)