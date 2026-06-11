# GeoGNNProject — Graph-Flashback experiments

This branch contains a clean, report-ready implementation of next-POI recommendation experiments based on Graph-Flashback ideas. The code focuses on two datasets:

1. **Gowalla Austin** — the main evaluated dataset for the project.
2. **Foursquare NYC** — an additional optional run that can be launched from Kaggle to complete a second-dataset results table.

The repository intentionally contains only source code, configs and experiment runners. It does **not** include generated checkpoints, prepared graphs, Kaggle outputs, old failed runs, smoke tests, offline wrappers or dataset files.

## Repository structure

```text
flashback/                  Core implementation
  data/                     Input adapters, preprocessing, sequence datasets
  kge/                      STKG construction, TransE training, graph construction
  model/                    Graph-Flashback and graph-memory model
  evaluation/               Acc@k, MAP@k, MRR, macro-user metrics
  analysis/                 Result visualizations
configs/                    Report-ready experiment configs
experiments/                Named experiment suites
scripts/                    Kaggle/local launch and packaging scripts
requirements.txt            Python dependencies
pyproject.toml              Package metadata
```

## Gowalla Austin dataset

The main data file is expected as a separate archive:

```text
gowalla_austin_enriched_195k.zip
```

It contains an Austin-only enriched subset of Gowalla with categories:

| Field | Value |
|---|---:|
| Check-ins before downstream filtering | 195,727 |
| Users | 769 |
| POIs | 10,077 |
| Optional friendship edges | 6,938 |
| Category coverage | about 98.4% |
| Time range | 2009-03-13 to 2010-10-22 |

The subset is built from Gowalla check-ins, Gowalla POI metadata and optional Gowalla friendship edges. Austin is selected by a coordinate bounding box, and the model applies stricter filters inside each experiment.

Expected columns:

```text
raw_user_id,timestamp,latitude,longitude,raw_poi_id,category
```

The dataset archive is not committed to GitHub. Attach it as Kaggle input or keep it locally.

## Implemented methods

### 1. Plain Flashback baseline

Experiment:

```text
plain_flashback_min20_20
```

This is the ordinary sequential baseline on the Austin min20/20 split. It disables KGE-derived transition/user-POI graphs and disables the additional ranking priors. It is included so that the graph-enhanced variants are not compared only against popularity baselines.

### 2. Original-like Graph-Flashback

Experiment:

```text
original_like_graph_flashback
```

This configuration follows the paper-style pipeline more closely:

```text
STKG → TransE → POI transition graph + user-POI preference graph → RNN + Flashback
```

It uses rank-based spatial relations, friendship triplets, small hidden dimension, no category/time priors, no BPR loss and a paper-like split. It is kept as a reproduction-oriented reference, not as the strongest tuned model.

### 3. Enhanced Graph-Flashback, common Austin split

Experiment:

```text
enhanced_graph_flashback_common
```

This is the project’s common Austin protocol. It uses:

- rolling windows of length 20;
- GRU hidden size 128;
- category embedding;
- hour-of-week and time-gap embeddings;
- two graph propagation layers;
- train-only personal, recent, global, geographical and category-transition priors;
- repeat/explore gate;
- cross-entropy plus BPR ranking loss;
- hard negatives;
- validation checkpointing by MRR.

### 4. Enhanced Graph-Flashback, Austin min20/20

Experiment:

```text
enhanced_graph_flashback_min20_20
```

This is the strongest result from the clean Austin sweep already run in Kaggle. It uses the same enhanced model as above, but on the stronger min20/20 filtered Austin protocol.

### 5. Graph-memory Enhanced Graph-Flashback, Austin min20/20

Experiment:

```text
graph_memory_graph_flashback_min20_20
```

This is the strongest integrated variant. It adds graph-memory priors to the enhanced model:

- dynamic user-history prior built only from locations visited before the target;
- direct transition-graph prior from the last observed POI to graph-neighbor candidate POIs;
- learnable prior weights;
- repeat gate;
- min20/20 Austin filtering.

The goal is to make the graph signal affect the final ranking directly, not only through graph-smoothed embeddings. This is useful because the Austin diagnostics show that the true next POI is sometimes present in KGE graph neighborhoods, but the indirect GCN signal alone is weak.

## Gowalla Austin experiments to run

Install the dataset and run the compact report suite:

```python
%cd /kaggle/working/GeoGNNProject

!python scripts/install_austin_dataset.py
!python scripts/run_gowalla_austin_experiments.py --profile report
!python scripts/rebuild_summary.py
!python scripts/package_results.py
```

Faster run with only the two strongest Austin models:

```python
!python scripts/run_gowalla_austin_experiments.py --profile best
```

The result ZIP is written to:

```text
/kaggle/working/graph_flashback_report_results.zip
```

## Foursquare NYC experiments

The Foursquare NYC run is optional and intended to complete the report table with a second dataset.

The script first looks for an attached Kaggle input containing the standard TSMC2014 NYC file. If it is not attached, it downloads the public Kaggle dataset `chetanism/foursquare-nyc-and-tokyo-checkin-dataset` while Kaggle Internet is enabled.

Run:

```python
%cd /kaggle/working/GeoGNNProject

!python scripts/run_foursquare_nyc_experiments.py --profile report
!python scripts/rebuild_summary.py
!python scripts/package_results.py
```

Faster Foursquare run with only the enhanced and graph-memory models:

```python
!python scripts/run_foursquare_nyc_experiments.py --profile best
```

Foursquare NYC experiments:

```text
foursquare_nyc_plain_flashback
foursquare_nyc_enhanced_graph_flashback
foursquare_nyc_graph_memory_graph_flashback
```
## Results

The table below summarizes the obtained results for the Austin Gowalla subset and the new Foursquare NYC/Tokyo runs. Paper-reference rows are included only as external reference points and are not directly comparable, because they use the full processed Gowalla dataset and a different evaluation protocol.

| Model                                     | Dataset / protocol             |      Acc@1 |      Acc@5 |     Acc@10 |      MAP@5 |     MAP@10 |        MRR |
| ----------------------------------------- | ------------------------------ | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Global popularity                         | Gowalla Austin common split    |     0.0136 |     0.0519 |     0.0846 |     0.0288 |     0.0333 |     0.0393 |
| Personal popularity                       | Gowalla Austin common split    |     0.1021 |     0.2381 |     0.3044 |     0.1513 |     0.1600 |     0.1688 |
| Original-like Graph-Flashback             | Gowalla Austin paper-like/rank |     0.0791 |     0.1641 |     0.2032 |     0.1105 |     0.1158 |     0.1232 |
| Enhanced Graph-Flashback                  | Gowalla Austin common split    |     0.1149 |     0.2507 |     0.3214 |     0.1643 |     0.1737 |     0.1852 |
| Enhanced Graph-Flashback                  | Gowalla Austin min20/20        |     0.1329 |     0.2882 |     0.3631 |     0.1898 |     0.1998 |     0.2117 |
| **Graph-memory Enhanced Graph-Flashback** | **Gowalla Austin min20/20**    | **0.1422** | **0.2932** | **0.3740** | **0.1974** | **0.2081** | **0.2201** |
| Plain Flashback                           | Foursquare NYC min20/20        |     0.2973 |     0.6252 |     0.7100 |     0.4238 |     0.4353 |     0.4415 |
| Enhanced Graph-Flashback                  | Foursquare NYC min20/20        |     0.3143 |     0.6384 |     0.7253 |     0.4366 |     0.4485 |     0.4547 |
| **Graph-memory Enhanced Graph-Flashback** | **Foursquare NYC min20/20**    | **0.3206** | **0.6492** | **0.7378** | **0.4460** | **0.4580** | **0.4642** |
| Plain Flashback                           | Foursquare Tokyo min20/20      |     0.2444 |     0.5298 |     0.6300 |     0.3505 |     0.3640 |     0.3729 |
| Enhanced Graph-Flashback                  | Foursquare Tokyo min20/20      |     0.2586 |     0.5342 |     0.6347 |     0.3605 |     0.3741 |     0.3831 |
| **Graph-memory Enhanced Graph-Flashback** | **Foursquare Tokyo min20/20**  |     0.2568 | **0.5427** | **0.6457** | **0.3628** | **0.3768** | **0.3860** |
| Flashback, paper reference                | Full Gowalla                   |     0.1158 |     0.2754 |     0.3479 |          — |          — |     0.1925 |
| Graph-Flashback, paper reference          | Full Gowalla                   |     0.1512 |     0.3425 |     0.4256 |          — |          — |     0.2422 |

### Notes

* The strongest Gowalla Austin result is obtained by the graph-memory enhanced variant on the min20/20 filtered protocol.
* The strongest Foursquare NYC result is also obtained by the graph-memory enhanced variant.
* On Foursquare Tokyo, the graph-memory enhanced variant gives the best MRR, Acc@5, Acc@10, MAP@5 and MAP@10, while the enhanced Graph-Flashback variant has the highest Acc@1.
* Paper-reference rows are not directly comparable with the Austin subset experiments, because the paper uses the full processed Gowalla dataset and a different protocol.

## Summary files produced by the runners

After a run, the following files are created under `runs/summary/`:

```text
experiment_metrics.csv
common_protocol_results.csv
filtered_protocol_results.csv
foursquare_nyc_results.csv
graph_diagnostics_summary.csv
all_experiments_rebuilt.csv
```

The most useful final table is usually:

```text
runs/summary/all_experiments_rebuilt.csv
```

## References

- Graph-Flashback paper: Graph-Flashback Network for Next Location Recommendation, KDD 2022.
- Official Graph-Flashback implementation: `kevin-xuan/Graph-Flashback`.
- Gowalla source data: SNAP Gowalla location check-ins and friendship network.
- Foursquare NYC/Tokyo public release: TSMC2014 Foursquare check-ins, available as a Kaggle dataset.
