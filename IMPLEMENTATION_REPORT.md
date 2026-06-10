# Implementation report

Implemented:

- canonical Gowalla/Foursquare adapters;
- objective Gowalla metro ranking instead of hard-coded Austin;
- chronological robust and paper-compatible splits;
- train-only STKG with visits, temporal, spatial and friendship relations;
- built-in TransE with relation-aware negative sampling;
- exact chunked KGE graph construction;
- KGE transition, preference, spatial and friend graphs;
- vectorized full Graph-Flashback model;
- corrected friend-graph path;
- Acc@1/5/10, MAP@5/10 and MRR with micro and macro-user aggregation;
- deterministic tie handling;
- global and personal popularity baselines;
- checkpoints, early stopping, gradient clipping and seeds;
- Kaggle downloader/instructions/notebook;
- results analysis notebook and generated plots;
- Foursquare NYC/Tokyo presets;
- synthetic end-to-end smoke test.

Validation performed in the build environment:

```text
python -m pytest -q
8 passed
```

The full multi-million-row Gowalla experiment was not executed in the build environment because it has no outbound Git/GitHub/SNAP access or Kaggle GPU session. The included Kaggle notebook performs that run and writes the real metrics and visualizations. The example output is explicitly synthetic.
