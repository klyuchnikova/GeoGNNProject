# Pipeline architecture

```text
SNAP/TSMC raw files
        |
canonical adapters + city ranking + chronological split
        |
train-only STKG (visits, temporal, spatial, friend)
        |
TransE with relation-aware negative sampling
        |
exact chunked top-k KGE graphs (sparse CSR)
        |
GCN-style graph propagation + RNN/GRU/LSTM
        |
vectorized temporal/spatial/preference flashback
        |
next-POI logits
        |
Acc@1/5/10, MAP@5/10, MRR + diagnostics
```
