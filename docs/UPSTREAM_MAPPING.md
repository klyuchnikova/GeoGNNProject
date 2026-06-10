# Mapping to the original Graph-Flashback repository

The implementation is a cleaned, modern PyTorch reimplementation of the original repository rather than a simplified count-graph substitute.

| Original concept | This project |
|---|---|
| `dataloader.py`, `dataset.py` | `flashback/data/adapters.py`, `preprocessing.py`, `sequences.py` |
| `KGE/generate_triplet.py` | `flashback/kge/triplets.py` |
| external KGE training | `flashback/kge/transe.py`, `train.py` |
| KGE graph construction scripts | `flashback/kge/construct_graphs.py` |
| `network.py` | `flashback/model/graph_flashback.py` |
| `trainer.py`, `evaluation.py` | `flashback/train.py`, `flashback/evaluation/*` |

Deliberate corrections:

- train-only graph construction;
- validation checkpoint instead of repeated test selection;
- fixed social graph path;
- deterministic tie handling;
- sparse graph storage;
- vectorized but formula-equivalent flashback weights;
- configurable Gowalla/Foursquare adapters;
- reproducible seeds, clipping, checkpoints and tests.
