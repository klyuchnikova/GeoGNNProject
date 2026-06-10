import numpy as np
import scipy.sparse as sp
import torch

from flashback.config import ExperimentConfig, load_config
from flashback.kge.triplets import _spatial_pairs_rank
from flashback.model.graph_flashback import GraphFlashback


def test_main_config_uses_ranking_checkpoint_and_faithful_graph_defaults():
    cfg = load_config("configs/gowalla_auto.yaml")
    assert cfg.model.rnn == "rnn"
    assert cfg.model.hidden_dim == 10
    assert cfg.model.lambda_s == 1000.0
    assert not cfg.model.use_spatial_graph
    assert not cfg.model.use_friend_graph
    assert not cfg.model.graph_weight_projection
    assert cfg.train.checkpoint_metric == "MRR"
    assert cfg.data.min_poi_visits == 1


def test_rank_spatial_relation_is_symmetric():
    ids = np.array([0, 1, 2], dtype=np.int64)
    coords = np.radians(np.array([[0.0, 0.0], [0.0, 0.01], [0.0, 0.03]]))
    pairs = set(_spatial_pairs_rank(ids, coords, topk=1, symmetric=True))
    assert (0, 1) in pairs and (1, 0) in pairs


def test_projection_can_be_disabled():
    cfg = ExperimentConfig()
    cfg.model.hidden_dim = 4
    cfg.model.graph_weight_projection = False
    transition = sp.eye(3, format="csr")
    preference = sp.csr_matrix([[1, 0, 0]], dtype="float32")
    model = GraphFlashback(1, 3, cfg, transition, preference)
    assert isinstance(model.poi_graph_projection, torch.nn.Identity)
