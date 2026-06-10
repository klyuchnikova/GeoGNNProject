import numpy as np
import scipy.sparse as sp
import torch

from flashback.config import ExperimentConfig, load_config
from flashback.kge.triplets import _spatial_pairs_rank
from flashback.model.graph_flashback import GraphFlashback


def test_faithful_and_tuned_protocols_are_separate():
    faithful = load_config("configs/gowalla_faithful.yaml")
    tuned = load_config("configs/gowalla_auto.yaml")
    assert faithful.model.rnn == "rnn"
    assert faithful.model.hidden_dim == 10
    assert faithful.data.sequence_mode == "block_all"
    assert faithful.train.bpr_weight == 0
    assert tuned.model.rnn == "gru"
    assert tuned.model.hidden_dim == 128
    assert tuned.data.sequence_mode == "window_last"
    assert tuned.data.min_poi_visits == 10
    assert tuned.train.bpr_weight > 0
    assert tuned.model.personal_prior_weight > 0
    assert tuned.model.recent_prior_weight > 0
    assert tuned.model.geo_prior_weight > 0
    assert tuned.model.category_transition_weight > 0
    assert tuned.model.use_repeat_gate
    assert tuned.model.use_user_context


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
    model = GraphFlashback(1, 3, 1, cfg, transition, preference)
    assert isinstance(model.poi_graph_projection, torch.nn.Identity)
