import scipy.sparse as sp
import torch

from flashback.config import ExperimentConfig
from flashback.model.graph_flashback import GraphFlashback


def make_model(cfg: ExperimentConfig) -> GraphFlashback:
    transition = sp.eye(6, format="csr")
    preference = sp.csr_matrix(
        [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]],
        dtype="float32",
    )
    return GraphFlashback(
        3,
        6,
        4,
        cfg,
        transition,
        preference,
        global_prior=torch.linspace(0, 1, 6),
        personal_prior_index=torch.tensor([[1, 0], [2, 0], [3, 0]]),
        personal_prior_value=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        poi_coordinates=torch.tensor(
            [[30.0, -97.0], [30.1, -97.0], [30.2, -97.0],
             [30.3, -97.0], [30.4, -97.0], [30.5, -97.0]]
        ),
        poi_categories=torch.tensor([1, 1, 2, 2, 3, 3]),
        category_transition=torch.ones(4, 4) / 4,
        spatial=transition,
        friends=sp.eye(3, format="csr"),
    )


def inputs():
    locations = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]])
    categories = torch.tensor([[1, 1, 2, 2], [1, 2, 3, 3]])
    timestamps = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    ) * 86400
    coordinates = torch.tensor([
        [[30.0, -97.0], [30.1, -97.0], [30.2, -97.0], [30.3, -97.0]],
        [[30.0, -97.0], [30.1, -97.0], [30.2, -97.0], [30.3, -97.0]],
    ])
    return locations, categories, timestamps, coordinates


def test_forward_shapes_and_optional_priors():
    cfg = ExperimentConfig()
    cfg.data.sequence_mode = "block_all"
    cfg.model.hidden_dim = 8
    cfg.model.rnn = "gru"
    cfg.model.lambda_s = 0.2
    cfg.model.coordinate_distance = "haversine_km"
    cfg.model.use_friend_graph = True
    cfg.model.use_category_embedding = True
    cfg.model.use_time_embedding = True
    cfg.model.use_user_context = True
    cfg.model.personal_prior_weight = 2.0
    cfg.model.recent_prior_weight = 1.0
    cfg.model.global_prior_weight = 0.1
    cfg.model.geo_prior_weight = 0.2
    cfg.model.category_transition_weight = 0.2
    cfg.model.use_repeat_gate = True
    model = make_model(cfg)
    locations, categories, timestamps, coordinates = inputs()
    logits, _, auxiliary = model(
        locations,
        timestamps,
        coordinates,
        torch.tensor([0, 1]),
        torch.ones(2, 4, dtype=torch.bool),
        categories,
    )
    assert logits.shape == (2, 4, 6)
    weights = auxiliary["flashback_weights"]
    assert weights.shape == (2, 4, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4), atol=1e-5)
    assert auxiliary["repeat_probability"].shape == (2, 4, 1)


def test_window_mode_returns_one_prediction():
    cfg = ExperimentConfig()
    cfg.data.sequence_mode = "window_last"
    cfg.model.hidden_dim = 4
    model = GraphFlashback(
        1,
        3,
        1,
        cfg,
        sp.eye(3, format="csr"),
        sp.csr_matrix([[1, 0, 0]], dtype="float32"),
    )
    logits, _, auxiliary = model(
        torch.tensor([[0, 1, 2, 1]]),
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.zeros(1, 4, 2),
        torch.tensor([0]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.zeros(1, 4, dtype=torch.long),
    )
    assert logits.shape == (1, 1, 3)
    assert auxiliary["flashback_weights"].shape == (1, 4, 4)


def test_repeat_labels_use_train_prior_or_recent_context():
    cfg = ExperimentConfig()
    cfg.data.sequence_mode = "window_last"
    cfg.model.hidden_dim = 4
    cfg.model.personal_prior_topk = 2
    model = make_model(cfg)
    labels = model.repeat_labels(
        torch.tensor([0, 1]),
        torch.tensor([[4, 5, 4, 2], [0, 1, 4, 5]]),
        torch.tensor([[1], [4]]),
    )
    # user 0 has POI 1 in the train-only personal prior; user 1 has POI 4 in context.
    assert labels.tolist() == [[1.0], [1.0]]
