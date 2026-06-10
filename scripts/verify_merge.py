from pathlib import Path

from flashback.config import load_config

required = [
    "GeoGNNProject.pdf",
    "foursquaregraphs-eda.ipynb",
    "gowalla_validated_metadata_eda.ipynb",
    "README.md",
    "LICENSE",
    "flashback/pipeline.py",
    "flashback/model/graph_flashback.py",
    "configs/gowalla_auto.yaml",
    "configs/gowalla_faithful.yaml",
    "configs/gowalla_paper.yaml",
    "configs/gowalla_smoke.yaml",
    "notebooks/kaggle_graph_flashback.ipynb",
    "scripts/install_austin_dataset.py",
]
missing = [path for path in required if not Path(path).exists()]
if missing:
    raise SystemExit("Missing required project files: " + ", ".join(missing))

cfg = load_config("configs/gowalla_auto.yaml")
checks = {
    "prebuilt_austin_input": cfg.data.input_format == "canonical_csv" and cfg.data.preselected_city,
    "rolling_full_context": cfg.data.sequence_mode == "window_last" and cfg.data.sequence_stride == 1,
    "iterative_kcore": cfg.data.min_checkins >= 30 and cfg.data.min_poi_visits >= 10,
    "strong_recurrent_encoder": cfg.model.rnn == "gru" and cfg.model.hidden_dim >= 64,
    "graph_flashback_core": cfg.model.use_transition_graph and cfg.model.use_preference_graph,
    "context_features": cfg.model.use_category_embedding and cfg.model.use_time_embedding,
    "ranking_loss": cfg.train.bpr_weight > 0,
    "repeat_explore_gate": cfg.model.use_repeat_gate and cfg.train.repeat_gate_weight > 0,
    "candidate_priors": all([
        cfg.model.personal_prior_weight > 0,
        cfg.model.recent_prior_weight > 0,
        cfg.model.geo_prior_weight > 0,
        cfg.model.category_transition_weight > 0,
    ]),
    "balanced_kge": cfg.kge.sampling == "relation_balanced",
    "friendship_not_required": not cfg.stkg.include_friendship and cfg.data.raw_friendships is None,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Tuned config checks failed: " + ", ".join(failed))
print("Project verification passed")
