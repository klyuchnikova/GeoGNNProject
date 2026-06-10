from pathlib import Path
import yaml

def test_full_suite_has_required_ablations():
    suite=yaml.safe_load(Path("experiments/austin_suite.yaml").read_text())
    names=set(suite["profiles"]["full"])
    required={"tuned_full_seed42","tuned_full_seed7","tuned_full_seed2026","tuned_no_kge_graphs","tuned_transition_only","tuned_preference_only","tuned_no_priors","tuned_no_bpr","tuned_rank_scheme","faithful_rank"}
    assert required <= names
