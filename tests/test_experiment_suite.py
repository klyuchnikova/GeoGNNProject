from pathlib import Path
import yaml


def test_full_suite_has_report_ready_experiments():
    suite = yaml.safe_load(Path("experiments/austin_suite.yaml").read_text())
    names = set(suite["profiles"]["full"])
    required = {
        "tuned_full_seed42",
        "tuned_full_seed7",
        "tuned_full_seed2026",
        "tuned_no_kge_graphs",
        "tuned_transition_only",
        "tuned_preference_only",
        "tuned_no_priors",
        "tuned_rank_scheme",
        "tuned_with_friendship",
        "tuned_lstm",
        "tuned_rnn",
        "filter_min20_20",
        "filter_kcore20_20",
        "filter_combined10_10",
        "faithful_rank",
    }
    assert required <= names


def test_filter_sweep_matches_gugen_style_names():
    suite = yaml.safe_load(Path("experiments/austin_suite.yaml").read_text())
    names = set(suite["profiles"]["filter_sweep"])
    assert {
        "filter_no_filter_2_2",
        "filter_min10_10",
        "filter_min20_20",
        "filter_kcore10_10",
        "filter_kcore20_20",
        "filter_combined10_10",
    } <= names
