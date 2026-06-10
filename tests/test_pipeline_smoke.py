from pathlib import Path
import subprocess
import sys

from flashback.config import load_config


def test_smoke_assets_can_be_created_and_config_loads():
    subprocess.run([sys.executable, "scripts/make_synthetic_data.py"], check=True, timeout=30)
    cfg = load_config("configs/smoke.yaml")
    assert Path(cfg.data.raw_checkins).exists()
    assert cfg.data.sequence_length == 4
    assert cfg.model.rnn == "gru"
