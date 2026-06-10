import subprocess,sys
from pathlib import Path

def test_full_smoke_pipeline():
    subprocess.run([sys.executable,"scripts/make_synthetic_data.py"],check=True)
    subprocess.run([sys.executable,"-m","flashback.pipeline","--config","configs/smoke.yaml","--stage","all"],check=True)
    assert Path("data/synthetic/artifacts/results/smoke_metrics.json").exists()
