from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
REMOVE=[
 'docs','examples','artifacts','artifacts_faithful','artifacts_paper','checkpoints','checkpoints_faithful','checkpoints_paper',
 'data/synthetic','README_OLD.md','MERGE_INSTRUCTIONS.md','IMPLEMENTATION_REPORT.md','MANIFEST.sha256',
 'configs/gowalla_auto.yaml','configs/gowalla_faithful.yaml','configs/gowalla_paper.yaml','configs/gowalla_smoke.yaml',
 'notebooks/kaggle_graph_flashback.ipynb'
]
for rel in REMOVE:
 p=ROOT/rel
 if p.is_dir(): shutil.rmtree(p,ignore_errors=True)
 elif p.exists(): p.unlink()
for p in ROOT.rglob('__pycache__'): shutil.rmtree(p,ignore_errors=True)
for p in ROOT.rglob('.pytest_cache'): shutil.rmtree(p,ignore_errors=True)
print('Clean update applied; EDA notebooks and GeoGNNProject.pdf were preserved.')
