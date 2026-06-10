from pathlib import Path
import sys, yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from flashback.config import load_config
required=[
 'README.md','experiments/austin_suite.yaml','configs/austin_tuned.yaml','configs/austin_tuned_rank.yaml',
 'configs/austin_faithful.yaml','configs/smoke.yaml','scripts/run_austin_experiments.py',
 'scripts/check_offline_environment.py','scripts/package_results.py','notebooks/kaggle_austin_offline.ipynb'
]
missing=[x for x in required if not (ROOT/x).exists()]
if missing: raise SystemExit(f'Missing project files: {missing}')
for cfg in ['austin_tuned.yaml','austin_tuned_rank.yaml','austin_faithful.yaml','smoke.yaml']:
 load_config(ROOT/'configs'/cfg)
suite=yaml.safe_load((ROOT/'experiments/austin_suite.yaml').read_text())
for profile,names in suite['profiles'].items():
 unknown=[n for n in names if n not in suite['experiments']]
 if unknown: raise SystemExit(f'Unknown experiments in {profile}: {unknown}')
print('Project verification passed')
