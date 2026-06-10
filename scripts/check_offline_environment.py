from __future__ import annotations
import importlib, sys
from pathlib import Path

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "yaml", "tqdm", "matplotlib", "pyarrow", "torch"]
missing=[]
for name in REQUIRED:
    try:
        module=importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'ok')}")
    except Exception as exc:
        missing.append((name,str(exc)))
if missing:
    print("Missing packages:")
    for name,error in missing: print(f" - {name}: {error}")
    raise SystemExit(2)
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
from flashback.config import load_config
load_config(root/'configs'/'austin_tuned.yaml')
print("Offline environment check passed")
