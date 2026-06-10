from __future__ import annotations
import argparse, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=Path('/kaggle/working/graph_flashback_austin_experiments.zip') if Path('/kaggle/working').exists() else ROOT/'graph_flashback_austin_experiments.zip'
EXCLUDE_SUFFIX={'.pt','.pth','.ckpt','.npz','.npy'}
EXCLUDE_PARTS={'__pycache__','.pytest_cache','processed'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=DEFAULT); args=ap.parse_args()
    include=[ROOT/'runs'/'summary',ROOT/'runs'/'generated_configs',ROOT/'runs'/'experiments']
    files=[]
    for base in include:
        if not base.exists(): continue
        for p in base.rglob('*'):
            if not p.is_file() or p.suffix.lower() in EXCLUDE_SUFFIX: continue
            if any(part in EXCLUDE_PARTS for part in p.parts): continue
            files.append(p)
    if not files: raise RuntimeError('No experiment results found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
        for p in sorted(set(files)): zf.write(p,p.relative_to(ROOT))
    print(args.output)
    print(f'{args.output.stat().st_size/1024**2:.2f} MiB, {len(files)} files')
if __name__=='__main__': main()
