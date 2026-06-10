from __future__ import annotations
import argparse, json, subprocess, sys, time
from copy import deepcopy
from pathlib import Path
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from flashback.config import load_config

METRICS=["Acc@1","Acc@5","Acc@10","MAP@5","MAP@10","MRR"]

def deep_merge(base, update):
    result=deepcopy(base)
    for key,value in (update or {}).items():
        if isinstance(value,dict) and isinstance(result.get(key),dict): result[key]=deep_merge(result[key],value)
        else: result[key]=value
    return result

def run_cmd(args):
    print("$", " ".join(map(str,args)), flush=True)
    subprocess.run([str(x) for x in args],cwd=ROOT,check=True)

def ready(cfg, stage):
    if stage=='prepare': return any(Path(cfg.data.output_dir).glob('gowalla_*_manifest.json'))
    if stage=='stkg': return (Path(cfg.stkg.output_dir)/'stkg_manifest.json').exists() and (Path(cfg.stkg.output_dir)/'triplets_train.npy').exists()
    if stage=='kge': return Path(cfg.kge.checkpoint).exists() and (Path(cfg.stkg.output_dir)/'transe_diagnostics.json').exists()
    if stage=='graphs': return (Path(cfg.graphs.output_dir)/'graph_manifest.json').exists()
    return False

def generate_config(name, group_cfg_path, overrides):
    raw=yaml.safe_load((ROOT/group_cfg_path).read_text(encoding='utf-8'))
    raw=deep_merge(raw,overrides)
    raw['train']['run_name']=name
    raw['train']['checkpoint_dir']=f'runs/experiments/{name}/checkpoints'
    raw['artifacts_dir']=f'runs/experiments/{name}/artifacts'
    out=ROOT/'runs'/'generated_configs'/f'{name}.yaml'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(yaml.safe_dump(raw,sort_keys=False,allow_unicode=True),encoding='utf-8')
    return out

def collect(suite, selected, status):
    rows=[]
    for name in selected:
        meta=suite['experiments'][name]
        p=ROOT/'runs'/'experiments'/name/'artifacts'/'results'/f'{name}_metrics.json'
        if not p.exists(): continue
        data=json.loads(p.read_text())
        row={'experiment':name,'label':meta.get('label',name),'source':'experiment','seed_group':meta.get('seed_group',''),'best_epoch':data.get('best_epoch')}
        row.update({m:data.get('test',{}).get(m) for m in METRICS})
        rows.append(row)
    # baselines once from first completed experiment
    for name in selected:
        result_dir=ROOT/'runs'/'experiments'/name/'artifacts'/'results'
        if not result_dir.exists(): continue
        for baseline,label in [('global_popularity','Global popularity'),('personal_popularity','Personal popularity')]:
            p=result_dir/f'{baseline}_metrics.json'
            if p.exists():
                d=json.loads(p.read_text()).get('test',{})
                row={'experiment':baseline,'label':label,'source':'baseline','seed_group':'','best_epoch':None}
                row.update({m:d.get(m) for m in METRICS}); rows.append(row)
        break
    for label,vals in suite.get('published_reference',{}).items():
        row={'experiment':label.lower().replace(' ','_'),'label':label,'source':'published_reference','seed_group':'','best_epoch':None}
        row.update({m:vals.get(m) for m in METRICS}); rows.append(row)
    summary=ROOT/'runs'/'summary'; summary.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(summary/'experiment_metrics.csv',index=False)
    (summary/'experiment_metrics.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    seed_rows=[]
    if not frame.empty:
        for group,g in frame[(frame.source=='experiment') & (frame.seed_group!='')].groupby('seed_group'):
            r={'seed_group':group,'runs':len(g)}
            for m in METRICS:
                vals=pd.to_numeric(g[m],errors='coerce')
                r[f'{m}_mean']=float(vals.mean()); r[f'{m}_std']=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
            seed_rows.append(r)
    pd.DataFrame(seed_rows).to_csv(summary/'seed_summary.csv',index=False)
    (summary/'run_status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
    if not frame.empty:
        import matplotlib.pyplot as plt
        plot=frame[frame.source!='published_reference'].dropna(subset=['MRR']).sort_values('MRR')
        if not plot.empty:
            ax=plot.plot.barh(x='label',y='MRR',legend=False,figsize=(10,max(4,0.45*len(plot))))
            ax.set_xlabel('MRR'); ax.set_title('Austin experiment comparison'); plt.tight_layout(); plt.savefig(summary/'mrr_comparison.png',dpi=180); plt.close()
    return frame

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite',default='experiments/austin_suite.yaml')
    ap.add_argument('--profile',choices=['core','full'],default='full')
    ap.add_argument('--force-assets',action='store_true')
    ap.add_argument('--force-train',action='store_true')
    args=ap.parse_args()
    suite=yaml.safe_load((ROOT/args.suite).read_text(encoding='utf-8'))
    selected=suite['profiles'][args.profile]
    status={'profile':args.profile,'started_at':time.strftime('%Y-%m-%d %H:%M:%S'),'experiments':{}}
    # install dataset and check environment; no network calls
    run_cmd([sys.executable,'scripts/check_offline_environment.py'])
    run_cmd([sys.executable,'scripts/install_austin_dataset.py'])
    # build each shared asset group once
    groups=[]
    for name in selected:
        g=suite['experiments'][name]['group']
        if g not in groups: groups.append(g)
    for group in groups:
        base_path=suite['groups'][group]['base_config']
        cfg=load_config(ROOT/base_path)
        for stage in ['prepare','stkg','kge','graphs']:
            if args.force_assets or not ready(cfg,stage):
                run_cmd([sys.executable,'-m','flashback.pipeline','--config',base_path,'--stage',stage])
            else: print(f'[resume] {group}: {stage} already complete')
    # train experiments
    for name in selected:
        meta=suite['experiments'][name]; group=meta['group']; base_path=suite['groups'][group]['base_config']
        generated=generate_config(name,base_path,meta.get('overrides',{}))
        metrics=ROOT/'runs'/'experiments'/name/'artifacts'/'results'/f'{name}_metrics.json'
        started=time.time()
        try:
            if args.force_train or not metrics.exists():
                run_cmd([sys.executable,'-m','flashback.pipeline','--config',generated.relative_to(ROOT),'--stage','train'])
                run_cmd([sys.executable,'-m','flashback.pipeline','--config',generated.relative_to(ROOT),'--stage','analyze'])
            else: print(f'[resume] {name}: metrics already exist')
            status['experiments'][name]={'status':'complete','seconds':round(time.time()-started,2)}
        except Exception as exc:
            status['experiments'][name]={'status':'failed','error':repr(exc),'seconds':round(time.time()-started,2)}
            collect(suite,selected,status)
            raise
        collect(suite,selected,status)
    status['finished_at']=time.strftime('%Y-%m-%d %H:%M:%S')
    frame=collect(suite,selected,status)
    print('\nFinal summary:')
    if not frame.empty:
        print(frame[['label'] + METRICS].to_string(index=False))
    print('\nPackage results with: python scripts/package_results.py')

if __name__ == '__main__':
    main()
