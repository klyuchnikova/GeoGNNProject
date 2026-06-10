from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from flashback.config import load_config,resolve_device,seed_everything,dump_json
from flashback.data.sequences import NextPoiSequenceDataset
from flashback.model.graph_flashback import load_model
from flashback.evaluation.evaluator import evaluate
from flashback.utils import write_table
from flashback.popularity_baseline import evaluate_popularity


def _loader(path,split,cfg,shuffle=False):
    ds=NextPoiSequenceDataset.from_parquet(path,split,cfg.data.sequence_length,cfg.data.sequence_stride)
    return DataLoader(ds,batch_size=cfg.train.batch_size,shuffle=shuffle,num_workers=cfg.train.num_workers)

def train_graph_flashback(cfg,checkins_path):
    seed_everything(cfg.train.seed); device=resolve_device(cfg.train.device)
    model=load_model(cfg,checkins_path).to(device); opt=torch.optim.Adam(model.parameters(),lr=cfg.train.learning_rate,weight_decay=cfg.train.weight_decay)
    train_loader=_loader(checkins_path,"train",cfg,True)
    val_split="validation" if cfg.data.val_ratio>0 else "test"; val_loader=_loader(checkins_path,val_split,cfg)
    ckpt_dir=Path(cfg.train.checkpoint_dir); ckpt_dir.mkdir(parents=True,exist_ok=True); ckpt=ckpt_dir/f"{cfg.train.run_name}_best.pt"
    history=[]; best=float("inf"); bad=0
    for epoch in range(1,cfg.train.epochs+1):
        model.train(); total=count=0
        for batch in train_loader:
            batch={k:v.to(device) for k,v in batch.items()}; mask=batch["target_mask"]
            logits,_,_=model(batch["locations"],batch["timestamps"],batch["coordinates"],batch["user_id"],batch["valid_input"])
            loss=torch.nn.functional.cross_entropy(logits[mask],batch["targets"][mask])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.gradient_clip); opt.step()
            total+=float(loss.detach())*int(mask.sum()); count+=int(mask.sum())
        val,_=evaluate(model,val_loader,device); row={"epoch":epoch,"train_loss":total/max(1,count),**{f"validation_{k}":v for k,v in val.items()}}; history.append(row)
        if val["loss"]<best-1e-6:
            best=val["loss"];bad=0;torch.save({"state_dict":model.state_dict(),"config":cfg.to_dict(),"epoch":epoch},ckpt)
        else:
            bad+=1
            if bad>=cfg.train.patience: break
    out=Path(cfg.artifacts_dir)/"results";out.mkdir(parents=True,exist_ok=True);pd.DataFrame(history).to_csv(out/f"{cfg.train.run_name}_history.csv",index=False)
    saved=torch.load(ckpt,map_location=device,weights_only=False);model.load_state_dict(saved["state_dict"])
    results={"validation":evaluate(model,val_loader,device)[0],"best_epoch":saved["epoch"]}
    if cfg.train.evaluate_test_after_training:
        test_loader=_loader(checkins_path,"test",cfg); test,pred=evaluate(model,test_loader,device,True); results["test"]=test
        pred_dir=Path(cfg.artifacts_dir)/"predictions";pred_dir.mkdir(parents=True,exist_ok=True);write_table(pd.DataFrame(pred),pred_dir/f"{cfg.train.run_name}_test.parquet",index=False)
    dump_json(results,out/f"{cfg.train.run_name}_metrics.json")
    # Lightweight sanity baselines on the identical test targets.
    for personal, name in [(False, "global_popularity"), (True, "personal_popularity")]:
        baseline = evaluate_popularity(checkins_path, cfg.data.sequence_length, cfg.data.sequence_stride, cfg.train.batch_size, personal)
        dump_json({"test": baseline}, out/f"{name}_metrics.json")
    return ckpt,results

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkins",default=None);a=p.parse_args();cfg=load_config(a.config)
    checkins=a.checkins or next(Path(cfg.data.output_dir).glob(f"{cfg.data.dataset}_*_checkins.parquet"));train_graph_flashback(cfg,checkins)
if __name__=="__main__":main()
