from pathlib import Path
import pandas as pd
from flashback.config import load_config
from flashback.data.preprocessing import prepare_dataset
from flashback.utils import read_table

def test_synthetic_prepare(tmp_path):
    check=tmp_path/"c.tsv";friend=tmp_path/"f.tsv"
    rows=[]
    for u in range(3):
        for i in range(15): rows.append(f"{u}\t2020-01-{1+i//24:02d}T{i%24:02d}:00:00Z\t30.{u}\t-97.{i%3}\t{i%5}\n")
    check.write_text(''.join(rows));friend.write_text("0\t1\n1\t2\n")
    cfg=load_config("configs/gowalla_smoke.yaml");cfg.data.raw_checkins=str(check);cfg.data.raw_friendships=str(friend);cfg.data.output_dir=str(tmp_path/"out");cfg.data.min_checkins=10;cfg.data.max_users=0
    p=prepare_dataset(cfg)
    d=read_table(p.checkins)
    assert set(d.split)=={"train","validation","test"};assert d.user_id.nunique()==3
