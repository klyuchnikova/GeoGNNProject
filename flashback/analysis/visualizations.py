from __future__ import annotations
import argparse,json
import os
import tempfile
from pathlib import Path
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "geognn_mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flashback.utils import read_table


def create_visualizations(artifacts_dir="artifacts", processed_dir="data/processed"):
    art=Path(artifacts_dir); fig=art/"figures"; fig.mkdir(parents=True,exist_ok=True)
    ranking_files=list(Path(processed_dir).glob("city_ranking.csv"))
    if ranking_files:
        r=pd.read_csv(ranking_files[0]).sort_values("eligible_users")
        ax=r.plot.barh(x="label",y="eligible_users",legend=False,figsize=(9,5));ax.set_title("Gowalla city selection: eligible users");ax.set_xlabel("Users meeting min-checkins threshold");plt.tight_layout();plt.savefig(fig/"city_eligible_users.png",dpi=180);plt.close()
    result_dir=art/"results"
    histories=list(result_dir.glob("*_history.csv"))
    for path in histories:
        h=pd.read_csv(path)
        ax=h.plot(x="epoch",y=[c for c in ["train_loss","validation_loss"] if c in h],figsize=(8,4));ax.set_title("Graph-Flashback learning curves");plt.tight_layout();plt.savefig(fig/f"{path.stem}_loss.png",dpi=180);plt.close()
        metric_cols=[c for c in h if c in ["validation_Acc@1","validation_Acc@5","validation_Acc@10","validation_MAP@5","validation_MAP@10","validation_MRR"]]
        if metric_cols:
            ax=h.plot(x="epoch",y=metric_cols,figsize=(9,5));ax.set_title("Validation ranking metrics");plt.tight_layout();plt.savefig(fig/f"{path.stem}_metrics.png",dpi=180);plt.close()
    metrics=list(result_dir.glob("*_metrics.json"))
    rows=[]
    for p in metrics:
        data=json.loads(p.read_text()); test=data.get("test",{})
        for k in ["Acc@1","Acc@5","Acc@10","MAP@5","MAP@10","MRR"]:
            if k in test: rows.append({"run":p.stem.replace("_metrics",""),"metric":k,"value":test[k]})
    if rows:
        table=pd.DataFrame(rows); pivot=table.pivot(index="metric",columns="run",values="value")
        ax=pivot.plot.bar(figsize=(10,5));ax.set_ylabel("Score");ax.set_title("Test metrics");plt.xticks(rotation=0);plt.tight_layout();plt.savefig(fig/"test_metrics.png",dpi=180);plt.close();pivot.to_csv(result_dir/"metrics_comparison.csv")
    pred_files=list((art/"predictions").glob("*_test.parquet"))
    for p in pred_files:
        d=read_table(p)
        if not d.empty:
            ax=d["rank"].clip(upper=100).plot.hist(bins=50,figsize=(8,4));ax.set_title("Target rank distribution (clipped at 100)");ax.set_xlabel("Rank");plt.tight_layout();plt.savefig(fig/f"{p.stem}_rank_hist.png",dpi=180);plt.close()
    return sorted(str(p) for p in fig.glob("*.png"))

def main():
    p=argparse.ArgumentParser();p.add_argument("--artifacts-dir",default="artifacts");p.add_argument("--processed-dir",default="data/processed");a=p.parse_args();print("\n".join(create_visualizations(a.artifacts_dir,a.processed_dir)))
if __name__=="__main__":main()
