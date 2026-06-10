from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch


def exact_ranks(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """One-based deterministic ranks. Ties are broken by ascending POI id."""
    flat=logits.reshape(-1,logits.shape[-1]); target=targets.reshape(-1)
    target_score=flat.gather(1,target[:,None])
    greater=(flat>target_score).sum(1)
    ids=torch.arange(flat.shape[1],device=flat.device)[None,:]
    tied_before=((flat==target_score)&(ids<target[:,None])).sum(1)
    return (1+greater+tied_before).reshape(targets.shape)


def metrics_from_ranks(ranks: torch.Tensor) -> dict[str,float]:
    r=ranks.float(); n=max(1,r.numel())
    return {"Acc@1":float((r<=1).float().mean()),"Acc@5":float((r<=5).float().mean()),
            "Acc@10":float((r<=10).float().mean()),
            "MAP@5":float(torch.where(r<=5,1/r,torch.zeros_like(r)).mean()),
            "MAP@10":float(torch.where(r<=10,1/r,torch.zeros_like(r)).mean()),
            "MRR":float((1/r).mean()),"prediction_count":int(r.numel())}

class RankingAccumulator:
    def __init__(self): self.ranks=[]; self.by_user=defaultdict(list)
    def update(self, logits, targets, mask, users):
        ranks=exact_ranks(logits,targets); selected=ranks[mask]
        self.ranks.extend(selected.detach().cpu().tolist())
        for b,u in enumerate(users.detach().cpu().tolist()):
            self.by_user[int(u)].extend(ranks[b][mask[b]].detach().cpu().tolist())
    def compute(self):
        if not self.ranks: raise ValueError("No targets accumulated")
        micro=metrics_from_ranks(torch.tensor(self.ranks))
        keys=["Acc@1","Acc@5","Acc@10","MAP@5","MAP@10","MRR"]
        rows=[metrics_from_ranks(torch.tensor(v)) for v in self.by_user.values() if v]
        macro={f"macro_user_{k}":float(np.mean([r[k] for r in rows])) for k in keys}
        return micro|macro|{"user_count":len(rows)}
