from __future__ import annotations
import torch
from flashback.evaluation.metrics import RankingAccumulator

@torch.no_grad()
def evaluate(model, loader, device, save_predictions=False):
    model.eval(); acc=RankingAccumulator(); records=[]; loss_sum=count=0
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}
        logits,_,weights=model(batch["locations"],batch["timestamps"],batch["coordinates"],batch["user_id"],batch["valid_input"])
        mask=batch["target_mask"]; flat_logits=logits[mask]; flat_targets=batch["targets"][mask]
        if flat_targets.numel():
            loss=torch.nn.functional.cross_entropy(flat_logits,flat_targets,reduction="sum"); loss_sum+=float(loss); count+=flat_targets.numel()
            acc.update(logits,batch["targets"],mask,batch["user_id"])
            if save_predictions:
                ranks=__import__('flashback.evaluation.metrics',fromlist=['exact_ranks']).exact_ranks(logits,batch["targets"])
                for b,u in enumerate(batch["user_id"].tolist()):
                    for s in torch.where(mask[b])[0].tolist():
                        records.append({"user_id":u,"step":s,"target_poi":int(batch["targets"][b,s]),"rank":int(ranks[b,s]),"top1":int(logits[b,s].argmax())})
    result=acc.compute(); result["loss"]=loss_sum/max(1,count)
    return result,records
