import torch
from flashback.evaluation.metrics import exact_ranks,metrics_from_ranks

def test_exact_ranks_and_metrics():
    logits=torch.tensor([[[9.,8.,7.,6.],[1.,2.,3.,4.]]])
    targets=torch.tensor([[0,2]])
    ranks=exact_ranks(logits,targets)
    assert ranks.tolist()==[[1,2]]
    m=metrics_from_ranks(ranks)
    assert m["Acc@1"]==0.5
    assert m["Acc@5"]==1.0
    assert abs(m["MRR"]-0.75)<1e-7

def test_ties_are_deterministic_by_poi_id():
    logits=torch.tensor([[[1.,1.,1.]]])
    targets=torch.tensor([[2]])
    assert exact_ranks(logits,targets).item()==3
