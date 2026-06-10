import torch
from flashback.kge.transe import TransE

def test_transe_shapes_and_normalization():
    m=TransE(10,4,8)
    t=torch.tensor([[0,0,5],[1,3,2]])
    assert m(t).shape==(2,)
    norms=m.entity.weight.norm(dim=1)
    assert torch.allclose(norms,torch.ones_like(norms),atol=1e-5)
