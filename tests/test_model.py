import scipy.sparse as sp
import torch
from flashback.config import ExperimentConfig
from flashback.model.graph_flashback import GraphFlashback

def test_forward_shapes():
    cfg=ExperimentConfig();cfg.model.hidden_dim=8;cfg.model.lambda_s=.2;cfg.model.coordinate_distance="haversine_km";cfg.model.use_friend_graph=True
    transition=sp.eye(6,format="csr");preference=sp.csr_matrix([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]],dtype="float32")
    model=GraphFlashback(3,6,cfg,transition,preference,transition,sp.eye(3,format="csr"))
    loc=torch.tensor([[0,1,2,3],[2,3,4,5]])
    ts=torch.tensor([[1.,2.,3.,4.],[1.,2.,3.,4.]])*86400
    co=torch.tensor([[[30.,-97.],[30.1,-97.],[30.2,-97.],[30.3,-97.]],[[30.,-97.],[30.1,-97.],[30.2,-97.],[30.3,-97.]]])
    logits,_,w=model(loc,ts,co,torch.tensor([0,1]),torch.ones(2,4,dtype=torch.bool))
    assert logits.shape==(2,4,6);assert w.shape==(2,4,4)
    assert torch.allclose(w.sum(-1),torch.ones(2,4),atol=1e-5)
