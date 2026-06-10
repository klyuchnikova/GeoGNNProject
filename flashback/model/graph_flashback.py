from __future__ import annotations
import math
from pathlib import Path
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from flashback.config import ExperimentConfig
from flashback.utils import load_sparse, random_walk_normalize, scipy_to_torch_sparse, read_table


def _normalized_with_self(matrix: sp.spmatrix, self_weight: float, edge_weight: float) -> sp.csr_matrix:
    return random_walk_normalize(matrix.astype("float32") * edge_weight + sp.eye(matrix.shape[0], dtype="float32") * self_weight)


def _haversine_pairwise(coords: torch.Tensor) -> torch.Tensor:
    # coords [B,S,2] degrees, output km [B,S,S]
    rad=torch.deg2rad(coords); lat=rad[...,0]; lon=rad[...,1]
    dlat=lat[:,:,None]-lat[:,None,:]; dlon=lon[:,:,None]-lon[:,None,:]
    a=torch.sin(dlat/2)**2 + torch.cos(lat[:,:,None])*torch.cos(lat[:,None,:])*torch.sin(dlon/2)**2
    return 6371.0088 * 2 * torch.asin(torch.sqrt(torch.clamp(a,0,1)))


class GraphFlashback(nn.Module):
    """Graph-Flashback with STKG-derived graphs and vectorized flashback weights."""
    def __init__(self, n_users: int, n_pois: int, cfg: ExperimentConfig,
                 transition: sp.spmatrix, preference: sp.spmatrix,
                 spatial: sp.spmatrix | None = None, friends: sp.spmatrix | None = None):
        super().__init__(); self.cfg=cfg; h=cfg.model.hidden_dim; self.n_pois=n_pois
        self.poi_embedding=nn.Embedding(n_pois,h); self.user_embedding=nn.Embedding(n_users,h)
        self.poi_graph_linear=nn.Linear(h,h,bias=False); self.spatial_graph_linear=nn.Linear(h,h,bias=False)
        self.user_graph_linear=nn.Linear(h,h,bias=False)
        rnn_cls={"rnn":nn.RNN,"gru":nn.GRU,"lstm":nn.LSTM}[cfg.model.rnn]
        self.rnn=rnn_cls(h,h,batch_first=True); self.dropout=nn.Dropout(cfg.model.dropout)
        self.output=nn.Linear(2*h,n_pois)
        self.register_buffer("transition", scipy_to_torch_sparse(_normalized_with_self(transition,1.0,cfg.model.lambda_loc)))
        self.register_buffer("preference", scipy_to_torch_sparse(random_walk_normalize(preference)))
        self.spatial_enabled=spatial is not None and cfg.model.use_spatial_graph
        self.friend_enabled=friends is not None and cfg.model.use_friend_graph
        if self.spatial_enabled:
            self.register_buffer("spatial",scipy_to_torch_sparse(_normalized_with_self(spatial,1.0,cfg.model.lambda_loc)))
        if self.friend_enabled:
            self.register_buffer("friends",scipy_to_torch_sparse(_normalized_with_self(friends,1.0,cfg.model.lambda_user)))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.poi_embedding.weight); nn.init.xavier_uniform_(self.user_embedding.weight)
    def graph_embeddings(self):
        p=self.poi_graph_linear(torch.sparse.mm(self.transition,self.poi_embedding.weight))
        if self.spatial_enabled:
            p=0.5*(p+self.spatial_graph_linear(torch.sparse.mm(self.spatial,self.poi_embedding.weight)))
        u=self.user_embedding.weight
        if self.friend_enabled: u=0.5*(u+self.user_graph_linear(torch.sparse.mm(self.friends,u)))
        return p,u
    def forward(self, locations, timestamps, coordinates, user_id, valid_input=None, hidden=None):
        p_all,u_all=self.graph_embeddings(); x=p_all[locations]
        timestamps=timestamps.to(dtype=x.dtype); coordinates=coordinates.to(dtype=x.dtype)
        recurrent,hidden=self.rnn(self.dropout(x),hidden)
        B,S,H=recurrent.shape
        dt=torch.clamp(timestamps[:,:,None]-timestamps[:,None,:],min=0.0)
        day=dt/86400.0
        temporal=0.5*(torch.cos(2*math.pi*day)+1.0)*torch.exp(-self.cfg.model.lambda_t*day)
        if self.cfg.model.coordinate_distance == "haversine_km": dist=_haversine_pairwise(coordinates)
        else: dist=torch.cdist(coordinates,coordinates,p=2)
        spatial=torch.exp(-self.cfg.model.lambda_s*dist)
        pref=torch.sparse.mm(self.preference,p_all)  # [U,H]
        similarity=torch.exp(-torch.linalg.vector_norm(x-pref[user_id][:,None,:],ord=2,dim=-1))
        causal=torch.tril(torch.ones(S,S,device=x.device,dtype=torch.bool))[None,:,:]
        weights=temporal*spatial*similarity[:,None,:]
        if valid_input is not None: weights=weights*valid_input[:,None,:]
        weights=weights*causal
        weights=weights/(weights.sum(dim=-1,keepdim=True)+1e-12)
        flash=torch.bmm(weights,recurrent)
        user=u_all[user_id][:,None,:].expand(-1,S,-1)
        logits=self.output(self.dropout(torch.cat([flash,user],dim=-1)))
        return logits,hidden,weights


def load_model(cfg: ExperimentConfig, checkins_path: str|Path, graph_dir: str|Path | None=None) -> GraphFlashback:
    import pandas as pd
    graph_dir=Path(graph_dir or cfg.graphs.output_dir); frame=read_table(checkins_path)
    return GraphFlashback(int(frame.user_id.max())+1,int(frame.poi_id.max())+1,cfg,
        load_sparse(graph_dir/"poi_transition.npz"),load_sparse(graph_dir/"user_poi_preference.npz"),
        load_sparse(graph_dir/"poi_spatial.npz") if (graph_dir/"poi_spatial.npz").exists() else None,
        load_sparse(graph_dir/"user_friend.npz") if (graph_dir/"user_friend.npz").exists() else None)
