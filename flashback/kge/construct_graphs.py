from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.neighbors import BallTree
from flashback.config import ExperimentConfig, resolve_device, dump_json
from flashback.kge.transe import TransE
from flashback.kge.triplets import TEMPORAL, VISITS, SPATIAL, FRIEND
from flashback.utils import save_sparse, read_table


def _topk_relation(model, heads, targets, relation_id, k, chunk, device, exclude_self=False):
    rows=[]; cols=[]; vals=[]; tgt=model.entity.weight[targets].to(device)
    rel=model.relation.weight[relation_id].to(device)
    with torch.no_grad():
        for start in range(0, len(heads), chunk):
            hids=heads[start:start+chunk]
            q=model.entity.weight[hids].to(device)+rel
            d=torch.cdist(q, tgt, p=1)
            if exclude_self:
                local=torch.arange(len(hids),device=device)
                # heads/targets are aligned contiguous ranges for POIs.
                d[local, torch.as_tensor(hids,device=device)-targets[0]] = float("inf")
            kk=min(k, d.shape[1]); dist, idx=torch.topk(d,kk,largest=False,dim=1)
            score=torch.exp(-dist)
            for i in range(len(hids)):
                rows.extend([start+i]*kk); cols.extend(idx[i].cpu().tolist()); vals.extend(score[i].cpu().tolist())
    return sp.csr_matrix((np.asarray(vals,np.float32),(rows,cols)),shape=(len(heads),len(targets)))


def _spatial_graph(checkins, n_pois, radius_km, topk):
    train=checkins[checkins.split=="train"]
    coords=train.groupby("poi_id")[["latitude","longitude"]].median()
    mat=sp.lil_matrix((n_pois,n_pois),dtype=np.float32)
    if len(coords)>1:
        ids=coords.index.to_numpy(np.int64); rad=np.radians(coords.to_numpy()); tree=BallTree(rad,metric="haversine")
        ind,dist=tree.query_radius(rad,r=radius_km/6371.0088,return_distance=True,sort_results=True)
        for src,js,ds in zip(ids,ind,dist):
            kept=[(ids[j],d*6371.0088) for j,d in zip(js,ds) if ids[j]!=src][:topk]
            for dst,km in kept: mat[int(src),int(dst)]=np.exp(-km/max(radius_km,1e-6))
    return mat.tocsr()


def construct_graphs(cfg: ExperimentConfig, checkins_path: str|Path) -> dict[str,Path]:
    ck=torch.load(cfg.kge.checkpoint,map_location="cpu",weights_only=False); meta=ck["meta"]
    model=TransE(meta["n_entities"],meta["n_relations"],ck["dim"]); model.load_state_dict(ck["state_dict"]); model.eval()
    device=resolve_device(cfg.kge.device); model.to(device)
    U,L=meta["n_users"],meta["n_pois"]; out=Path(cfg.graphs.output_dir); out.mkdir(parents=True,exist_ok=True)
    poi_entities=np.arange(U,U+L,dtype=np.int64); user_entities=np.arange(U,dtype=np.int64)
    transition=_topk_relation(model,poi_entities,poi_entities,TEMPORAL,cfg.graphs.transition_topk,cfg.graphs.score_chunk_size,device,True)
    preference=_topk_relation(model,user_entities,poi_entities,VISITS,cfg.graphs.user_poi_topk,cfg.graphs.score_chunk_size,device,False)
    friends=_topk_relation(model,user_entities,user_entities,FRIEND,cfg.graphs.friend_topk,cfg.graphs.score_chunk_size,device,True)
    # Faithful KGE spatial graph uses the learned SPATIAL relation. A raw
    # geographic graph is also exported for diagnostics/ablations.
    spatial=_topk_relation(model,poi_entities,poi_entities,SPATIAL,cfg.stkg.spatial_topk,cfg.graphs.score_chunk_size,device,True)
    checkins=read_table(checkins_path); spatial_geo=_spatial_graph(checkins,L,cfg.stkg.spatial_radius_km,cfg.stkg.spatial_topk)
    paths={"transition":out/"poi_transition.npz","preference":out/"user_poi_preference.npz","friends":out/"user_friend.npz","spatial":out/"poi_spatial.npz","spatial_geo":out/"poi_spatial_geographic.npz"}
    for name,m in [("transition",transition),("preference",preference),("friends",friends),("spatial",spatial),("spatial_geo",spatial_geo)]: save_sparse(m,paths[name])
    dump_json({k:str(v) for k,v in paths.items()}|{"n_users":U,"n_pois":L},out/"graph_manifest.json")
    return paths
