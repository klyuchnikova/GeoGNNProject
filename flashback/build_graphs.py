import argparse
import os
import pickle

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix

from dataloader import PoiDataloader


def normalize_topk_rows(matrix, topk):
    matrix = matrix.tolil()
    rows, cols = matrix.shape
    for row in range(rows):
        data = matrix.data[row]
        col_idx = matrix.rows[row]
        if not data:
            continue

        order = np.argsort(data)[::-1]
        if topk > 0:
            order = order[:topk]
        selected_cols = [col_idx[i] for i in order]
        selected_data = [float(data[i]) for i in order]
        max_value = max(selected_data)
        if max_value > 0:
            selected_data = [value / max_value for value in selected_data]

        matrix.rows[row] = selected_cols
        matrix.data[row] = selected_data
    return matrix.tocsr()


def build_transition_graph(locs, loc_count, train_ratio, topk):
    graph = lil_matrix((loc_count, loc_count), dtype=np.float32)
    for user_locs in locs:
        train_len = max(2, int(len(user_locs) * train_ratio))
        train_locs = user_locs[:train_len]
        for src, dst in zip(train_locs[:-1], train_locs[1:]):
            if src != dst:
                graph[src, dst] += 1.0
    return normalize_topk_rows(graph, topk)


def build_user_loc_graph(locs, user_count, loc_count, train_ratio, topk):
    graph = lil_matrix((user_count, loc_count), dtype=np.float32)
    for user_id, user_locs in enumerate(locs):
        train_len = max(1, int(len(user_locs) * train_ratio))
        for loc in user_locs[:train_len]:
            graph[user_id, loc] += 1.0
    return normalize_topk_rows(graph, topk)


def build_spatial_graph(poi2gps, loc_count, radius_km, topk):
    coords = np.array([poi2gps[i] for i in range(loc_count)], dtype=np.float64)
    lat = coords[:, 0]
    lng = coords[:, 1]

    mean_lat_rad = np.deg2rad(np.mean(lat))
    xy_km = np.column_stack([
        lng * np.cos(mean_lat_rad) * 111.320,
        lat * 110.574,
    ])

    tree = cKDTree(xy_km)
    graph = lil_matrix((loc_count, loc_count), dtype=np.float32)

    for src in range(loc_count):
        neighbors = tree.query_ball_point(xy_km[src], r=radius_km)
        neighbors = [dst for dst in neighbors if dst != src]
        if not neighbors:
            continue

        distances = np.linalg.norm(xy_km[neighbors] - xy_km[src], axis=1)
        order = np.argsort(distances)
        if topk > 0:
            order = order[:topk]

        for idx in order:
            dst = neighbors[int(idx)]
            dist = max(float(distances[int(idx)]), 1e-6)
            graph[src, dst] = np.exp(-dist / radius_km)

    return normalize_topk_rows(graph, topk)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/checkins-gowalla-austin.txt")
    parser.add_argument("--output-dir", default="data/graphs")
    parser.add_argument("--dataset-name", default="gowalla_austin")
    parser.add_argument("--min-checkins", default=101, type=int)
    parser.add_argument("--max-users", default=0, type=int)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--transition-topk", default=100, type=int)
    parser.add_argument("--user-loc-topk", default=100, type=int)
    parser.add_argument("--spatial-topk", default=50, type=int)
    parser.add_argument("--spatial-radius-km", default=3.0, type=float)
    parser.add_argument("--no-spatial", action="store_true", help="skip spatial POI graph construction")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    loader = PoiDataloader(max_users=args.max_users, min_checkins=args.min_checkins)
    loader.read(args.dataset)

    loc_count = loader.locations()
    user_count = loader.user_count()
    print("Active users:", user_count)
    print("Active POIs:", loc_count)
    print("Check-ins:", loader.checkins_count())

    transition_graph = build_transition_graph(
        loader.locs,
        loc_count=loc_count,
        train_ratio=args.train_ratio,
        topk=args.transition_topk,
    )
    user_loc_graph = build_user_loc_graph(
        loader.locs,
        user_count=user_count,
        loc_count=loc_count,
        train_ratio=args.train_ratio,
        topk=args.user_loc_topk,
    )
    spatial_graph = None
    if not args.no_spatial:
        spatial_graph = build_spatial_graph(
            loader.poi2gps,
            loc_count=loc_count,
            radius_km=args.spatial_radius_km,
            topk=args.spatial_topk,
        )

    transition_path = os.path.join(args.output_dir, f"{args.dataset_name}_transition_top{args.transition_topk}.pkl")
    user_loc_path = os.path.join(args.output_dir, f"{args.dataset_name}_user_loc_top{args.user_loc_topk}.pkl")
    spatial_path = os.path.join(args.output_dir, f"{args.dataset_name}_spatial_top{args.spatial_topk}.pkl")

    with open(transition_path, "wb") as f:
        pickle.dump(transition_graph, f, protocol=2)
    with open(user_loc_path, "wb") as f:
        pickle.dump(user_loc_graph, f, protocol=2)
    if spatial_graph is not None:
        with open(spatial_path, "wb") as f:
            pickle.dump(spatial_graph, f, protocol=2)

    print("Saved transition graph:", transition_path)
    print("Saved user-location graph:", user_loc_path)
    if spatial_graph is not None:
        print("Saved spatial graph:", spatial_path)


if __name__ == "__main__":
    main()
