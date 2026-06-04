import argparse
import os
import pickle

import numpy as np
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

    transition_path = os.path.join(args.output_dir, f"{args.dataset_name}_transition_top{args.transition_topk}.pkl")
    user_loc_path = os.path.join(args.output_dir, f"{args.dataset_name}_user_loc_top{args.user_loc_topk}.pkl")

    with open(transition_path, "wb") as f:
        pickle.dump(transition_graph, f, protocol=2)
    with open(user_loc_path, "wb") as f:
        pickle.dump(user_loc_graph, f, protocol=2)

    print("Saved transition graph:", transition_path)
    print("Saved user-location graph:", user_loc_path)


if __name__ == "__main__":
    main()
