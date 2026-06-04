import argparse
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import PoiDataloader
from dataset import Split


def build_train_popularity(locs, train_ratio):
    counts = Counter()
    for user_locs in locs:
        labels = user_locs[1:]
        inputs = user_locs[:-1]
        train_thr = int(len(inputs) * train_ratio)
        counts.update(labels[:train_thr])
    return [poi for poi, _ in counts.most_common()]


def evaluate_popularity(dataset, dataloader, user_count, ranking, batch_size):
    rank = {poi: i + 1 for i, poi in enumerate(ranking)}

    iter_cnt = 0
    acc1 = 0.0
    acc5 = 0.0
    acc10 = 0.0
    map5 = 0.0
    map10 = 0.0
    mrr = 0.0

    dataset.reset()
    reset_count = torch.zeros(user_count)

    for _, _, _, _, y, _, _, _, reset_h, active_users in dataloader:
        active_users = active_users.squeeze()
        y = y.squeeze()

        for j, reset in enumerate(reset_h):
            if reset:
                reset_count[int(active_users[j].item())] += 1

        for j in range(batch_size):
            user_id = int(active_users[j].item())
            for target in y[:, j]:
                if reset_count[user_id] > 1:
                    continue

                target_rank = rank.get(int(target.item()))
                reciprocal_rank = 0.0 if target_rank is None else 1.0 / target_rank

                iter_cnt += 1
                acc1 += target_rank is not None and target_rank <= 1
                acc5 += target_rank is not None and target_rank <= 5
                acc10 += target_rank is not None and target_rank <= 10
                map5 += reciprocal_rank if target_rank is not None and target_rank <= 5 else 0.0
                map10 += reciprocal_rank if target_rank is not None and target_rank <= 10 else 0.0
                mrr += reciprocal_rank

    if iter_cnt == 0:
        raise RuntimeError("No predictions were evaluated.")

    return {
        "predictions": float(iter_cnt),
        "Acc@1": acc1 / iter_cnt,
        "Acc@5": acc5 / iter_cnt,
        "Acc@10": acc10 / iter_cnt,
        "MAP@5": map5 / iter_cnt,
        "MAP@10": map10 / iter_cnt,
        "MRR": mrr / iter_cnt,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/checkins-gowalla-austin.txt")
    parser.add_argument("--min-checkins", default=101, type=int)
    parser.add_argument("--max-users", default=0, type=int)
    parser.add_argument("--sequence-length", default=20, type=int)
    parser.add_argument("--batch-size", default=200, type=int)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    loader = PoiDataloader(max_users=args.max_users, min_checkins=args.min_checkins)
    loader.read(args.dataset)

    ranking = build_train_popularity(loader.locs, args.train_ratio)
    dataset_test = loader.create_dataset(args.sequence_length, args.batch_size, Split.TEST)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    metrics = evaluate_popularity(
        dataset=dataset_test,
        dataloader=dataloader_test,
        user_count=loader.user_count(),
        ranking=ranking,
        batch_size=args.batch_size,
    )

    print("Baseline: global train popularity")
    print("Active users:", loader.user_count())
    print("Active POIs:", loader.locations())
    print("Check-ins:", loader.checkins_count())
    for key, value in metrics.items():
        if key == "predictions":
            print(f"{key}: {value:.1f}")
        else:
            print(f"{key}: {value:.8f}")


if __name__ == "__main__":
    main()
