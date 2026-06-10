import torch


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @staticmethod
    def acc_at_k(target, ranking, k):
        return int(target in ranking[:k])

    @staticmethod
    def map_at_k(target, ranking, k):
        pos = (ranking[:k] == target).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            return 0.0
        return 1.0 / (pos.item() + 1)

    @staticmethod
    def mrr(target, ranking):
        pos = (ranking == target).nonzero(as_tuple=True)[0]
        return 1.0 / (pos.item() + 1)

    def evaluate(self, dataloader):
        self.model.eval()

        metrics = {
            "acc1": 0.0,
            "acc5": 0.0,
            "acc10": 0.0,
            "map5": 0.0,
            "map10": 0.0,
            "mrr": 0.0,
        }

        total = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits, targets = self.model.predict_batch(batch)

                # logits: [N, num_pois]
                # targets: [N]
                rankings = torch.argsort(
                    logits,
                    dim=1,
                    descending=True
                )
                for target, ranking in zip(targets, rankings):
                    target = target.item()

                    metrics["acc1"] += self.acc_at_k(target, ranking, 1)
                    metrics["acc5"] += self.acc_at_k(target, ranking, 5)
                    metrics["acc10"] += self.acc_at_k(target, ranking, 10)

                    metrics["map5"] += self.map_at_k(target, ranking, 5)
                    metrics["map10"] += self.map_at_k(target, ranking, 10)

                    metrics["mrr"] += self.mrr(target, ranking)

                    total += 1

        for key in metrics:
            metrics[key] /= total
        return metrics

def print_metrics(metrics):
    print(
        f"{'ACC@1':>10}"
        f"{'ACC@5':>10}"
        f"{'ACC@10':>10}"
        f"{'MAP@5':>10}"
        f"{'MAP@10':>10}"
        f"{'MRR':>10}"
    )

    print(
        f"{metrics['acc1']:10.4f}"
        f"{metrics['acc5']:10.4f}"
        f"{metrics['acc10']:10.4f}"
        f"{metrics['map5']:10.4f}"
        f"{metrics['map10']:10.4f}"
        f"{metrics['mrr']:10.4f}"
    )
