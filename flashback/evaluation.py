import torch
import numpy as np
from utils import log_string


class Evaluation:
    """
    Handles evaluation on a given POI dataset and loader.

    The model predicts a sequence of next locations determined by sequence_length
    in one pass. During evaluation each sequence entry is treated as one next-POI
    prediction over the full POI vocabulary.

    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.

    Using the --report_user argument one can access the statistics per user.
    """

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting, log):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting
        self._log = log

    def evaluate(self):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)

        with torch.no_grad():
            iter_cnt = 0
            acc1 = 0.
            acc5 = 0.
            acc10 = 0.
            map5 = 0.
            map10 = 0.
            mrr = 0.

            u_iter_cnt = np.zeros(self.user_count)
            u_acc1 = np.zeros(self.user_count)
            u_acc5 = np.zeros(self.user_count)
            u_acc10 = np.zeros(self.user_count)
            u_map5 = np.zeros(self.user_count)
            u_map10 = np.zeros(self.user_count)
            u_mrr = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)

            for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1

                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                t_slot = t_slot.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)

                y = y.squeeze()
                y_t = y_t.squeeze().to(self.setting.device)
                y_t_slot = y_t_slot.squeeze().to(self.setting.device)
                y_s = y_s.squeeze().to(self.setting.device)
                active_users_device = active_users.to(self.setting.device)

                # evaluate:
                out, h = self.trainer.evaluate(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users_device)

                for j in range(self.setting.batch_size):
                    # o contains a per user list of votes for all locations for each sequence entry
                    o = out[j]

                    o_n = o.cpu().detach().numpy()
                    y_j = y[:, j]
                    user_id = int(active_users[j].item())

                    for k in range(len(y_j)):
                        if reset_count[user_id] > 1:
                            continue  # skip already evaluated users.

                        target = int(y_j[k].item())
                        scores = o_n[k, :]
                        target_score = scores[target]
                        rank = int(np.count_nonzero(scores > target_score) + 1)
                        reciprocal_rank = 1.0 / rank

                        # store
                        u_iter_cnt[user_id] += 1
                        u_acc1[user_id] += rank <= 1
                        u_acc5[user_id] += rank <= 5
                        u_acc10[user_id] += rank <= 10
                        u_map5[user_id] += reciprocal_rank if rank <= 5 else 0.
                        u_map10[user_id] += reciprocal_rank if rank <= 10 else 0.
                        u_mrr[user_id] += reciprocal_rank

            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                acc1 += u_acc1[j]
                acc5 += u_acc5[j]
                acc10 += u_acc10[j]
                map5 += u_map5[j]
                map10 += u_map10[j]
                mrr += u_mrr[j]

                if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0 and u_iter_cnt[j] > 0:
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'Acc@1',
                          formatter.format(u_acc1[j] / u_iter_cnt[j]), 'MRR',
                          formatter.format(u_mrr[j] / u_iter_cnt[j]), sep='\t')

            if iter_cnt == 0:
                log_string(self._log, 'No predictions were evaluated.')
                return

            log_string(self._log, 'Acc@1: ' + formatter.format(acc1 / iter_cnt))
            log_string(self._log, 'Acc@5: ' + formatter.format(acc5 / iter_cnt))
            log_string(self._log, 'Acc@10: ' + formatter.format(acc10 / iter_cnt))
            log_string(self._log, 'MAP@5: ' + formatter.format(map5 / iter_cnt))
            log_string(self._log, 'MAP@10: ' + formatter.format(map10 / iter_cnt))
            log_string(self._log, 'MRR: ' + formatter.format(mrr / iter_cnt))
            print('predictions:', iter_cnt)
