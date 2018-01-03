import numpy as np
from sklearn.metrics import roc_auc_score
from ml_metrics import apk


def user_row_to_scores(row,
                       n_items,
                       item_embs,
                       item_biases,
                       k=10):
    y_t = np.zeros(n_items, dtype=bool)
    y_t[row.pos_inds] = True

    if hasattr(row.factors, '__iter__'):
        y_p = np.dot(row.factors, item_embs)
    else:
        # bypass everything and just score the popularity (item bias) benchmark
        y_p = 0.

    y_p += item_biases
    # ignore user bias for user-wise scoring (just a scalar offset)

    y_p[np.isnan(y_p)] = 0.  # or -inf depending on our view of cold items

    score_auc = auc_via_y(y_t, y_p)

    order = np.argsort(-y_p)

    score_apk = apk_via_inds_orders(row.pos_inds, order, k=k)
    score_mrr = mrr_via_inds_orders(row.pos_inds, order)

    ret_d = {
        'auc': score_auc,
        'apk': score_apk,
        'mrr': score_mrr,
    }
    return ret_d


auc_via_y = roc_auc_score


def apk_via_inds_orders(pos_inds, order, k=10):
    # Ranking for apk
    # Note: Usually we would use
    #   ```
    #   topk_part = np.argpartition(-y_p, k)[:k]  # topk partition
    #   topk = topk_part[np.argsort(-y_p[topk_part])]  # recompute actual
    #   ```
    # which is faster than `np.argsort(-y_p)[:k]`
    # however, we want `order` which can be reused for other metrics (MRR)
    # No need to truncate `order[:k]`
    score_apk = apk(list(pos_inds), order, k=k)
    return score_apk


def mrr_via_inds_orders(pos_inds, order):
    # Note: `scipy.stats.rankdata` could work,
    #   but we already did half the work by calculating `order`
    #   The downside is that that we cannot control ties
    #   But we don't really expect any ties
    ranks = np.argsort(order) + 1
    score_mrr = (1. / ranks[pos_inds]).mean()

    return score_mrr
