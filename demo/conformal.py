import math

import numpy as np

from .metrics import decoding_loss


def cp_quantile(scores, alpha):
    N = len(scores)
    k = math.ceil((N + 1) * (1.0 - alpha))
    if k > N:
        return np.inf
    return float(np.sort(scores)[k - 1])


def evaluate_sets(scores_test, y_test, thresholds, nclass):
    """thresholds: scalar (pooled) or (nclass,) (class-conditional)."""
    if np.ndim(thresholds) == 0:
        in_set = scores_test <= float(thresholds)
    else:
        in_set = scores_test <= np.asarray(thresholds)[np.newaxis, :]
    covered = in_set[np.arange(len(y_test)), y_test]
    return dict(marginal_cov=float(covered.mean()),
                class_cov=[float(covered[y_test == c].mean()) for c in range(nclass)],
                avg_size=float(in_set.sum(1).mean()),
                empty=float((in_set.sum(1) == 0).mean()))


def calibrate(R_cal, y_cal, R_test, y_test, M, alpha):
    nclass = M.shape[0]
    s_cal, s_test = decoding_loss(R_cal, M), decoding_loss(R_test, M)
    q_pool = cp_quantile(s_cal[np.arange(len(y_cal)), y_cal], alpha)
    q_cls = np.array([cp_quantile(s_cal[y_cal == c, c], alpha) for c in range(nclass)])
    return dict(pooled=dict(q=[q_pool], **evaluate_sets(s_test, y_test, q_pool, nclass)),
                class_conditional=dict(q=q_cls.tolist(), **evaluate_sets(s_test, y_test, q_cls, nclass)))
