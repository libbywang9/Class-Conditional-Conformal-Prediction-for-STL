import numpy as np


def decoding_loss(R, M):
    """ploss[i, c] = sum_k relu(-M[c, k] R[i, k])  (also the conformal score)."""
    return np.stack([np.maximum(0.0, -M[c] * R).sum(1) for c in range(M.shape[0])], 1)


def metrics(R, y, M):
    My = M[y]
    pos = My > 0
    sat = (R > 0) & pos
    pred = decoding_loss(R, M).argmin(1)
    return dict(correct=int((pred == y).sum()), n=len(y),
                acc=float((pred == y).mean()),
                PSR=float(sat.sum() / pos.sum()),
                CSR=float(((R * My) > 0).all(1).mean()),
                R_bar_pos=float(R[pos].mean()),
                mean_R_satpos=float(R[sat].mean()) if sat.any() else float('nan'))
