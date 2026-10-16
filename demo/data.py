import pickle
import numpy as np

RATIOS = (0.60, 0.15, 0.125, 0.125)   # train / val / cal / test
SPLITS = ('train', 'val', 'cal', 'test')


def load_naval(path, seed=42, ratios=RATIOS):
    with open(path, 'rb') as f:
        tr_data, tr_label, va_data, va_label = pickle.load(f)

    tr_data = np.asarray(tr_data).transpose(0, 2, 1).copy()
    va_data = np.asarray(va_data).transpose(0, 2, 1).copy()
    x_min, x_max = float(tr_data[:, 0, :].min()), float(tr_data[:, 0, :].max())
    y_min, y_max = float(tr_data[:, 1, :].min()), float(tr_data[:, 1, :].max())
    for d in (tr_data, va_data):
        d[:, 0, :] = (d[:, 0, :] - x_min) / (x_max - x_min)
        d[:, 1, :] = (d[:, 1, :] - y_min) / (y_max - y_min)

    data = np.concatenate([tr_data, va_data], axis=0)
    label = np.concatenate([np.asarray(tr_label), np.asarray(va_label)], axis=0)

    nclass = 3
    M = -np.ones((nclass, nclass), dtype=np.float64) + 2.0 * np.eye(nclass, dtype=np.float64)

    rng = np.random.default_rng(seed)
    parts = {k: ([], []) for k in SPLITS}
    for c in sorted(np.unique(label).astype(int)):
        idx = np.where(label == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train, n_val, n_cal = int(n * ratios[0]), int(n * ratios[1]), int(n * ratios[2])
        bounds = {'train': idx[:n_train],
                  'val':   idx[n_train:n_train + n_val],
                  'cal':   idx[n_train + n_val:n_train + n_val + n_cal],
                  'test':  idx[n_train + n_val + n_cal:]}
        for k, ik in bounds.items():
            parts[k][0].append(data[ik])
            parts[k][1].append(label[ik])

    splits = {}
    for k in SPLITS:
        X = np.concatenate(parts[k][0], axis=0)
        y = (np.concatenate(parts[k][1], axis=0) - 1).astype(np.int64)
        splits[k] = (X, y)
    norm = {'x': (x_min, x_max), 'y': (y_min, y_max)}
    return splits, M, norm
