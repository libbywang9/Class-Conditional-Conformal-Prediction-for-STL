import pickle
import random as rd

import numpy as np
import torch
import torch.nn.functional as F

from .stl_layers import (Always, Conjunction, Disjunction, Eventually, RMinTimeWeight,
                         STEstimator)

torch.set_default_dtype(torch.float64)
STE = STEstimator.apply


def seed_all(seed):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class STLNet:
    def __init__(self, M, length, cfg):
        """Random initialisation; draws from the torch RNG in a fixed order (t1, t2, Wc, b)."""
        self.M = torch.tensor(M, dtype=torch.float64)
        self.length = length
        self.ncode = self.M.shape[1]
        self.fn_per = cfg['fn_per']
        self.f_conj = cfg['f_conj']
        self.fn = self.ncode * self.fn_per
        self.sm = cfg['sparsemax']
        fn, ncode, f_conj = self.fn, self.ncode, self.f_conj

        self.t1 = torch.randint(0, length - 1, (fn, 1), dtype=torch.float64, requires_grad=True)
        self.t2 = torch.randint(0, length - 1, (fn, 1), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            self.t2.clamp_(1, length - 1)
            self.t1.clamp_(0, length - 2)
            self.t1.copy_(torch.minimum(self.t1, self.t2 - 1))

        self.Wc = torch.rand((ncode, f_conj, fn), requires_grad=True)
        self.Wd = torch.ones((ncode, f_conj), requires_grad=False)

        # each code bit uses its own block of predicates
        self.Wc_mask = torch.zeros((ncode, f_conj, fn), dtype=torch.float64)
        for j in range(ncode):
            self.Wc_mask[j, :, j * self.fn_per:(j + 1) * self.fn_per] = 1.0
        with torch.no_grad():
            self.Wc.mul_(self.Wc_mask)

        # axis-aligned box faces: x, -x, y, -y
        an = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.a = torch.tensor(np.tile(an, (fn, 1, 1)), dtype=torch.float64)
        self.b = torch.randint(-10, 10, (fn, 4, 1), dtype=torch.float64, requires_grad=True)

        self.Spatial = []
        for _c in range(ncode):
            self.Spatial += ['F'] * (self.fn_per // 2) + ['G'] * (self.fn_per // 2)
        self._build_operators()

    def _build_operators(self):
        tl1, tl2 = 0, self.length - 1
        self.W = RMinTimeWeight(torch.tensor(1), self.t1, self.t2)
        self.W1 = torch.tensor(range(self.length))
        self.Formula = []
        for j, sp in enumerate(self.Spatial):
            fo = (Eventually if sp == 'F' else Always)(self.a[j], self.b[j], tl1, tl2)
            fo.init_sparsemax(self.sm['predicate'], 0, 1, 2)
            self.Formula.append(fo)
        self.conjunc = Conjunction()
        self.conjunc.init_sparsemax(self.sm['conjunction'], 0, 1, 1)
        self.disjunc = Disjunction()
        self.disjunc.init_sparsemax(self.sm['disjunction'], 0, 1, 1)

    # ---- checkpoints -------------------------------------------------------------
    def state(self, Wd):
        return [self.Wc.detach().clone(), self.b.detach().clone(), self.t1.detach().clone(),
                self.t2.detach().clone(), Wd.detach().clone(), self.a, self.Spatial, self.M]

    def load_state(self, st):
        Wc, b, t1, t2, Wd, a, Spatial, M = st
        assert torch.allclose(a, self.a), 'checkpoint: predicate directions differ'
        assert torch.allclose(M, self.M), 'checkpoint: code matrix differs'
        assert list(Spatial) == self.Spatial, 'checkpoint: temporal operators differ'
        with torch.no_grad():
            self.Wc.copy_(Wc); self.b.copy_(b); self.t1.copy_(t1); self.t2.copy_(t2); self.Wd.copy_(Wd)

    def save(self, path, Wd):
        with open(path, 'wb') as f:
            pickle.dump(self.state(Wd), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.load_state(pickle.load(f))

    # ---- forward -----------------------------------------------------------------
    def active_disjuncts(self, Wcs):
        """Wd: a conjunction enters the disjunction iff it selects at least one predicate."""
        return (Wcs.sum(dim=2) != 0).double()

    def forward(self, X, Wd=None, return_branches=False):
        """X: (N, 2, length) -> R (N, ncode); optionally also branch robustness (N, ncode, f_conj)."""
        N = X.shape[0]
        W1s = STE(self.W.get_weight(self.W1))
        r1o = torch.empty((N, self.fn, 1))
        for k, fo in enumerate(self.Formula):
            r1o[:, k, :] = fo.robustness_trace(X, W1s[k, :], N, need_trace=False,
                                               use_tightness=False)[:, 0]
        Wcs = STE(self.Wc)
        if Wd is None:
            Wd = self.active_disjuncts(Wcs)
        r2i = torch.squeeze(r1o)
        r2o = torch.empty((N, self.f_conj))
        r2 = torch.empty((N, self.ncode, self.f_conj))
        for j in range(self.ncode):
            for k in range(self.f_conj):
                r2o[:, k] = self.conjunc.forward(r2i, Wcs[j, k, :])[:, 0]
            r2[:, j, :] = r2o
        R = torch.empty((N, self.ncode))
        for j in range(self.ncode):
            R[:, j] = self.disjunc.forward(r2[:, j, :], Wd[j, :])[:, 0]
        return (R, r2) if return_branches else R

    def time_windows(self):
        """Per-predicate [lo, hi] time window from the network's own STE time mask (None if empty)."""
        W1s = STE(self.W.get_weight(self.W1)).detach().numpy()
        out = []
        for k in range(self.fn):
            idx = np.where(W1s[k] > 0.5)[0]
            if idx.size == 0:
                out.append(None)
                continue
            assert idx.size == idx.max() - idx.min() + 1, 'non-contiguous time mask, predicate {}'.format(k)
            out.append((int(idx.min()), int(idx.max())))
        return out


def decode_ploss(R, M, eps_dec=0.0):
    """Decoding loss ploss[i, c] = sum_j relu(eps_dec - M[c, j] R[i, j]); prediction = argmin_c."""
    ploss = torch.stack([torch.sum(F.relu(eps_dec - M[i, :] * R), dim=1) for i in range(M.shape[0])], dim=1)
    return ploss, torch.argmin(ploss, dim=1)


def sample_batch(class_idx, per_class):
    """per_class random training indices from every class, shuffled."""
    idx = torch.cat([ci[torch.tensor(rd.sample(range(len(ci)), per_class))] for ci in class_idx])
    return idx[torch.randperm(len(idx))]
