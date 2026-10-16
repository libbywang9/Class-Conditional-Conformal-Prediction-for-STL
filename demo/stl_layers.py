import torch
import torch.nn as nn
import numpy as np

class STEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        g_clip = torch.clamp(g, min=0, max=1)
        gs = g_clip.clone()
        gs[gs >= 0.5] = 1
        gs[gs < 0.5]  = 0
        return gs

    @staticmethod
    def backward(ctx, grad_output):
        return torch.clone(grad_output)


class Clip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        gs = g.clone()
        gs[gs > 0]  = 1
        gs[gs <= 0] = -1
        return gs

    @staticmethod
    def backward(ctx, grad_output):
        return torch.clone(grad_output)


class Clip_multi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        gs = g.clone()
        gs[gs > 0]  = 10
        gs[gs <= 0] = -10
        return gs

    @staticmethod
    def backward(ctx, grad_output):
        return torch.clone(grad_output)


class RMinTimeWeight(object):
    def __init__(self, tau, t1, t2):
        self.t1   = t1
        self.t2   = t2
        self.tau  = tau
        self.relu = nn.ReLU()

    def get_weight(self, w):
        f1 = (self.relu(w - self.t1 + self.tau) - self.relu(w - self.t1)) / self.tau
        f2 = (self.relu(-w + self.t2 + self.tau) - self.relu(-w + self.t2)) / self.tau
        return torch.min(f1, f2)


class SparseMax(object):
    def __init__(self, beta, a, dim):
        self.beta = beta
        self.a    = a
        self.dim  = dim

    def f(self, r):
        return torch.exp(self.beta * r) - self.a

    def forward(self, r):
        r = self.f(r)
        r_sum = torch.sum(r, dim=self.dim, keepdim=True)
        r_sum[r_sum == 0] = 1
        return torch.div(r, r_sum)


class NormRobust(object):
    def __init__(self, smax, scale):
        self.smax  = smax
        self.scale = scale

    def forward(self, s, r, d):
        r_w  = s * r
        mx   = torch.abs(torch.max(r_w, dim=d, keepdim=True)[0])
        mx[mx == 0] = 1
        r_re = torch.div(r_w, mx)
        if torch.sum(r_w == 0):
            r_re[r_w == 0] = -1e-5
        s_norm = self.smax.forward(r_re)
        return s_norm


class Disjunction(object):
    def __init__(self):
        pass

    def forward(self, X, w):
        s     = torch.clone(X)
        w_sum = w.sum()
        w_norm = w / w_sum if w_sum != 0 else w
        s_norm = self.normalize_robust.forward(w_norm, s, 1)
        sw = torch.sum(torch.mul(s_norm, w_norm), dim=1)
        if torch.any(sw == 0):
            s_norm[sw == 0, :] = 0.1
        denominator = torch.sum(torch.mul(s_norm, w_norm), dim=1, keepdim=True)
        numerator   = torch.sum(torch.mul(torch.mul(s_norm, w_norm), s), dim=1, keepdim=True)
        denominator_old = torch.clone(denominator)
        denominator[denominator_old == 0] = 1
        robust = numerator / denominator
        if torch.sum(denominator_old == 0):
            if torch.all(denominator_old == 0):
                robust[denominator_old == 0] = -1
            else:
                robust[denominator_old == 0] = torch.min(robust[denominator_old != 0])
        return robust

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax            = SparseMax(beta, a, dim)
        self.normalize_robust = NormRobust(self.smax, scale)


class Conjunction(object):
    def __init__(self):
        pass

    def forward(self, X, w):
        s     = torch.clone(-X)
        w_sum = w.sum()
        w_norm = w / w_sum if w_sum != 0 else w
        s_norm = self.normalize_robust.forward(w_norm, s, 1)
        sw = torch.sum(torch.mul(s_norm, w_norm), dim=1)
        if torch.any(sw == 0):
            s_norm[sw == 0, :] = 0.1
        denominator = torch.sum(torch.mul(s_norm, w_norm), dim=1, keepdim=True)
        numerator   = torch.sum(torch.mul(torch.mul(s_norm, w_norm), s), dim=1, keepdim=True)
        denominator_old = torch.clone(denominator)
        numerator_old   = torch.clone(numerator)
        denominator[denominator_old == 0] = 1
        robust = -numerator / denominator
        if torch.sum(denominator_old == 0):
            if torch.all(denominator_old == 0):
                robust[denominator_old == 0] = -1
            else:
                robust[denominator_old == 0] = torch.min(robust[denominator_old != 0])
        return robust

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax            = SparseMax(beta, a, dim)
        self.normalize_robust = NormRobust(self.smax, scale)


# ── TeLEx tightness function (shared by Eventually and Always) ──
def _tightness_function(rp, beta=20):
    """TeLEx tightness: 1/(rp + exp(-beta*rp)) - exp(-rp)"""
    return 1.0 / (rp + torch.exp(-beta * rp)) - torch.exp(-rp)


class Eventually(object):
    def __init__(self, A, b, t1, t2):
        self.A        = A
        self.b        = b
        self.t1       = t1
        self.t2       = t2
        self.duration = t2 - t1 + 1

    def inner_conjunction(self, x):
        mx = torch.abs(torch.max(x, dim=1, keepdim=True)[0])
        mx[mx == 0] = 1
        rre  = torch.div(x, mx)
        if torch.sum(x == 0):
            rre[x == 0] = -1e-5
        rexp  = torch.exp(1.5 * rre)
        r_sum = torch.sum(rexp, dim=1, keepdim=True)
        r_sum[r_sum == 0] = 1
        return torch.div(rexp, r_sum)

    def initialize_robustness(self, x):
        Ar    = self.A.repeat(self.batch_size, 1, 1)
        r_dim = torch.matmul(Ar, x) - self.b
        s     = torch.clone(-r_dim)
        s_norm = self.inner_conjunction(s)
        w      = torch.ones((4, 1))
        w_norm = w / w.sum()
        denominator = torch.sum(torch.mul(s_norm, w_norm), dim=1, keepdim=True)
        numerator   = torch.sum(torch.mul(torch.mul(s_norm, w_norm), s), dim=1, keepdim=True)
        r = -numerator / denominator
        return r

    def robustness(self, X, w, t1, t2):
        r     = X[:, :, t1:t2 + 1]
        s     = torch.clone(r)
        w_sum = w.sum()
        w_norm = w / w_sum if w_sum != 0 else w
        s_norm = self.normalize_robust.forward(w_norm, s, 2)
        sw = torch.sum(torch.mul(s_norm, w_norm), dim=2)
        if torch.any(sw == 0):
            s_norm[sw == 0, :] = 0.1
        denominator = torch.sum(torch.mul(s_norm, w_norm), dim=2, keepdim=True)
        numerator   = torch.sum(torch.mul(torch.mul(s_norm, w_norm), s), dim=2, keepdim=True)
        denominator_old = torch.clone(denominator)
        denominator[denominator_old == 0] = 1
        robust = numerator / denominator
        if torch.sum(denominator_old == 0):
            if torch.all(denominator_old == 0):
                robust[denominator_old == 0] = -1
            else:
                robust[denominator_old == 0] = torch.min(robust[denominator_old != 0])
        return robust

    def robustness_trace(self, data, w, batch_size, need_trace, use_tightness=True):
        self.batch_size = batch_size
        self.data_dim   = data.shape[1]
        X = self.initialize_robustness(data)

        # ── optional TeLEx tightness on the spatial robustness (skipped when use_tightness=False) ──
        if use_tightness:
            X = _tightness_function(X)
        # ─────────────────────────────────────────────────────────────────

        trace = self.robustness(X, w, self.t1, self.t2)
        return trace

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax            = SparseMax(beta, a, dim)
        self.normalize_robust = NormRobust(self.smax, scale)


class Always(object):
    def __init__(self, A, b, t1, t2):
        self.A        = A
        self.b        = b
        self.t1       = t1
        self.t2       = t2
        self.duration = t2 - t1 + 1

    def inner_conjunction(self, x):
        mx = torch.abs(torch.max(x, dim=1, keepdim=True)[0])
        mx[mx == 0] = 1
        rre  = torch.div(x, mx)
        if torch.sum(x == 0):
            rre[x == 0] = -1e-5
        rexp  = torch.exp(1.5 * rre)
        r_sum = torch.sum(rexp, dim=1, keepdim=True)
        r_sum[r_sum == 0] = 1
        return torch.div(rexp, r_sum)

    def initialize_robustness(self, x):
        Ar    = self.A.repeat(self.batch_size, 1, 1)
        r_dim = torch.matmul(Ar, x) - self.b
        s     = torch.clone(-r_dim)
        s_norm = self.inner_conjunction(s)
        w      = torch.ones((4, 1))
        w_norm = w / w.sum()
        denominator = torch.sum(torch.mul(s_norm, w_norm), dim=1, keepdim=True)
        numerator   = torch.sum(torch.mul(torch.mul(s_norm, w_norm), s), dim=1, keepdim=True)
        r = -numerator / denominator
        return r

    def robustness(self, X, w, t1, t2):
        r     = X[:, :, t1:t2 + 1]
        s     = torch.clone(-r)
        w_sum = w.sum()
        w_norm = w / w_sum if w_sum != 0 else w
        s_norm = self.normalize_robust.forward(w_norm, s, 2)
        sw = torch.sum(torch.mul(s_norm, w_norm), dim=2)
        if torch.any(sw == 0):
            s_norm[sw == 0, :] = 0.1
        denominator = torch.sum(torch.mul(s_norm, w_norm), dim=2, keepdim=True)
        numerator   = torch.sum(torch.mul(torch.mul(s_norm, w_norm), s), dim=2, keepdim=True)
        denominator_old = torch.clone(denominator)
        denominator[denominator_old == 0] = 1
        robust = -numerator / denominator
        if torch.sum(denominator_old == 0):
            if torch.all(denominator_old == 0):
                robust[denominator_old == 0] = -1
            else:
                robust[denominator_old == 0] = torch.min(robust[denominator_old != 0])
        return robust

    def robustness_trace(self, data, w, batch_size, need_trace, use_tightness=True):
        self.batch_size = batch_size
        self.data_dim   = data.shape[1]
        X = self.initialize_robustness(data)

        # ── optional TeLEx tightness on the spatial robustness (skipped when use_tightness=False) ──
        if use_tightness:
            X = _tightness_function(X)
        # ─────────────────────────────────────────────────────────────────

        trace = self.robustness(X, w, self.t1, self.t2)
        return trace

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax            = SparseMax(beta, a, dim)
        self.normalize_robust = NormRobust(self.smax, scale)
