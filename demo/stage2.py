from time import time

import torch
import torch.nn.functional as F

from .metrics import metrics
from .model import STE, decode_ploss, sample_batch
from .stl_layers import _tightness_function


def refine_stage2(net, train, val, M_np, cfg, log=print):
    Xtr, ytr = train
    Xv, yv = val
    yv_np = yv.numpy()
    M = net.M
    nclass = M.shape[0]

    for p in (net.Wc, net.t1, net.t2):
        p.requires_grad_(False)
    Wd = net.active_disjuncts(STE(net.Wc))
    assert torch.equal(Wd, net.Wd), 'Stage 1 checkpoint Wd inconsistent with its Wc'
    b_s1 = net.b.detach().clone()
    opt = torch.optim.Adam([net.b], lr=cfg['lr'])
    class_idx = [(ytr == c).nonzero(as_tuple=True)[0] for c in range(nclass)]

    with torch.no_grad():
        m0 = metrics(net.forward(Xv, Wd).numpy(), yv_np, M_np)
    floor = m0['correct'] - round(cfg['acc_tolerance'] * m0['n'])
    log('  [stage2] Stage 1 val: acc {:.4f} ({}/{}), PSR {:.4f}; accuracy gate: val_correct >= {}'.format(
        m0['acc'], m0['correct'], m0['n'], m0['PSR'], floor))

    history = []          # (epoch, metrics, b)
    t0 = time()
    for epoch in range(1, cfg['epochs'] + 1):
        idx = sample_batch(class_idx, cfg['batch_per_class'])
        x, y = Xtr[idx], ytr[idx]
        R = net.forward(x, Wd)
        pos = (M[y] > 0).double()
        lt = -(_tightness_function(torch.clamp(R, min=1e-6), beta=cfg['beta']) * pos).sum() / pos.sum()
        ls = (F.relu(cfg['gamma'] - R) ** 2 * pos).sum() / pos.sum()
        ploss, _ = decode_ploss(R, M)
        n = len(y)
        p_true = ploss[torch.arange(n), y]
        p_inf = ploss.clone(); p_inf[torch.arange(n), y] = float('inf')
        lr_ = torch.mean(F.relu(cfg['rank_margin'] - (p_inf.min(dim=1).values - p_true)) ** 2)
        la = torch.mean((net.b - b_s1) ** 2)
        loss = cfg['eta_tight'] * lt + cfg['eta_sign'] * ls + cfg['eta_rank'] * lr_ + cfg['eta_anchor'] * la
        if not torch.isfinite(loss):
            raise SystemExit('Stage 2: non-finite loss at epoch {}'.format(epoch))
        loss.backward()
        opt.step(); opt.zero_grad()

        if epoch == 1 or epoch % cfg['eval_every'] == 0:
            with torch.no_grad():
                m = metrics(net.forward(Xv, Wd).numpy(), yv_np, M_np)
            m['feasible'] = m['correct'] >= floor
            history.append((epoch, m, net.b.detach().clone()))
            if epoch == 1 or epoch % (10 * cfg['eval_every']) == 0:
                log('  [stage2] epoch {:4d}  val acc {:.4f}  PSR {:.4f}  CSR {:.4f}  mean R sat.pos. {:.4f}  {}'.format(
                    epoch, m['acc'], m['PSR'], m['CSR'], m['mean_R_satpos'], 'gate ok' if m['feasible'] else 'below gate'))
    log('  [stage2] {} epochs, {} evaluations ({:.1f}s)'.format(cfg['epochs'], len(history), time() - t0))

    feas = [h for h in history if h[1]['feasible']]
    info = dict(n_evaluated=len(history), n_feasible=len(feas), floor=floor, stage1_val=m0, history=history)
    if not feas:
        with torch.no_grad():
            net.b.copy_(b_s1)
        info.update(selected_epoch=None, n_band=0)
        log('  [stage2] no checkpoint passes the accuracy gate: keeping the Stage 1 model')
        return info, Wd
    r_min = min(h[1]['mean_R_satpos'] for h in feas)
    band = [h for h in feas if h[1]['mean_R_satpos'] <= r_min + cfg['r_band']]
    best = sorted(band, key=lambda h: (-h[1]['PSR'], h[1]['mean_R_satpos'], h[0]))[0]
    with torch.no_grad():
        net.b.copy_(best[2])
    info.update(selected_epoch=best[0], selected_val=best[1], r_min=r_min, n_band=len(band))
    log('  [stage2] gated {}/{}; lowest mean R sat.pos. {:.4f}; band (+{}) {} checkpoints; '
        'selected epoch {} (val acc {:.4f}, PSR {:.4f}, mean R sat.pos. {:.4f})'.format(
            len(feas), len(history), r_min, cfg['r_band'], len(band), best[0],
            best[1]['acc'], best[1]['PSR'], best[1]['mean_R_satpos']))
    return info, Wd
