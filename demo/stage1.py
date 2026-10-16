from time import time

import torch
import torch.nn.functional as F

from .model import STE, decode_ploss, sample_batch


def train_stage1(net, train, val, cfg, log=print):
    Xtr, ytr = train
    Xv, yv = val
    M = net.M
    nclass = M.shape[0]
    per_class = cfg['batch_per_class']
    delta, rank_margin = cfg['delta'], cfg['rank_margin']

    _ = torch.rand(1)   # one RNG draw kept from the original script so the random stream matches
    opt_wc = torch.optim.Adam([net.Wc], lr=cfg['lr_Wc'])
    opt_b = torch.optim.Adam([net.b], lr=cfg['lr_b'])
    opt_t = torch.optim.Adam([net.t1, net.t2], lr=cfg['lr_t'])
    tl2 = net.length - 1

    class_idx = [(ytr == c).nonzero(as_tuple=True)[0] for c in range(nclass)]
    assert all(len(ci) >= per_class for ci in class_idx)

    best_correct, best_epoch, best_state, no_improve = -1, 0, None, 0
    t0 = time()
    for epoch in range(1, cfg['max_epochs'] + 1):
        idx = sample_batch(class_idx, per_class)
        x, y = Xtr[idx], ytr[idx]
        R = net.forward(x)

        My = M[y]
        pos, neg = (My > 0).double(), (My < 0).double()
        m_k = F.relu(My * R).min(dim=0).values                 # per-code batch min-margin
        loss_pos = (F.relu(m_k[None, :] - R) * pos).sum() / pos.sum()
        loss_neg = (F.relu(m_k[None, :] + R) * neg).sum() / neg.sum()
        margin_reward = -delta * m_k.mean()
        ploss, _ = decode_ploss(R, M)
        n = len(y)
        p_true = ploss[torch.arange(n), y]
        p_inf = ploss.clone(); p_inf[torch.arange(n), y] = float('inf')
        loss_rank = torch.mean(F.relu(rank_margin - (p_inf.min(dim=1).values - p_true)) ** 2)
        loss = loss_pos + loss_neg + margin_reward + loss_rank
        if not torch.isfinite(loss):
            raise SystemExit('Stage 1: non-finite loss at epoch {}'.format(epoch))
        loss.backward()
        opt_wc.step(); opt_wc.zero_grad()
        opt_b.step(); opt_b.zero_grad()
        opt_t.step(); opt_t.zero_grad()

        with torch.no_grad():                                   # projections
            net.Wc[net.Wc <= 0] = 0
            net.Wc[net.Wc >= 1] = 1
            net.Wc.mul_(net.Wc_mask)
            net.t2.clamp_(1, tl2)
            net.t1.clamp_(0, tl2 - 1)
            net.t1.copy_(torch.minimum(net.t1, net.t2 - 1))

        if epoch == 1 or epoch % cfg['eval_every'] == 0:
            with torch.no_grad():
                Rv = net.forward(Xv)
                _, pred = decode_ploss(Rv, M)
                correct = int((pred == yv).sum())
            if correct > best_correct:
                best_correct, best_epoch, no_improve = correct, epoch, 0
                best_state = net.state(net.active_disjuncts(STE(net.Wc)))
                log('  [stage1] epoch {:4d}  val acc {:.4f}  (new best)'.format(epoch, correct / len(yv)))
            else:
                no_improve += 1
                if no_improve >= cfg['early_stop_evals']:
                    log('  [stage1] early stop at epoch {}: {} evaluations without improvement'.format(
                        epoch, cfg['early_stop_evals']))
                    break
    net.load_state(best_state)
    log('  [stage1] best val acc {:.4f} at epoch {}  ({:.1f}s)'.format(best_correct / len(yv), best_epoch, time() - t0))
    return dict(best_epoch=best_epoch, val_correct=best_correct)
