import numpy as np
import torch

from .model import STE

DIR_NAME = {(1.0, 0.0): 'x', (-1.0, 0.0): '-x', (0.0, 1.0): 'y', (0.0, -1.0): '-y'}


def render_formulas(net, Xv, yv, norm, active_share=0.05):
    (xlo, xhi), (ylo, yhi) = norm['x'], norm['y']

    def raw_atom(sym, v):
        if sym == 'x':
            return 'x < {:.2f}'.format(v * (xhi - xlo) + xlo)
        if sym == '-x':
            return 'x > {:.2f}'.format(-v * (xhi - xlo) + xlo)
        if sym == 'y':
            return 'y < {:.2f}'.format(v * (yhi - ylo) + ylo)
        return 'y > {:.2f}'.format(-v * (yhi - ylo) + ylo)

    win = net.time_windows()
    with torch.no_grad():
        R, r2 = net.forward(Xv, return_branches=True)
    R, r2 = R.numpy(), r2.numpy()
    Wcs = STE(net.Wc).detach().numpy()
    a, b = net.a.numpy(), net.b.detach().numpy()

    def node(k):
        lo, hi = win[k] if win[k] is not None else (-1, -1)
        atoms = [raw_atom(DIR_NAME[(float(-a[k][i][0]), float(-a[k][i][1]))], float(-b[k][i][0])) for i in range(4)]
        return '{}_[{},{}]({})'.format(net.Spatial[k], lo, hi, ' AND '.join(atoms))

    out = []
    for c in range(net.ncode):
        mask = yv == c
        share = np.bincount(np.argmax(r2[mask, c, :], 1), minlength=net.f_conj) / mask.sum()
        act = [kb for kb in range(net.f_conj) if share[kb] > active_share and Wcs[c, kb].sum() > 0]
        out.append('class {}  (val: R > 0 on {:.1%} of class-{} samples; active branches: {})'.format(
            c, float((R[mask, c] > 0).mean()), c, ', '.join('b{} {:.1%}'.format(kb, share[kb]) for kb in act)))
        for n_b, kb in enumerate(act):
            sel = np.where(Wcs[c, kb] > 0.5)[0]
            out.append(('     ' if n_b == 0 else '  OR ') + '[b{}] '.format(kb)
                       + '\n          AND '.join(node(k) for k in sel))
        out.append('')
    return '\n'.join(out)
