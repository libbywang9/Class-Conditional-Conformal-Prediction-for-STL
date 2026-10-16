import argparse
import csv
import os
import pickle
from time import time

import torch
import yaml

from demo.conformal import calibrate
from demo.data import load_naval
from demo.formulas import render_formulas
from demo.metrics import metrics
from demo.model import STLNet, seed_all
from demo.stage1 import train_stage1
from demo.stage2 import refine_stage2

HERE = os.path.dirname(os.path.abspath(__file__))


def here(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage1', choices=['pretrained', 'train'], default='pretrained')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--data', default=None, help='naval data file (default: data.path in the config)')
    ap.add_argument('--out', default=None, help='output folder (default: outputs/<stage1 mode>)')
    args = ap.parse_args()

    with open(here(args.config)) as f:
        cfg = yaml.safe_load(f)
    out_dir = here(args.out or os.path.join('outputs', args.stage1))
    os.makedirs(out_dir, exist_ok=True)
    report = []

    def log(s=''):
        print(s, flush=True)
        report.append(s)

    t_start = time()
    torch.set_default_dtype(torch.float64)
    splits, M_np, norm = load_naval(here(args.data or cfg['data']['path']), seed=cfg['seed'],
                                    ratios=tuple(cfg['data']['split_ratios']))
    T = {k: (torch.tensor(X), torch.tensor(y)) for k, (X, y) in splits.items()}
    log('Naval data: ' + ', '.join('{} {}'.format(k, len(y)) for k, (_, y) in splits.items())
        + ';  trajectories {} x {} steps'.format(splits['train'][0].shape[1], splits['train'][0].shape[2]))

    seed_all(cfg['seed'])
    net = STLNet(M_np, splits['train'][0].shape[2], cfg['model'])
    log('\n== Stage 1 ({})'.format(args.stage1))
    if args.stage1 == 'train':
        train_stage1(net, T['train'], T['val'], cfg['stage1'], log)
        with open(here(cfg['pretrained_stage1']), 'rb') as f:
            ref = pickle.load(f)
        same = all(torch.equal(x, y) for x, y in zip(net.state(net.Wd)[:5], ref[:5]))
        log('  trained Stage 1 identical to {}: {}'.format(cfg['pretrained_stage1'], 'yes' if same else 'no'))
    else:
        net.load(here(cfg['pretrained_stage1']))
        log('  loaded ' + cfg['pretrained_stage1'])
    net.save(os.path.join(out_dir, 'stage1.pkl'), net.Wd)

    def evaluate(tag):
        with torch.no_grad():
            R = {k: net.forward(T[k][0], net.Wd).numpy() for k in ('val', 'cal', 'test')}
        m = {k: metrics(R[k], splits[k][1], M_np) for k in ('val', 'test')}
        cp = calibrate(R['cal'], splits['cal'][1], R['test'], splits['test'][1], M_np, cfg['conformal']['alpha'])
        fml = render_formulas(net, T['val'][0], splits['val'][1], norm, cfg['formulas']['active_share'])
        with open(os.path.join(out_dir, 'formulas_{}.txt'.format(tag)), 'w') as f:
            f.write(fml + '\n')
        return m, cp, fml

    m1, cp1, fml1 = evaluate('stage1')

    log('\n== Stage 2 refinement')
    seed_all(cfg['seed'])
    info, Wd = refine_stage2(net, T['train'], T['val'], M_np, cfg['stage2'], log)
    net.save(os.path.join(out_dir, 'stage2_refined.pkl'), Wd)
    with open(os.path.join(out_dir, 'stage2_history.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'val_correct', 'n_val', 'val_acc', 'val_PSR', 'val_CSR', 'val_R_bar_pos',
                    'val_mean_R_satpos', 'passes_gate'])
        for ep, m, _ in info['history']:
            w.writerow([ep, m['correct'], m['n'], m['acc'], m['PSR'], m['CSR'], m['R_bar_pos'],
                        m['mean_R_satpos'], int(m['feasible'])])
    m2, cp2, fml2 = evaluate('refined')
    sel = 'epoch {}'.format(info['selected_epoch']) if info['selected_epoch'] else 'none (Stage 1 kept)'

    log('\n== Classification and robustness (before = Stage 1, after = Stage 2 selected: {})'.format(sel))
    log('{:5s} {:7s} {:>8s} {:>8s} {:>8s} {:>10s} {:>16s}'.format('split', 'model', 'acc', 'PSR', 'CSR', 'R_bar_pos',
                                                                  'mean R sat.pos.'))
    for sp in ('val', 'test'):
        for tag, m in (('before', m1), ('after', m2)):
            log('{:5s} {:7s} {:8.4f} {:8.4f} {:8.4f} {:10.4f} {:16.4f}'.format(
                sp, tag, m[sp]['acc'], m[sp]['PSR'], m[sp]['CSR'], m[sp]['R_bar_pos'], m[sp]['mean_R_satpos']))

    a = cfg['conformal']['alpha']
    log('\n== Conformal calibration (score = decoding loss, alpha = {}, target coverage {:.2f}; cal {} / test {})'.format(
        a, 1 - a, len(splits['cal'][1]), len(splits['test'][1])))
    log('{:7s} {:18s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>7s}  thresholds'.format(
        'model', 'method', 'marginal', 'class 0', 'class 1', 'class 2', 'avg size', 'empty'))
    for tag, cp in (('before', cp1), ('after', cp2)):
        for meth in ('pooled', 'class_conditional'):
            r = cp[meth]
            log('{:7s} {:18s} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:7.4f}  {}'.format(
                tag, meth, r['marginal_cov'], *r['class_cov'], r['avg_size'], r['empty'],
                ' '.join('{:.4f}'.format(q) for q in r['q'])))

    log('\n== Learned formulas after refinement, original coordinates (unpruned; time in steps)')
    log('   x in [{:.2f}, {:.2f}], y in [{:.2f}, {:.2f}] on the original training data'.format(
        *norm['x'], *norm['y']))
    log(fml2)
    log('Stage 1 formulas: {}'.format(os.path.relpath(os.path.join(out_dir, 'formulas_stage1.txt'), HERE)))
    log('\nTotal time {:.1f}s'.format(time() - t_start))
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
