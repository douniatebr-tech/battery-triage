"""Cold start for a new chemistry: is it the number of cells or the number of measurements that matters?

Leave one chemistry out (targets: Dataset 1 and Dataset 2; Dataset 3 has too few cells). The model is
trained on the two other datasets (10 cycles per cell, as in robust.py loco) plus a budget of B labelled
measurements of the new chemistry, taken as m measurements from each of k cells (k * m = B):
    B = 40: 4 x 10, 8 x 5, 20 x 2, 40 x 1      B = 80: 8 x 10, 16 x 5, 40 x 2      plus k = 0.
The m measurements of a cell are drawn at random, one in each of m equal stretches of its life, so that a
single measurement is a random point of the cell's life (as a facility testing each battery once would see).
Stricter than robust.py loco: the target cells contribute only their m labelled measurements, to training
and to the cross-fitted calibration (3 inner folds). In each repeat the 40 candidate cells are drawn at
random and all designs are scored on the same test cells (the target cells not drawn), every cycle,
equal weight per cell. 5 repeats.
Usage: python3 -I -u coldstart.py
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, '.')
from baselines import per_cell_sample  # noqa: E402
from calibrate import scale_factor  # noqa: E402
from robust import COARSE, add_curve, base, gp_model, metrics  # noqa: E402

DESIGNS = [(0, 0), (4, 10), (8, 5), (20, 2), (40, 1), (8, 10), (16, 5), (40, 2)]
POOL, REPS, SRC_N = 40, 5, 10


def sample_life(d, m, rng):
    """m cycles per cell: one random cycle in each of m equal stretches of the cell's life."""
    idx = []
    for _, g in d.sort_values(['cell_id', 'cycle']).groupby('cell_id'):
        edges = np.linspace(0, len(g), m + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            lo = int(np.floor(a)); hi = max(lo + 1, int(np.floor(b)))
            idx.append(g.index[rng.integers(lo, min(hi, len(g)))])
    return d.loc[idx]


def fit(src, tgt_lab, cols, seed):
    """GP on 10 cycles per source cell plus all labelled target measurements; cross-fitted scale factor."""
    tr = pd.concat([src, tgt_lab])
    tr = tr.assign(w=1.0 / tr.groupby('cell_id').cycle.transform('size'))
    oof = []
    for fi, ci in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + 1).split(tr, tr.cond, tr.cell_id):
        f = tr.iloc[fi]
        s = pd.concat([per_cell_sample(f[f.is_src], SRC_N), f[~f.is_src]])
        mu, sd = gp_model(s[cols].to_numpy(), s.soh_nom.to_numpy(), seed)(tr.iloc[ci][cols].to_numpy())
        oof.append(pd.DataFrame({'y': tr.iloc[ci].soh_nom.to_numpy(), 'w': tr.iloc[ci].w.to_numpy(), 'm': mu, 's': sd}))
    oof = pd.concat(oof)
    k = scale_factor(oof['y'].to_numpy(), oof['m'].to_numpy(), oof['s'].to_numpy(), oof['w'].to_numpy())
    s = pd.concat([per_cell_sample(src, SRC_N), tgt_lab])
    return gp_model(s[cols].to_numpy(), s.soh_nom.to_numpy(), seed), k


def main(ok):
    d = base(ok)
    cols = add_curve(d, COARSE)
    d['is_src'] = True
    rows = []
    for tgt in [1, 2]:
        src = d[d.ds != tgt]
        cells = np.array(sorted(d[d.ds == tgt].cell_id.unique()))
        for rep in range(REPS):
            rng = np.random.default_rng(1000 * tgt + rep)
            order = rng.permutation(cells)
            pool, test_cells = order[:POOL], order[POOL:]
            te = d[d.cell_id.isin(test_cells)]
            for kc, m in DESIGNS:
                t0 = time.time()
                lab = sample_life(d[d.cell_id.isin(pool[:kc])], m, np.random.default_rng(rng.integers(1 << 31))) if kc else d.iloc[:0]
                lab = lab.assign(is_src=False)
                pr, kk = fit(src, lab, cols, seed=rep)
                mu, sd = pr(te[cols].to_numpy())
                r = metrics(pd.DataFrame({'soh_nom': te.soh_nom.to_numpy(), 'w': te.w.to_numpy(), 'mu': mu, 'sd': sd, 'k': kk}))
                rows.append({'target': f'D{tgt}', 'rep': rep, 'cells': kc, 'per_cell': m, 'budget': kc * m,
                             'test_cells': len(test_cells), 'scale': kk, **r})
                print(f"D{tgt} rep {rep} {kc:>2} x {m:>2}: MAE {r['MAE']:.2f}, cover {r['cover90_cal']:.3f}, "
                      f"cost {r['cost_ours_5_20']:.3f}, retest {r['retest_ours_5_20']:.3f}; {time.time() - t0:.0f} s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv('data/results_coldstart.csv', index=False)
    S = R.groupby(['target', 'budget', 'cells', 'per_cell'])[['MAE', 'cover90_cal', 'cost_ours_5_20', 'retest_ours_5_20', 'cost_ours_grid_mean']]
    print(S.mean().round(3).to_string())
    print(S.std().round(3).to_string())


if __name__ == '__main__':
    main(pd.read_pickle('data/clean.pkl'))
    print('EXIT OK')
