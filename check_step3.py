"""Independent re-computation of step-3 numbers (per-cell loop, no code shared with decide.py)."""
import math

import pandas as pd

R = pd.read_csv('data/results_step3_grid.csv')


def phi_sf(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


for ds in ['D1', 'D2']:
    P = pd.read_pickle(f'data/preds_step3_{ds}.pkl')
    for m in ['gp', 'xgb']:
        for c_fn, c_fp in [(5, 20), (2, 50), (10, 5)]:
            per_cell = []
            for cell, g in P.groupby('cell_id'):
                n = len(g); tot = dict(rt=0, fn=0, el=0, fp=0, inel=0, cost=0.0)
                for y, mu, sd, s in zip(g.soh_nom, g[m + '_mu'], g[m + '_sd'], g[m + '_s']):
                    p = phi_sf((80 - mu) / (s * sd))
                    costs = {'SL': (1 - p) * c_fp, 'RC': p * c_fn, 'RT': 1.0}
                    a = min(costs, key=lambda k: (costs[k], ['SL', 'RC', 'RT'].index(k)))
                    el = y >= 80
                    tot['el'] += el; tot['inel'] += (not el)
                    if a == 'RT':
                        tot['rt'] += 1; tot['cost'] += 1
                    elif a == 'SL' and not el:
                        tot['fp'] += 1; tot['cost'] += c_fp
                    elif a == 'RC' and el:
                        tot['fn'] += 1; tot['cost'] += c_fn
                per_cell.append({k: v / n for k, v in tot.items()})
            C = pd.DataFrame(per_cell)
            mine = dict(retest_share=C.rt.mean(), FN_rate=C.fn.sum() / C.el.sum(), FP_rate=C.fp.sum() / C.inel.sum(), cost_per_battery=C.cost.mean())
            row = R[(R.data == ds) & (R.model == m) & (R.C_FN == c_fn) & (R.C_FP == c_fp) & (R.policy == 'three-way (ours)')].iloc[0]
            ok = all(abs(mine[k] - row[k]) < 1e-9 for k in mine)
            print(ds, m, c_fn, c_fp, {k: round(v, 4) for k, v in mine.items()}, 'MATCH' if ok else 'MISMATCH')
