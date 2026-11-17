"""Step 3b: expected-cost decisions on calibrated predictions.

Costs are in units of one full capacity test (C_T = 1). For a battery with
p = P(SoH >= 80%):  second life costs (1 - p) * C_FP, recycling costs p * C_FN,
retest costs C_T (the measured SoH then sets the route). Realised cost uses the
true SoH. Every metric weights each cell equally (D-14).
Usage: python3 -I decide.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

THRESH, Z90 = 80.0, 1.6449
SL, RC, RT = 0, 1, 2  # second life, recycle, retest


def prob_eligible(mu, sd, s):
    return norm.sf(THRESH, loc=mu, scale=s * sd)


def policies(mu, p, c_fn, c_fp):
    exp_sl, exp_rc = (1 - p) * c_fp, p * c_fn
    return {
        'point': np.where(mu >= THRESH, SL, RC),
        'two-way by cost': np.where(exp_sl <= exp_rc, SL, RC),
        'screen then test': np.where(exp_rc <= 1.0, RC, RT),  # never straight to second life
        'three-way (ours)': np.argmin(np.column_stack([exp_sl, exp_rc, np.ones_like(p)]), axis=1),
        'retest all': np.full(len(p), RT),
    }


def evaluate(action, y, w, c_fn, c_fp):
    w = w / w.sum()
    elig = y >= THRESH
    cost = np.where(action == RT, 1.0, np.where((action == SL) & ~elig, c_fp, np.where((action == RC) & elig, c_fn, 0.0)))
    return {
        'retest_share': float(np.sum(w * (action == RT))),
        'FN_rate': float(np.sum(w * (elig & (action == RC))) / np.sum(w * elig)),
        'FP_rate': float(np.sum(w * (~elig & (action == SL))) / np.sum(w * ~elig)),
        'cost_per_battery': float(np.sum(w * cost)),
    }


def calibration(P, m):
    w = P.w / P.w.sum()
    y = P.soh_nom
    out = {}
    for name, s in [('raw', 1.0), ('calibrated', P[m + '_s'])]:
        half = Z90 * s * P[m + '_sd']
        out[f'cover90_{name}'] = float(np.sum(w * ((y - P[m + '_mu']).abs() <= half)))
        out[f'width90_{name}'] = float(np.sum(w * 2 * half))
        p = prob_eligible(P[m + '_mu'], P[m + '_sd'], s)
        out[f'brier_{name}'] = float(np.sum(w * (p - (y >= THRESH)) ** 2))
    out['scale_by_fold'] = [round(float(v), 3) for v in P.groupby('fold')[m + '_s'].first()]
    return out


if __name__ == '__main__':
    grid_fn, grid_fp = [2, 5, 10], [5, 10, 20, 50]
    res, rows = {}, []
    for ds in ['D1', 'D2']:
        P = pd.read_pickle(f'data/preds_step3_{ds}.pkl')
        res[ds] = {}
        for m in ['gp', 'xgb']:
            res[ds][m] = {'calibration': calibration(P, m)}
            p = prob_eligible(P[m + '_mu'].to_numpy(), P[m + '_sd'].to_numpy(), P[m + '_s'].to_numpy())
            for c_fn in grid_fn:
                for c_fp in grid_fp:
                    for name, a in policies(P[m + '_mu'].to_numpy(), p, c_fn, c_fp).items():
                        r = evaluate(a, P.soh_nom.to_numpy(), P.w.to_numpy(), c_fn, c_fp)
                        rows.append({'data': ds, 'model': m, 'C_FN': c_fn, 'C_FP': c_fp, 'policy': name, **r})
    R = pd.DataFrame(rows)
    R.to_csv('data/results_step3_grid.csv', index=False)
    json.dump(res, open('data/results_step3_calibration.json', 'w'), indent=1)
    pd.set_option('display.width', 220)
    print(json.dumps(res, indent=1))
    head = R[(R.C_FN == 5) & (R.C_FP == 20)]
    print(head.round(3).to_string(index=False))
