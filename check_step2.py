"""Independent re-computation of step-2 metrics from the saved predictions (no shared code)."""
import json

import numpy as np
import pandas as pd

res = json.load(open('data/results_step2.json'))
for ds in ['D1', 'D2']:
    P = pd.read_pickle(f'data/preds_step2_{ds}.pkl')
    # sanity: every test cell appears in exactly one fold; no cell in train and test is checked in baselines by construction
    assert P.groupby('cell_id').fold.nunique().max() == 1
    for m in ['ridge', 'xgb', 'gp']:
        # per-cell metrics first, then the plain mean over cells (= equal weight per cell)
        rows = []
        for c, g in P.groupby('cell_id'):
            e = g[m] - g.soh_nom
            el, pe = g.soh_nom >= 80, g[m] >= 80
            rows.append(dict(mae=e.abs().mean(), mse=(e ** 2).mean(), acc=(el == pe).mean(),
                             n_el=el.mean(), fn=(el & ~pe).mean(), n_in=(~el).mean(), fp=(~el & pe).mean()))
        R = pd.DataFrame(rows)
        mae, rmse = R.mae.mean(), np.sqrt(R.mse.mean())
        fnr, fpr = R.fn.sum() / R.n_el.sum(), R.fp.sum() / R.n_in.sum()
        s = res[ds][m]
        ok = all(abs(a - b) < 1e-9 for a, b in [(mae, s['MAE']), (rmse, s['RMSE']), (R.acc.mean(), s['accuracy']), (fnr, s['FN_rate']), (fpr, s['FP_rate'])])
        print(ds, m, 'cells', len(R), f'MAE {mae:.3f} RMSE {rmse:.3f} FNR {fnr:.3f} FPR {fpr:.3f}', 'MATCH' if ok else 'MISMATCH')
