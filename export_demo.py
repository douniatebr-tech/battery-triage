"""Export one calibrated Gaussian process per chemistry for the browser demo.

Features: rest voltage at the start and its drop at 2, 4, 6, 10, 16 and 26 min (as in step 5).
Two cells per chemistry are held out of training and supply the example batteries.
Writes demo/model.json and checks a hand-written prediction against scikit-learn.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from baselines import per_cell_sample  # noqa: E402
from calibrate import scale_factor  # noqa: E402
from robust import COARSE, add_curve, base  # noqa: E402

NAMES = {1: 'NCA', 2: 'NCM', 3: 'NCM+NCA blend'}
PER_CELL = {1: 5, 2: 6, 3: 45}


def fit(tr, cols, n):
    s = per_cell_sample(tr, n)
    sc = StandardScaler().fit(s[cols].to_numpy())
    kern = ConstantKernel(1.0) * RBF(np.ones(len(cols))) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(sc.transform(s[cols].to_numpy()), s.soh_nom.to_numpy())
    return sc, gp


ok = pd.read_pickle('data/clean.pkl')
d = base(ok)
cols = add_curve(d, COARSE)
os.makedirs('demo', exist_ok=True)
out = {'times_min': [t / 60 for t in COARSE], 'threshold': 80.0, 'models': {}}
for ds in [1, 2, 3]:
    dd = d[d.ds == ds]
    cells = np.array(sorted(dd.cell_id.unique()))
    held = list(np.random.default_rng(7 + ds).choice(cells, 2, replace=False))
    tr = dd[~dd.cell_id.isin(held)]
    # cross-fitted calibration scale (3 inner folds), as in step 5
    oof = []
    for fi, ci in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1).split(tr, tr.cond, tr.cell_id):
        sc, gp = fit(tr.iloc[fi], cols, PER_CELL[ds])
        mu, sd = gp.predict(sc.transform(tr.iloc[ci][cols].to_numpy()), return_std=True)
        oof.append(pd.DataFrame({'y': tr.iloc[ci].soh_nom.to_numpy(), 'w': tr.iloc[ci].w.to_numpy(), 'm': mu, 's': sd}))
    oof = pd.concat(oof)
    k = scale_factor(oof['y'].to_numpy(), oof['m'].to_numpy(), oof['s'].to_numpy(), oof['w'].to_numpy())
    sc, gp = fit(tr, cols, PER_CELL[ds])
    kp = gp.kernel_.get_params()
    c, ls, noise = kp['k1__k1__constant_value'], np.atleast_1d(kp['k1__k2__length_scale']), kp['k2__noise_level']
    Xs = gp.X_train_
    L = gp.L_
    # hand-written prediction (what the page does) checked against scikit-learn
    te = dd[dd.cell_id.isin(held)]
    Xq = sc.transform(te[cols].to_numpy())
    Kq = c * np.exp(-0.5 * (((Xq[:, None, :] - Xs[None, :, :]) / ls) ** 2).sum(-1))
    mu_h = gp._y_train_mean + gp._y_train_std * (Kq @ gp.alpha_)
    v = solve_triangular(L, Kq.T, lower=True)
    sd_h = gp._y_train_std * np.sqrt(np.maximum(c + noise - (v ** 2).sum(0), 1e-12))
    mu_s, sd_s = gp.predict(Xq, return_std=True)
    assert np.allclose(mu_h, mu_s, atol=1e-6) and np.allclose(sd_h, sd_s, atol=1e-6), 'hand prediction differs'
    # examples: cycles of the held-out cells spread across SoH
    ex = []
    for target in [93, 85, 81, 79, 75, 71]:
        j = (te.soh_nom - target).abs().idxmin()
        r = te.loc[j]
        if any(abs(e['true_soh'] - r.soh_nom) < 1.0 for e in ex):
            continue
        ex.append({'label': f"{r.cell} cycle {int(r.cycle)}", 'true_soh': round(float(r.soh_nom), 2),
                   'volts': [round(float(r.v0), 5)] + [round(float(r.v0 + r[cc]), 5) for cc in cols[1:]]})
    err = np.abs(mu_s - te.soh_nom.to_numpy())
    print(f'D{ds} ({NAMES[ds]}): {len(Xs)} training points, held-out cells {held}, scale {k:.2f}, '
          f'held-out MAE {err.mean():.2f}, examples {[e["true_soh"] for e in ex]}')
    out['models'][NAMES[ds]] = {
        'scaler_mean': sc.mean_.round(8).tolist(), 'scaler_scale': sc.scale_.round(8).tolist(),
        'const': float(c), 'length_scales': ls.round(6).tolist(), 'noise': float(noise),
        'y_mean': float(gp._y_train_mean), 'y_std': float(gp._y_train_std), 'calib_scale': round(k, 4),
        'X': Xs.round(5).tolist(), 'alpha': gp.alpha_.round(8).tolist(),
        'L': [L[i, :i + 1].round(7).tolist() for i in range(len(L))],
        'held_out_cells': held, 'examples': ex, 'n_train_cells': int(tr.cell_id.nunique()),
    }
json.dump(out, open('demo/model.json', 'w'), separators=(',', ':'))
print('model.json', os.path.getsize('demo/model.json') // 1024, 'KB')
