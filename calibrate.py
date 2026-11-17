"""Step 3a: models with calibrated uncertainty.

Same outer folds as step 2 (same test cells). Inside each outer fold, a 5-fold
split by cell of the training cells gives out-of-fold predictions for every
training cell (cross-fitting). The predictive spread is rescaled so that the
cell-weighted 90% range covers 90% of those out-of-fold errors; the final models
are then fitted on all training cells. Saves test predictions with mean, spread
and scale for the decision step.
Usage: python3 -I -u calibrate.py
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, '.')
from baselines import FEATS, features, per_cell_sample  # noqa: E402

Z90 = 1.6449


def scale_factor(y, mu, sd, w, q=0.90):
    """Factor s so that mu +/- Z90*s*sd covers a cell-weighted share q of errors."""
    r = np.abs(y - mu) / sd
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / w.sum()
    return float(r[o][np.searchsorted(cw, q)] / Z90)


def run(ok, ds, seed=0):
    d = ok[ok.ds == ds].copy()
    d[FEATS] = features(d)
    d['cond'] = d.temp.astype(str) + '_' + d.chg_rate.astype(str) + '_' + d.dis_rate.astype(str)
    d['w'] = 1.0 / d.groupby('cell_id').cycle.transform('size')
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    out = []
    for k, (tr_i, te_i) in enumerate(outer.split(d, d.cond, d.cell_id)):
        t0 = time.time()
        tr, te = d.iloc[tr_i], d.iloc[te_i]
        def fit_models(part):
            fs = per_cell_sample(part)
            Xf, yf = fs[FEATS].to_numpy(), fs.soh_nom.to_numpy()
            sc = StandardScaler().fit(Xf)
            kern = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(FEATS))) + WhiteKernel(0.1)
            gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=1, random_state=seed).fit(sc.transform(Xf), yf)
            xp = dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, min_child_weight=5, random_state=seed)
            xm = xgb.XGBRegressor(**xp).fit(Xf, yf)
            xq = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=np.array([0.05, 0.95]), **xp).fit(Xf, yf)

            def predict(p):
                X = p[FEATS].to_numpy()
                mu_g, sd_g = gp.predict(sc.transform(X), return_std=True)
                q = xq.predict(X)
                return mu_g, sd_g, xm.predict(X), np.maximum((q[:, 1] - q[:, 0]) / (2 * Z90), 1e-3)
            return predict

        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
        oof = []
        for fi, ci in inner.split(tr, tr.cond, tr.cell_id):
            pr = fit_models(tr.iloc[fi])(tr.iloc[ci])
            oof.append(pd.DataFrame({'y': tr.iloc[ci].soh_nom.to_numpy(), 'w': tr.iloc[ci].w.to_numpy(),
                                     'gm': pr[0], 'gs': pr[1], 'xm': pr[2], 'xs': pr[3]}))
        oof = pd.concat(oof)
        s_gp = scale_factor(oof['y'].to_numpy(), oof['gm'].to_numpy(), oof['gs'].to_numpy(), oof['w'].to_numpy())
        s_xgb = scale_factor(oof['y'].to_numpy(), oof['xm'].to_numpy(), oof['xs'].to_numpy(), oof['w'].to_numpy())
        predict = fit_models(tr)
        g_mu, g_sd, x_mu, x_sd = predict(te)
        o = te[['cell_id', 'cycle', 'cond', 'soh_nom', 'w']].copy()
        o['fold'] = k
        o['gp_mu'], o['gp_sd'], o['gp_s'] = g_mu, g_sd, s_gp
        o['xgb_mu'], o['xgb_sd'], o['xgb_s'] = x_mu, x_sd, s_xgb
        out.append(o)
        print(f'D{ds} fold {k}: train {tr.cell_id.nunique()} cells (all used for calibration by cross-fitting), test {te.cell_id.nunique()}; '
              f'scale GP {s_gp:.2f}, XGB {s_xgb:.2f}; {time.time() - t0:.0f} s', flush=True)
    return pd.concat(out)


if __name__ == '__main__':
    ok = pd.read_pickle('data/clean.pkl')
    for ds in [1, 2]:
        run(ok, ds).to_pickle(f'data/preds_step3_D{ds}.pkl')
    print('EXIT OK')
