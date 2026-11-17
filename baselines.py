"""Step 2: ridge, XGBoost and Gaussian-process baselines on six relaxation-curve features.

Label: SoH against rated capacity (D-13). Split: by cell, 5 folds stratified by test
condition (D-03). Each cell weighs the same in training and scoring (D-14).
Usage: python3 -I baselines.py
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, '.')
from load import relax_matrix  # noqa: E402

GRID = np.arange(0, 1561, 120.0)     # 26-min rest, a point every 2 min
THRESH = 80.0                        # second-life threshold, % of rated capacity (D-04)
PER_CELL = 25                        # training cycles per cell (equal weight, D-14; keeps the exact GP tractable)
Z90 = 1.6449
FEATS = ['var', 'skew', 'max', 'min', 'mean', 'kurt']


def features(d):
    X = relax_matrix(d, GRID)
    return np.column_stack([X.var(1), stats.skew(X, 1), X.max(1), X.min(1), X.mean(1), stats.kurtosis(X, 1)])


def per_cell_sample(d, n=PER_CELL):
    idx = []
    for _, g in d.groupby('cell_id'):
        take = np.unique(np.linspace(0, len(g) - 1, min(n, len(g))).round().astype(int))
        idx.extend(g.index[take])
    return d.loc[idx]


def scores(y, p, w, lo=None, hi=None):
    """Cell-weighted SoH error and routing errors at the 80% threshold."""
    w = w / w.sum()
    err = p - y
    elig, pe = y >= THRESH, p >= THRESH
    out = {
        'MAE': float(np.sum(w * np.abs(err))),
        'RMSE': float(np.sqrt(np.sum(w * err ** 2))),
        'accuracy': float(np.sum(w * (elig == pe))),
        'FN_rate': float(np.sum(w * (elig & ~pe)) / np.sum(w * elig)),     # eligible sent to recycling
        'FP_rate': float(np.sum(w * (~elig & pe)) / np.sum(w * ~elig)),    # unsuitable sent to second life
        'FN_count': int(np.sum(elig & ~pe)), 'FP_count': int(np.sum(~elig & pe)), 'n': int(len(y)),
    }
    tp, fp, fn = np.sum(w * (elig & pe)), np.sum(w * (~elig & pe)), np.sum(w * (elig & ~pe))
    out['F1'] = float(2 * tp / (2 * tp + fp + fn))
    if lo is not None:
        out['cover90'] = float(np.sum(w * ((y >= lo) & (y <= hi))))
        out['width90'] = float(np.sum(w * (hi - lo)))
    return out


def run(ok, ds, seed=0):
    d = ok[ok.ds == ds].copy()
    F = features(d)
    d[FEATS] = F
    d['cond'] = d.temp.astype(str) + '_' + d.chg_rate.astype(str) + '_' + d.dis_rate.astype(str)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = []
    for k, (tr_i, te_i) in enumerate(cv.split(d, d.cond, d.cell_id)):
        tr, te = d.iloc[tr_i], d.iloc[te_i]
        trs = per_cell_sample(tr)
        Xtr, ytr, Xte = trs[FEATS].to_numpy(), trs.soh_nom.to_numpy(), te[FEATS].to_numpy()
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        out = te[['cell_id', 'cycle', 'cond', 'soh_nom']].copy()
        out['fold'] = k
        t0 = time.time()
        out['ridge'] = Ridge(alpha=1.0).fit(Ztr, ytr).predict(Zte)
        xp = dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, min_child_weight=5, random_state=seed)
        out['xgb'] = xgb.XGBRegressor(**xp).fit(Xtr, ytr).predict(Xte)
        q = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=np.array([0.05, 0.95]), **xp).fit(Xtr, ytr).predict(Xte)
        out['xgb_lo'], out['xgb_hi'] = q[:, 0], q[:, 1]
        kern = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(FEATS))) + WhiteKernel(0.1)
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=1, random_state=seed).fit(Ztr, ytr)
        mu, sd = gp.predict(Zte, return_std=True)
        out['gp'], out['gp_lo'], out['gp_hi'] = mu, mu - Z90 * sd, mu + Z90 * sd
        print(f'D{ds} fold {k}: train cells {tr.cell_id.nunique()} ({len(trs)} cycles), test cells {te.cell_id.nunique()} ({len(te)} cycles), {time.time() - t0:.0f} s', flush=True)
        preds.append(out)
    P = pd.concat(preds)
    P['w'] = 1.0 / P.groupby('cell_id').cycle.transform('size')
    return P


def summarise(P):
    res = {}
    for m in ['ridge', 'xgb', 'gp']:
        lo = P.get(m + '_lo'); hi = P.get(m + '_hi')
        allf = scores(P.soh_nom.to_numpy(), P[m].to_numpy(), P.w.to_numpy(),
                      None if lo is None else lo.to_numpy(), None if hi is None else hi.to_numpy())
        folds = [scores(g.soh_nom.to_numpy(), g[m].to_numpy(), g.w.to_numpy()) for _, g in P.groupby('fold')]
        allf['MAE_fold_sd'] = float(np.std([f['MAE'] for f in folds]))
        allf['FN_rate_fold_sd'] = float(np.std([f['FN_rate'] for f in folds]))
        allf['FP_rate_fold_sd'] = float(np.std([f['FP_rate'] for f in folds]))
        res[m] = allf
    return res


if __name__ == '__main__':
    ok = pd.read_pickle('data/clean.pkl')
    results = {}
    for ds in [1, 2]:
        P = run(ok, ds)
        P.to_pickle(f'data/preds_step2_D{ds}.pkl')
        results[f'D{ds}'] = summarise(P)
    json.dump(results, open('data/results_step2.json', 'w'), indent=1)
    rows = []
    for ds, r in results.items():
        for m, s in r.items():
            rows.append({'data': ds, 'model': m, **{k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}})
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False))
