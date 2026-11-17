"""Step 5c: the three-way decision rule on real retired notebook cells (Salinas data, CC BY 4.0).

Two input sets, both available without a capacity test:
  physical  - voltage on arrival, rated capacity, manufacturing year, corrosion / burns / holes
  + BMS     - the above plus the pack's own capacity estimate and cycle count (passport-style data)
Label: measured capacity as % of rated (Co). Thresholds: 70% (Salinas' criterion) and 80%.
Cells with no conductivity (46) are recycled by the arrival voltage check and left out of the model;
cells without a measured capacity (45 more) are dropped. The pack column 'sb' is not used.
Gaussian process, 5-fold split by pack, cross-fitted calibration (3 inner folds), equal weight per cell.
Usage: python3 -I salinas.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from calibrate import scale_factor  # noqa: E402

D = 'data/salinas'  # put '1034 cells.xlsx' and '152 batteries.xlsx' here (see data/README.md)
Z90 = 1.6449
COSTS = [(c_fn, c_fp) for c_fn in (2, 5, 10) for c_fp in (5, 10, 20, 50)]


def load():
    c = pd.read_excel(f'{D}/1034 cells.xlsx', sheet_name='1034 cells')
    b = pd.read_excel(f'{D}/152 batteries.xlsx', sheet_name=0)[['Code', 'cb', 'zb']]
    n_all, n_dead = len(c), int(c['No conductivity'].sum())
    c = c[~c['No conductivity'] & c.Co.notna()].merge(b, on='Code', how='left').reset_index(drop=True)
    c['year'] = c.dn.dt.year + c.dn.dt.dayofyear / 365.25
    for col in ['year', 'cb', 'zb']:
        c[col + '_missing'] = c[col].isna().astype(float)
        c[col] = c[col].fillna(c[col].median())
    for col in ['Corrosion', 'Burns', 'Holes']:
        c[col] = c[col].astype(float)
    return c, n_all, n_dead


PHYS = ['Vo', 'Cn', 'year', 'year_missing', 'Corrosion', 'Burns', 'Holes']
BMS = PHYS + ['cb', 'zb', 'cb_missing', 'zb_missing']


def gp_fit(X, y):
    sc = StandardScaler().fit(X)
    sc.scale_[sc.scale_ == 0] = 1.0
    gp = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(np.ones(X.shape[1])) + WhiteKernel(0.1),
                                  normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(sc.transform(X), y)
    return lambda Z: gp.predict(sc.transform(Z), return_std=True)


def run(c, cols, thr):
    lab = (c.Co >= thr).astype(int)
    out = []
    for tr_i, te_i in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(c, lab, c.Code):
        tr, te = c.iloc[tr_i], c.iloc[te_i]
        oof = []
        for fi, ci in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1).split(tr, lab.iloc[tr_i], tr.Code):
            mu, sd = gp_fit(tr.iloc[fi][cols].to_numpy(), tr.iloc[fi].Co.to_numpy())(tr.iloc[ci][cols].to_numpy())
            oof.append((tr.iloc[ci].Co.to_numpy(), mu, sd))
        y, mu, sd = (np.concatenate(z) for z in zip(*oof))
        k = scale_factor(y, mu, sd, np.ones_like(y))
        mu, sd = gp_fit(tr[cols].to_numpy(), tr.Co.to_numpy())(te[cols].to_numpy())
        out.append(pd.DataFrame({'Co': te.Co.to_numpy(), 'mu': mu, 'sd': sd, 'k': k}))
    return pd.concat(out)


def evaluate(P, thr):
    y, mu, sd, k = P.Co.to_numpy(), P.mu.to_numpy(), P.sd.to_numpy(), P.k.to_numpy()
    elig = y >= thr
    p = norm.sf(thr, loc=mu, scale=k * sd)
    point = mu >= thr
    r = {'MAE': np.mean(np.abs(mu - y)), 'cover90': np.mean(np.abs(y - mu) <= Z90 * k * sd),
         'routing_acc': np.mean(point == elig),
         'balanced_acc': 0.5 * (np.mean(point[elig]) + np.mean(~point[~elig])),
         'share_eligible': elig.mean()}
    for c_fn, c_fp in COSTS:
        cost = np.column_stack([(1 - p) * c_fp, p * c_fn, np.ones_like(p)])
        a = np.argmin(cost, axis=1)                       # 0 second life, 1 recycle, 2 retest
        real = np.where(a == 2, 1.0, np.where((a == 0) & ~elig, c_fp, np.where((a == 1) & elig, c_fn, 0.0)))
        pt = np.where(point & ~elig, c_fp, np.where(~point & elig, c_fn, 0.0))
        r[f'cost_ours_{c_fn}_{c_fp}'] = real.mean()
        r[f'retest_{c_fn}_{c_fp}'] = np.mean(a == 2)
        r[f'FN_{c_fn}_{c_fp}'] = np.mean((a == 1) & elig) / elig.mean()
        r[f'FP_{c_fn}_{c_fp}'] = np.mean((a == 0) & ~elig) / (~elig).mean()
        r[f'cost_point_{c_fn}_{c_fp}'] = pt.mean()
    return r


if __name__ == '__main__':
    c, n_all, n_dead = load()
    print(f'cells: {n_all} in file, {n_dead} with no conductivity (recycled on the voltage check), {len(c)} modelled, '
          f'{c.Code.nunique()} packs; full test time per cell (t): mean {c.t.mean():.1f} h')
    rows = []
    for thr in [70, 80]:
        for name, cols in [('physical checks only', PHYS), ('physical + BMS data', BMS)]:
            r = evaluate(run(c, cols, thr), thr)
            rows.append({'threshold': thr, 'inputs': name, **r})
            print(f"thr {thr}, {name}: MAE {r['MAE']:.1f}, cover {r['cover90']:.2f}, balanced acc {r['balanced_acc']:.3f}, "
                  f"cost {r['cost_ours_5_20']:.3f} (point rule {r['cost_point_5_20']:.3f}), retest {r['retest_5_20']:.2f}, "
                  f"FN {r['FN_5_20']:.3f}, FP {r['FP_5_20']:.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv('data/results_step5_salinas.csv', index=False)
    cols = [f'cost_ours_{a}_{b}' for a, b in COSTS]
    print(R[['threshold', 'inputs']].assign(cost_min=R[cols].min(1).round(3), cost_max=R[cols].max(1).round(3),
          retest_min=R[[f'retest_{a}_{b}' for a, b in COSTS]].min(1).round(2), retest_max=R[[f'retest_{a}_{b}' for a, b in COSTS]].max(1).round(2)).to_string(index=False))
    print('EXIT OK')


# ---- classifiers: on these tabular inputs the GP regression is near chance (balanced accuracy ~0.53),
# so the decision rule is fed calibrated probabilities from classifiers instead (cross-fitted Platt scaling).
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
import xgboost as xgb  # noqa: E402

MODELS = {
    'logistic': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    'xgboost': lambda: xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0),
}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def run_clf(c, cols, thr, make):
    y = (c.Co >= thr).astype(int).to_numpy()
    p_out = np.zeros(len(c))
    for tr_i, te_i in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(c, y, c.Code):
        tr = c.iloc[tr_i]
        oof_p, oof_y = [], []
        for fi, ci in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1).split(tr, y[tr_i], tr.Code):
            m = make().fit(tr.iloc[fi][cols], y[tr_i][fi])
            oof_p.append(m.predict_proba(tr.iloc[ci][cols])[:, 1]); oof_y.append(y[tr_i][ci])
        platt = LogisticRegression().fit(logit(np.concatenate(oof_p))[:, None], np.concatenate(oof_y))
        m = make().fit(tr[cols], y[tr_i])
        p_out[te_i] = platt.predict_proba(logit(m.predict_proba(c.iloc[te_i][cols])[:, 1])[:, None])[:, 1]
    return y.astype(bool), p_out


def evaluate_p(elig, p):
    point = p >= 0.5
    r = {'balanced_acc': 0.5 * (np.mean(point[elig]) + np.mean(~point[~elig])), 'brier': np.mean((p - elig) ** 2),
         'share_eligible': elig.mean()}
    for c_fn, c_fp in COSTS:
        a = np.argmin(np.column_stack([(1 - p) * c_fp, p * c_fn, np.ones_like(p)]), axis=1)
        real = np.where(a == 2, 1.0, np.where((a == 0) & ~elig, c_fp, np.where((a == 1) & elig, c_fn, 0.0)))
        two = np.where((1 - p) * c_fp <= p * c_fn, 0, 1)
        real2 = np.where((two == 0) & ~elig, c_fp, np.where((two == 1) & elig, c_fn, 0.0))
        pt = np.where(point & ~elig, c_fp, np.where(~point & elig, c_fn, 0.0))
        r[f'cost_ours_{c_fn}_{c_fp}'] = real.mean(); r[f'retest_{c_fn}_{c_fp}'] = np.mean(a == 2)
        r[f'FN_{c_fn}_{c_fp}'] = np.mean((a == 1) & elig) / elig.mean(); r[f'FP_{c_fn}_{c_fp}'] = np.mean((a == 0) & ~elig) / (~elig).mean()
        r[f'cost_point_{c_fn}_{c_fp}'] = pt.mean(); r[f'cost_twoway_{c_fn}_{c_fp}'] = real2.mean()
    return r


def classifiers():
    c, _, _ = load()
    rows = []
    for thr in [70, 80]:
        for name, cols in [('physical checks only', PHYS), ('physical + BMS data', BMS)]:
            for mname, make in MODELS.items():
                elig, p = run_clf(c, cols, thr, make)
                r = evaluate_p(elig, p)
                rows.append({'threshold': thr, 'inputs': name, 'model': mname, **r})
                print(f"thr {thr}, {name}, {mname}: balanced acc {r['balanced_acc']:.3f}, Brier {r['brier']:.3f}, "
                      f"cost {r['cost_ours_5_20']:.3f} (point {r['cost_point_5_20']:.3f}, two-way {r['cost_twoway_5_20']:.3f}), "
                      f"retest {r['retest_5_20']:.2f}, FN {r['FN_5_20']:.3f}, FP {r['FP_5_20']:.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv('data/results_step5_salinas_clf.csv', index=False)
    cc = [f'cost_ours_{a}_{b}' for a, b in COSTS]; rr = [f'retest_{a}_{b}' for a, b in COSTS]
    print(R[['threshold', 'inputs', 'model']].assign(cost_min=R[cc].min(1).round(3), cost_max=R[cc].max(1).round(3),
          retest_min=R[rr].min(1).round(2), retest_max=R[rr].max(1).round(2),
          below1=(R[cc] < 1).sum(1)).to_string(index=False))
