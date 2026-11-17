"""Step 5: robustness of the Gaussian-process triage (main model, D-15).

(a) Rest length: features are the rest voltage at the start and its drop at fixed times up to
    T. Datasets 1 and 2 (a point every 2 min): T = 2, 4, 6, 10, 16, 26 min, same 5 outer folds
    as before. Dataset 3 (a point every 30 s): T = 0.5, 1, 2, 4, 10, 26, 59 min, leave one cell out.
(b) Leave one chemistry out: train on two datasets, test on the third at T = 26 min; then add
    k = 1, 2 or 4 cells of the target chemistry to training (3 random repeats).
Calibration is cross-fitted on the training cells (3 inner folds), as in step 3.
Usage: python3 -I -u robust.py time|loco
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from baselines import per_cell_sample, scores  # noqa: E402
from calibrate import scale_factor  # noqa: E402
from decide import evaluate, policies, prob_eligible  # noqa: E402
from load import relax_matrix  # noqa: E402

Z90 = 1.6449
COARSE = [0, 120, 240, 360, 600, 960, 1560]
FINE = [0, 30, 60, 120, 240, 600, 960, 1560, 2400, 3540]
COSTS = [(c_fn, c_fp) for c_fn in (2, 5, 10) for c_fp in (5, 10, 20, 50)]


def add_curve(d, times):
    V = relax_matrix(d, np.array(times, float))
    cols = ['v0'] + [f'dv{t}' for t in times[1:]]
    d[cols] = np.column_stack([V[:, 0], V[:, 1:] - V[:, :1]])
    return cols


def gp_model(X, y, seed=0):
    sc = StandardScaler().fit(X)
    kern = ConstantKernel(1.0) * RBF(np.ones(X.shape[1])) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=0, random_state=seed).fit(sc.transform(X), y)
    return lambda Xn: gp.predict(sc.transform(Xn), return_std=True)


def fit_with_scale(tr, cols, n, seed=0):
    oof = []
    for fi, ci in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + 1).split(tr, tr.cond, tr.cell_id):
        s = per_cell_sample(tr.iloc[fi], n)
        mu, sd = gp_model(s[cols].to_numpy(), s.soh_nom.to_numpy(), seed)(tr.iloc[ci][cols].to_numpy())
        oof.append(pd.DataFrame({'y': tr.iloc[ci].soh_nom.to_numpy(), 'w': tr.iloc[ci].w.to_numpy(), 'm': mu, 's': sd}))
    oof = pd.concat(oof)
    k = scale_factor(oof['y'].to_numpy(), oof['m'].to_numpy(), oof['s'].to_numpy(), oof['w'].to_numpy())
    s = per_cell_sample(tr, n)
    return gp_model(s[cols].to_numpy(), s.soh_nom.to_numpy(), seed), k


def metrics(P):
    y, mu, sd, k, w = (P[c].to_numpy() for c in ['soh_nom', 'mu', 'sd', 'k', 'w'])
    r = scores(y, mu, w)
    ww = w / w.sum()
    r['cover90_cal'] = float(np.sum(ww * (np.abs(y - mu) <= Z90 * k * sd)))
    p = prob_eligible(mu, sd, k)
    for c_fn, c_fp in COSTS:
        for pol in ['three-way (ours)', 'point']:
            e = evaluate(policies(mu, p, c_fn, c_fp)[pol], y, w, c_fn, c_fp)
            tag = 'ours' if pol.startswith('three') else 'point'
            r[f'cost_{tag}_{c_fn}_{c_fp}'] = e['cost_per_battery']
            r[f'retest_{tag}_{c_fn}_{c_fp}'] = e['retest_share']
    r['cost_ours_grid_mean'] = float(np.mean([r[f'cost_ours_{a}_{b}'] for a, b in COSTS]))
    r['retest_ours_grid_min'] = float(min(r[f'retest_ours_{a}_{b}'] for a, b in COSTS))
    r['retest_ours_grid_max'] = float(max(r[f'retest_ours_{a}_{b}'] for a, b in COSTS))
    return r


def base(ok):
    d = ok.copy()
    d['cond'] = 'D' + d.ds.astype(str) + '_' + d.temp.astype(str) + '_' + d.chg_rate.astype(str) + '_' + d.dis_rate.astype(str)
    d['w'] = 1.0 / d.groupby('cell_id').cycle.transform('size')
    return d


def time_study(ok):
    rows = []
    for ds, times, Ts, n in [(1, COARSE, [120, 240, 360, 600, 960, 1560], 15),
                             (2, COARSE, [120, 240, 360, 600, 960, 1560], 15),
                             (3, FINE, [30, 60, 120, 240, 600, 1560, 3540], 60)]:
        d = base(ok[ok.ds == ds])
        allcols = add_curve(d, times)
        if ds == 3:
            splits = list(LeaveOneGroupOut().split(d, groups=d.cell_id))
        else:
            splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(d, d.cond, d.cell_id))
        for T in Ts:
            t0 = time.time()
            cols = allcols[:1 + sum(1 for t in times[1:] if t <= T)]
            out = []
            for k, (tr_i, te_i) in enumerate(splits):
                tr, te = d.iloc[tr_i], d.iloc[te_i]
                pr, kk = fit_with_scale(tr, cols, n, seed=k)
                mu, sd = pr(te[cols].to_numpy())
                out.append(pd.DataFrame({'soh_nom': te.soh_nom.to_numpy(), 'w': te.w.to_numpy(), 'mu': mu, 'sd': sd, 'k': kk}))
            r = metrics(pd.concat(out))
            rows.append({'data': f'D{ds}', 'rest_min': T / 60, 'n_points': len(cols), **r})
            print(f"D{ds} rest {T / 60:g} min: MAE {r['MAE']:.2f}, routing {r['accuracy']:.3f}, cover {r['cover90_cal']:.3f}, "
                  f"cost {r['cost_ours_5_20']:.3f}, retest {r['retest_ours_5_20']:.3f}; {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv('data/results_step5_time.csv', index=False)


def loco(ok):
    d = base(ok)
    cols = add_curve(d, COARSE)
    rows = []
    for tgt in [1, 2, 3]:
        src = d[d.ds != tgt]
        tcells = np.array(sorted(d[d.ds == tgt].cell_id.unique()))
        for kshot in [0, 1, 2, 4]:
            for rep in range(1 if kshot == 0 else 3):
                t0 = time.time()
                pick = np.random.default_rng(100 * tgt + rep).choice(tcells, kshot, replace=False) if kshot else []
                tr = pd.concat([src, d[d.cell_id.isin(pick)]])
                te = d[(d.ds == tgt) & ~d.cell_id.isin(pick)]
                pr, kk = fit_with_scale(tr, cols, 10, seed=rep)
                mu, sd = pr(te[cols].to_numpy())
                r = metrics(pd.DataFrame({'soh_nom': te.soh_nom.to_numpy(), 'w': te.w.to_numpy(), 'mu': mu, 'sd': sd, 'k': kk}))
                rows.append({'target': f'D{tgt}', 'k_target_cells': kshot, 'rep': rep, 'test_cells': te.cell_id.nunique(), 'scale': kk, **r})
                print(f"target D{tgt}, k={kshot}, rep {rep}: MAE {r['MAE']:.2f}, routing {r['accuracy']:.3f}, cover {r['cover90_cal']:.3f}, "
                      f"cost {r['cost_ours_5_20']:.3f}, retest {r['retest_ours_5_20']:.3f}; {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv('data/results_step5_loco.csv', index=False)


if __name__ == '__main__':
    ok = pd.read_pickle('data/clean.pkl')
    {'time': time_study, 'loco': loco}[sys.argv[1]](ok)
    print('EXIT OK')
