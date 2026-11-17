"""Independent re-run of one leave-one-chemistry-out setting (target D2, no target cells)
with separately written features, sampling, calibration and scoring."""
import math
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ok = pd.read_pickle('data/clean.pkl')
times = np.array([0, 120, 240, 360, 600, 960, 1560.0])
def feats(df):
    rows = []
    for t, v in zip(df.rest_t, df.rest_v):
        vv = np.interp(times, np.array(t.split(';'), float), np.array(v.split(';'), float))
        rows.append(np.r_[vv[0], vv[1:] - vv[0]])
    return np.array(rows)
def sample(df, n):
    keep = []
    for _, g in df.groupby('cell_id'):
        pos = np.unique(np.round(np.linspace(0, len(g) - 1, min(n, len(g)))).astype(int))
        keep += list(g.index[pos])
    return df.loc[keep]
def gp(X, y):
    sc = StandardScaler().fit(X)
    m = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(np.ones(X.shape[1])) + WhiteKernel(0.1), normalize_y=True, n_restarts_optimizer=0, random_state=0).fit(sc.transform(X), y)
    return lambda Z: m.predict(sc.transform(Z), return_std=True)
d = ok.copy(); d['F'] = list(feats(d))
d['cond'] = 'D' + d.ds.astype(str) + '_' + d.temp.astype(str) + '_' + d.chg_rate.astype(str) + '_' + d.dis_rate.astype(str)
d['w'] = 1.0 / d.groupby('cell_id').cycle.transform('size')
tr, te = d[d.ds != 2], d[d.ds == 2]
res = []
for fi, ci in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1).split(tr, tr.cond, tr.cell_id):
    s = sample(tr.iloc[fi], 10); f = gp(np.stack(s.F), s.soh_nom.to_numpy())
    mu, sd = f(np.stack(tr.iloc[ci].F))
    res += list(zip(np.abs(tr.iloc[ci].soh_nom - mu) / sd, tr.iloc[ci].w))
res.sort(); tot = sum(w for _, w in res); acc = 0
for r, w in res:
    acc += w
    if acc / tot > 0.90 - 1e-12: k = r / 1.6449; break
s = sample(tr, 10); mu, sd = gp(np.stack(s.F), s.soh_nom.to_numpy())(np.stack(te.F))
cells = {}
for c, y, m_, s_ in zip(te.cell_id, te.soh_nom, mu, sd):
    p = 0.5 * math.erfc((80 - m_) / (k * s_) / math.sqrt(2))
    e = {'SL': (1 - p) * 20, 'RC': p * 5, 'RT': 1.0}; a = min(e, key=lambda q: (e[q], ['SL', 'RC', 'RT'].index(q)))
    cost = 1 if a == 'RT' else (20 if a == 'SL' and y < 80 else (5 if a == 'RC' and y >= 80 else 0))
    cells.setdefault(c, []).append((abs(m_ - y), cost, a == 'RT'))
C = np.array([np.mean(v, axis=0) for v in cells.values()]).mean(0)
row = pd.read_csv('data/results_step5_loco.csv').query("target=='D2' and k_target_cells==0").iloc[0]
print(f'scale {k:.4f} vs {row.scale:.4f}; MAE {C[0]:.4f} vs {row.MAE:.4f}; cost {C[1]:.4f} vs {row.cost_ours_5_20:.4f}; retest {C[2]:.4f} vs {row.retest_ours_5_20:.4f}')
print('MATCH' if abs(C[0] - row.MAE) < 1e-6 and abs(C[1] - row.cost_ours_5_20) < 1e-6 and abs(C[2] - row.retest_ours_5_20) < 1e-6 else 'MISMATCH')
