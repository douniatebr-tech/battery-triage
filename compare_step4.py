"""Step 4 comparison: CNN ensemble against GP and XGBoost on the same test cells.

Also fits XGBoost on all training cycles (equal weight per cell) as a fairness check,
since the CNN sees all cycles while GP/XGBoost in steps 2-3 used 25 per cell.
Usage: python3 -I compare_step4.py
"""
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, '.')
from baselines import FEATS, features, scores  # noqa: E402
from decide import evaluate, policies, prob_eligible  # noqa: E402

Z90 = 1.6449
out, rows = {}, []
for ds in ['D1', 'D2']:
    A = pd.read_pickle(f'data/preds_step3_{ds}.pkl')
    B = pd.read_pickle(f'data/preds_step4_{ds}.pkl')[['cell_id', 'soh_nom', 'cnn_mu', 'cnn_sd', 'cnn_s']]
    # rows share the index of data/clean.pkl (one cell restarted its cycle count, so match by row)
    assert A.index.is_unique and B.index.is_unique and set(A.index) == set(B.index)
    assert (A.cell_id == B.cell_id.reindex(A.index)).all() and np.allclose(A.soh_nom, B.soh_nom.reindex(A.index))
    P = A.join(B[['cnn_mu', 'cnn_sd', 'cnn_s']])
    # fairness check: XGBoost on all training cycles, weighted equally per cell
    ok = pd.read_pickle('data/clean.pkl')
    d = ok[ok.ds == int(ds[1])].copy()
    d[FEATS] = features(d)
    d['cond'] = d.temp.astype(str) + '_' + d.chg_rate.astype(str) + '_' + d.dis_rate.astype(str)
    d['wc'] = 1.0 / d.groupby('cell_id').cycle.transform('size')
    pred = pd.Series(index=d.index, dtype=float)
    for tr_i, te_i in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(d, d.cond, d.cell_id):
        tr, te = d.iloc[tr_i], d.iloc[te_i]
        m = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, min_child_weight=5, random_state=0)
        m.fit(tr[FEATS], tr.soh_nom, sample_weight=tr.wc)
        pred.loc[te.index] = m.predict(te[FEATS])
    P = P.join(pred.rename('xgb_all'))
    assert P.xgb_all.notna().all()
    y, w = P.soh_nom.to_numpy(), P.w.to_numpy()
    out[ds] = {}
    for m in ['gp', 'xgb', 'cnn', 'xgb_all']:
        mu = P[m + '_mu' if m != 'xgb_all' else m].to_numpy()
        r = scores(y, mu, w)
        if m != 'xgb_all':
            sd, s = P[m + '_sd'].to_numpy(), P[m + '_s'].to_numpy()
            ww = w / w.sum()
            r['cover90_raw'] = float(np.sum(ww * (np.abs(y - mu) <= Z90 * sd)))
            r['cover90_cal'] = float(np.sum(ww * (np.abs(y - mu) <= Z90 * s * sd)))
            r['width90_cal'] = float(np.sum(ww * 2 * Z90 * s * sd))
            p = prob_eligible(mu, sd, s)
            for c_fn in [2, 5, 10]:
                for c_fp in [5, 10, 20, 50]:
                    e = evaluate(policies(mu, p, c_fn, c_fp)['three-way (ours)'], y, w, c_fn, c_fp)
                    rows.append({'data': ds, 'model': m, 'C_FN': c_fn, 'C_FP': c_fp, **e})
        out[ds][m] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
R = pd.DataFrame(rows)
R.to_csv('data/results_step4_grid.csv', index=False)
json.dump(out, open('data/results_step4.json', 'w'), indent=1)
pd.set_option('display.width', 250)
for ds in out:
    print(ds); print(pd.DataFrame(out[ds]).T[['MAE', 'RMSE', 'accuracy', 'FN_rate', 'FP_rate', 'cover90_raw', 'cover90_cal', 'width90_cal']].to_string())
print(R[(R.C_FN == 5) & (R.C_FP == 20)].round(3).to_string(index=False))
W = R.pivot_table(index=['data', 'C_FN', 'C_FP'], columns='model', values='cost_per_battery')
print('\nthree-way cost by model across the grid'); print(W.round(3).to_string())
print('CNN cheaper than GP in', int((W.cnn < W.gp).sum()), 'of', len(W), 'settings')
