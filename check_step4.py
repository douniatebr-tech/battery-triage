"""Independent re-computation of step-4 CNN numbers (per-cell loop)."""
import json, math
import pandas as pd
res = json.load(open('data/results_step4.json')); R = pd.read_csv('data/results_step4_grid.csv')
phi_sf = lambda x: 0.5 * math.erfc(x / math.sqrt(2))
for ds in ['D1', 'D2']:
    B = pd.read_pickle(f'data/preds_step4_{ds}.pkl')
    cells = []
    for c, g in B.groupby('cell_id'):
        n = len(g); mae = 0; cost = 0; rt = 0
        for y, mu, sd, s in zip(g.soh_nom, g.cnn_mu, g.cnn_sd, g.cnn_s):
            mae += abs(mu - y)
            p = phi_sf((80 - mu) / (s * sd)); e = {'SL': (1 - p) * 20, 'RC': p * 5, 'RT': 1.0}
            a = min(e, key=lambda k: (e[k], ['SL', 'RC', 'RT'].index(k)))
            cost += 1 if a == 'RT' else (20 if a == 'SL' and y < 80 else (5 if a == 'RC' and y >= 80 else 0)); rt += a == 'RT'
        cells.append((mae / n, cost / n, rt / n))
    C = pd.DataFrame(cells, columns=['mae', 'cost', 'rt']).mean()
    row = R[(R.data == ds) & (R.model == 'cnn') & (R.C_FN == 5) & (R.C_FP == 20)].iloc[0]
    ok = abs(C.mae - res[ds]['cnn']['MAE']) < 1e-3 and abs(C.cost - row.cost_per_battery) < 1e-9 and abs(C.rt - row.retest_share) < 1e-9
    print(ds, 'cells', len(cells), f'MAE {C.mae:.4f} cost {C.cost:.4f} retest {C.rt:.4f}', 'MATCH' if ok else 'MISMATCH')
