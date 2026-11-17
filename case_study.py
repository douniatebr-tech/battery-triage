"""Cost case study, savings map and cell-level bootstrap confidence intervals (Gaussian process, step-3 predictions).

Literature-based cost cases, all per kWh of nameplate capacity (assumptions stated in CASES and in the paper):
  repurposing (collection, test, packaging) R = $25-49/kWh; testing taken as half of R (labour-dominated)
  second-life price P = 30-70% of a $151/kWh new pack; recycling value V = $23-31/kWh (retired-battery purchase price, China)
  C_T = R/2;  C_FN = P - R - V (margin lost when an eligible battery is recycled);  C_FP = P + C_T (buyer loses the
  battery and a replacement test/installation when an unsuitable battery fails early); all divided by C_T.
Usage: python3 -I case_study.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

THR = 80.0
rng = np.random.default_rng(0)


def case(P, R, V):
    ct = R / 2
    return {'C_T_usd': ct, 'C_FN': max(P - R - V, 0) / ct, 'C_FP': (P + ct) / ct}


NEW = 151.0
CASES = {
    'central (P 50% of new, R $37, V $27)': case(0.5 * NEW, 37, 27),
    'high margin (P 70% of new, R $25, V $23)': case(0.7 * NEW, 25, 23),
    'low margin (P 30% of new, R $49, V $31)': case(0.3 * NEW, 49, 31),
    'illustrative (C_FN 5, C_FP 20)': {'C_T_usd': None, 'C_FN': 5.0, 'C_FP': 20.0},
}


def row_costs(P, cfn, cfp):
    """Realised cost per row for each policy (units of one full test)."""
    y, mu = P.soh_nom.to_numpy(), P.gp_mu.to_numpy()
    p = norm.sf(THR, loc=mu, scale=P.gp_s.to_numpy() * P.gp_sd.to_numpy())
    elig = y >= THR

    def realised(a):  # 0 second life, 1 recycle, 2 retest
        return np.where(a == 2, 1.0, np.where((a == 0) & ~elig, cfp, np.where((a == 1) & elig, cfn, 0.0)))
    exp_sl, exp_rc = (1 - p) * cfp, p * cfn
    return {
        'three-way (ours)': realised(np.argmin(np.column_stack([exp_sl, exp_rc, np.ones_like(p)]), axis=1)),
        'point rule': realised(np.where(mu >= THR, 0, 1)),
        'two-way by cost': realised(np.where(exp_sl <= exp_rc, 0, 1)),
        'screen then test': realised(np.where(exp_rc <= 1.0, 1, 2)),
        'retest all': np.ones(len(P)),
    }


def cell_means(P, cfn, cfp):
    rc = row_costs(P, cfn, cfp)
    df = pd.DataFrame(rc); df['cell'] = P.cell_id.to_numpy()
    return df.groupby('cell').mean()          # one row per cell: equal weight per cell (D-14)


def summarise(C, B=2000):
    ours = C['three-way (ours)'].to_numpy()
    others = C.drop(columns='three-way (ours)')
    out = {'ours': ours.mean()}
    idx = rng.integers(0, len(C), size=(B, len(C)))
    boot_ours = ours[idx].mean(1)
    out['ours_ci'] = np.percentile(boot_ours, [2.5, 97.5]).tolist()
    for name in others.columns:
        o = others[name].to_numpy()
        diff = o[idx].mean(1) - boot_ours
        out[f'saving_vs_{name}'] = o.mean() - ours.mean()
        out[f'saving_vs_{name}_ci'] = np.percentile(diff, [2.5, 97.5]).tolist()
    best = others.mean().idxmin()
    ob = others[best].to_numpy()
    out['best_other'] = best
    out['saving_vs_best_other_pct'] = 100 * (1 - ours.mean() / ob.mean())
    rel = 100 * (1 - boot_ours / ob[idx].mean(1))
    out['saving_vs_best_other_pct_ci'] = np.percentile(rel, [2.5, 97.5]).tolist()
    return out


if __name__ == '__main__':
    res = {'cases': {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in CASES.items()}}
    for ds in ['D1', 'D2']:
        P = pd.read_pickle(f'data/preds_step3_{ds}.pkl')
        res[ds] = {}
        for name, c in CASES.items():
            s = summarise(cell_means(P, c['C_FN'], c['C_FP']))
            res[ds][name] = s
            print(f"{ds} | {name}: C_FN {c['C_FN']:.2f}, C_FP {c['C_FP']:.2f} -> ours {s['ours']:.3f} "
                  f"[{s['ours_ci'][0]:.3f}, {s['ours_ci'][1]:.3f}]; best other = {s['best_other']}, saving "
                  f"{s['saving_vs_best_other_pct']:.0f}% [{s['saving_vs_best_other_pct_ci'][0]:.0f}, {s['saving_vs_best_other_pct_ci'][1]:.0f}]; "
                  f"vs screen {s['saving_vs_screen then test']:.3f} [{s['saving_vs_screen then test_ci'][0]:.3f}, {s['saving_vs_screen then test_ci'][1]:.3f}]; "
                  f"vs point {s['saving_vs_point rule']:.3f} [{s['saving_vs_point rule_ci'][0]:.3f}, {s['saving_vs_point rule_ci'][1]:.3f}]", flush=True)
        # savings map on a fine grid
        fn = np.geomspace(0.5, 20, 25); fp = np.geomspace(1, 60, 25)
        M = np.zeros((len(fp), len(fn))); R = np.zeros_like(M)
        for i, b in enumerate(fp):
            for j, a in enumerate(fn):
                C = cell_means(P, a, b).mean()
                others = C.drop('three-way (ours)')
                M[i, j] = 100 * (1 - C['three-way (ours)'] / others.min())
                R[i, j] = C['three-way (ours)']
        res[ds]['map'] = {'C_FN': fn.tolist(), 'C_FP': fp.tolist(), 'saving_pct': M.round(2).tolist(), 'cost_ours': R.round(4).tolist()}
        print(f'{ds} map: saving vs best other ranges {M.min():.1f}% to {M.max():.1f}%; '
              f'negative cells: {(M < -0.5).sum()} of {M.size}', flush=True)
    json.dump(res, open('data/results_case_study.json', 'w'), indent=1, default=float)
    print('EXIT OK')
