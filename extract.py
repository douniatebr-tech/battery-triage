"""Extract per-cycle relaxation curves and discharge capacity from Zhu et al. 2022 CSVs.

For every cycle: the rest that follows the charge (CC + CV) gives the relaxation
voltage curve; the discharge that follows gives the label (discharge capacity).

Usage: python3 -I extract.py OUTDIR DATASET_TAG csv1 [csv2 ...]
Writes OUTDIR/<DATASET_TAG>__<cell>.npz per CSV.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

GRID = np.arange(0, 1561, 120, dtype=float)  # 0..26 min every 2 min (the logged rest)
COLS = ['t', 'ctrl', 'V', 'I', 'Qd', 'Qc', 'cV', 'cI', 'cyc']


def parse_name(path):
    name = os.path.basename(path)[:-4]
    m = re.match(r'CY(\d+)-([\d]+)_(\d+)-#(\d+)', name)
    temp = int(m.group(1))
    ch = m.group(2)
    charge_c = float('0.' + ch[1:]) if ch.startswith('0') else float(ch)
    dis_c = float(m.group(3))
    return name, temp, charge_c, dis_c


def process(path):
    df = pd.read_csv(path, dtype='float64')
    df.columns = COLS
    key = df[['ctrl', 'cV', 'cI']].to_numpy()
    seg_id = np.r_[0, np.cumsum(np.any(np.diff(key, axis=0) != 0, axis=1))]
    df['seg'] = seg_id
    rows = []
    for cyc, d in df.groupby('cyc', sort=True):
        segs = [(s, g) for s, g in d.groupby('seg', sort=True)]
        kinds = []
        for s, g in segs:
            cI, cV = g.cI.iloc[0], g.cV.iloc[0]
            if cV > 0:
                kinds.append('cv')
            elif cI > 0:
                kinds.append('cc_chg')
            elif cI < 0:
                kinds.append('dis')
            else:
                kinds.append('rest')
        # first rest that directly follows a charge step
        rest_i = next((i for i in range(1, len(segs)) if kinds[i] == 'rest' and kinds[i - 1] in ('cv', 'cc_chg')), None)
        if rest_i is None:
            continue
        dis_i = next((i for i in range(rest_i + 1, len(segs)) if kinds[i] == 'dis'), None)
        if dis_i is None:
            continue
        r = segs[rest_i][1]
        dseg = segs[dis_i][1]
        tr = r.t.to_numpy() - r.t.iloc[0]
        if tr[-1] < GRID[-1] - 1 or len(tr) < 10:
            continue
        v = np.interp(GRID, tr, r.V.to_numpy())
        cap = dseg.Qd.max()
        v_end = dseg.V.iloc[-1]
        chg = segs[rest_i - 1][1]
        rows.append(dict(cycle=int(cyc), cap=cap, v_dis_end=v_end, v_rest0=r.V.iloc[0],
                         qc=chg.Qc.max(), n_rest=len(tr), relax=v))
    return rows


def main():
    outdir, tag, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(outdir, exist_ok=True)
    for f in files:
        name, temp, cc, dc = parse_name(f)
        rows = process(f)
        if not rows:
            print(f'{tag} {name}: no usable cycles', flush=True)
            continue
        np.savez_compressed(
            os.path.join(outdir, f'{tag}__{name}.npz'),
            cell=name, dataset=tag, temp=temp, charge_c=cc, dis_c=dc,
            cycle=np.array([r['cycle'] for r in rows]),
            cap=np.array([r['cap'] for r in rows]),
            v_dis_end=np.array([r['v_dis_end'] for r in rows]),
            v_rest0=np.array([r['v_rest0'] for r in rows]),
            qc=np.array([r['qc'] for r in rows]),
            n_rest=np.array([r['n_rest'] for r in rows]),
            relax=np.stack([r['relax'] for r in rows]),
            grid=GRID)
        print(f'{tag} {name}: {len(rows)} cycles, cap {rows[0]["cap"]:.0f} -> {rows[-1]["cap"]:.0f} mAh', flush=True)


if __name__ == '__main__':
    main()
