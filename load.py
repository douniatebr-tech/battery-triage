"""Load the extract, clean it, and add SoH labels."""
import numpy as np
import pandas as pd

NOMINAL = {1: 3500.0, 2: 3500.0, 3: 2500.0}      # mAh, from the C-rate definitions in Zhu et al. 2022
CUTOFF = {1: 2.65, 2: 2.50, 3: 2.50}             # discharge cut-off voltage seen in the data
REST_LEN = {1: 1560.0, 2: 1560.0, 3: 3540.0}     # logged rest length (s)


def load(path='data/zhu_relaxation_extract.csv.gz'):
    df = pd.read_csv(path)
    df['ds'] = df.dataset.str.extract(r'Dataset_(\d)').astype(int)
    m = df.cell.str.extract(r'CY(\d+)-(\d+)_(\d+)-#(\d+)')
    df['temp'] = m[0].astype(int)
    df['chg_rate'] = m[1].map(lambda s: float('0.' + s[1:]) if s.startswith('0') else float(s))
    df['dis_rate'] = m[2].astype(float)
    df['cell_id'] = 'D' + df.ds.astype(str) + '_' + df.cell
    df['rest_dur'] = df.rest_t.str.split(';').map(lambda s: float(s[-1]))
    # row order within a cell (one cell, D1_CY25-05_1-#16, restarts its cycle count after two cycles)
    df['seq'] = df.groupby('cell_id').cumcount() + 1
    return df


def clean(df):
    """Flag cycles that are not usable; returns df with 'ok' and 'reason'."""
    df = df.sort_values(['cell_id', 'cycle']).copy()
    reason = pd.Series('', index=df.index)
    ts = df.rest_t.str.split(';').map(lambda s: np.array(s, float))
    max_gap = ts.map(lambda t: np.diff(t).max() if len(t) > 1 else np.inf)
    step = df.ds.map({1: 120.0, 2: 120.0, 3: 30.0})
    full_rest = (df.rest_dur >= df.ds.map(REST_LEN) - 5) & (max_gap <= 1.1 * step)
    reason[~full_rest] = 'rest short or interrupted'
    vs = df.rest_v.str.split(';').map(lambda s: np.array(s, float))
    # a rest after a full charge relaxes slowly: no sudden drops, no fall far below the start
    smooth = vs.map(lambda v: len(v) > 1 and np.diff(v).min() > -0.02 and v.min() > v[0] - 0.1 and v.max() < v[0] + 0.02)
    reason[(reason == '') & ~smooth] = 'rest curve not smooth'
    cut = (df.v_dis_end - df.ds.map(CUTOFF)).abs() < 0.01
    reason[(reason == '') & ~cut] = 'discharge not to cut-off'
    med = df.groupby('cell_id').cap_mAh.transform(lambda s: s.rolling(9, center=True, min_periods=3).median())
    jump = (df.cap_mAh / med - 1).abs() > 0.03
    reason[(reason == '') & jump] = 'capacity jump vs neighbours'
    df['reason'] = reason
    df['ok'] = reason == ''
    return df


def add_soh(df):
    ok = df[df.ok].sort_values(['cell_id', 'cycle'])
    init = ok.groupby('cell_id').head(3).groupby('cell_id').cap_mAh.median()
    df['cap_init'] = df.cell_id.map(init)
    df['soh_init'] = 100 * df.cap_mAh / df.cap_init
    df['soh_nom'] = 100 * df.cap_mAh / df.ds.map(NOMINAL)
    return df


def relax_matrix(df, grid):
    """Interpolate each rest curve onto a common time grid (s)."""
    out = np.empty((len(df), len(grid)))
    for i, (t, v) in enumerate(zip(df.rest_t, df.rest_v)):
        t = np.fromstring(t, sep=';'); v = np.fromstring(v, sep=';')
        out[i] = np.interp(grid, t, v)
    return out


if __name__ == '__main__':
    # Builds data/clean.pkl, the input of every later step: 58,806 usable cycles from 130 cells.
    df = add_soh(clean(load()))
    print(df.reason.replace('', 'kept').value_counts().to_string())
    df[df.ok].to_pickle('data/clean.pkl')
    print('wrote data/clean.pkl with', int(df.ok.sum()), 'cycles')
