"""Step 4: deep ensemble of small 1D-CNNs on the raw rest curve.

Input: the 14-point rest curve (0-26 min, every 2 min) as two channels per time step:
the voltage and its drop since the start of the rest, both standardised on training cells.
Output: mean and variance of SoH (Gaussian negative log-likelihood). Five networks are
trained in parallel with different seeds and batches; their mixture gives the prediction.
Minibatches are drawn with equal probability per cell (D-14). Same outer folds as steps 2-3;
the spread is calibrated by cross-fitting on the training cells as in step 3.
Usage: python3 -I -u cnn.py [pilot]
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import linen as nn
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, '.')
from baselines import GRID  # noqa: E402
from calibrate import scale_factor  # noqa: E402
from load import relax_matrix  # noqa: E402

M, BATCH, LR = 5, 256, 1e-3


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        for ch in (16, 32, 32):
            x = nn.relu(nn.Conv(ch, (3,), padding='SAME')(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(64)(x))
        out = nn.Dense(2)(x)
        return out[:, 0], jnp.clip(out[:, 1], -8.0, 4.0)


MODEL = CNN()
OPT = optax.adamw(LR, weight_decay=1e-4)


def nll(params, x, y):
    mu, lv = MODEL.apply(params, x)
    return jnp.mean(0.5 * lv + 0.5 * (y - mu) ** 2 * jnp.exp(-lv))


@jax.jit
def train_step(params, opt_state, x, y):
    loss, g = jax.value_and_grad(nll)(params, x, y)
    upd, opt_state = OPT.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss


v_step = jax.jit(jax.vmap(train_step))
v_apply = jax.jit(jax.vmap(lambda p, x: MODEL.apply(p, x), in_axes=(0, None)))


class Prep:
    """Two channels per time step, standardised on the training cells; target standardised too."""

    def __init__(self, V, y):
        D = V - V[:, :1]
        self.vm, self.vs = V.mean(0), V.std(0) + 1e-9
        self.dm, self.ds = D.mean(0), D.std(0) + 1e-9
        self.ym, self.ys = y.mean(), y.std()

    def x(self, V):
        D = V - V[:, :1]
        return np.stack([(V - self.vm) / self.vs, (D - self.dm) / self.ds], -1).astype(np.float32)


def fit_ensemble(V, y, cell, steps, seed=0):
    prep = Prep(V, y)
    X, Y = jnp.asarray(prep.x(V)), jnp.asarray(((y - prep.ym) / prep.ys).astype(np.float32))
    n_cell = pd.Series(cell).map(pd.Series(cell).value_counts()).to_numpy()
    prob = (1.0 / n_cell) / np.sum(1.0 / n_cell)            # equal probability per cell
    keys = jax.random.split(jax.random.PRNGKey(seed), M)
    params = jax.vmap(lambda k: MODEL.init(k, X[:2]))(keys)
    opt_state = jax.vmap(OPT.init)(params)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.choice(len(Y), size=(M, BATCH), p=prob)
        params, opt_state, _ = v_step(params, opt_state, X[idx], Y[idx])

    def predict(Vn):
        mu, lv = v_apply(params, jnp.asarray(prep.x(Vn)))
        mu, var = np.asarray(mu), np.exp(np.asarray(lv))
        m = mu.mean(0)
        v = (var + mu ** 2).mean(0) - m ** 2
        return prep.ym + prep.ys * m, prep.ys * np.sqrt(v)
    return predict


def prepare(ok, ds):
    d = ok[ok.ds == ds].copy()
    d['cond'] = d.temp.astype(str) + '_' + d.chg_rate.astype(str) + '_' + d.dis_rate.astype(str)
    d['w'] = 1.0 / d.groupby('cell_id').cycle.transform('size')
    V = relax_matrix(d, GRID)
    return d, V


def pilot(ok, ds=1):
    """Choose the number of steps on an inner split of outer fold 0's training cells only."""
    d, V = prepare(ok, ds)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    tr_i, _ = next(outer.split(d, d.cond, d.cell_id))
    tr, Vtr = d.iloc[tr_i], V[tr_i]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
    fi, vi = next(inner.split(tr, tr.cond, tr.cell_id))
    yv, wv = tr.iloc[vi].soh_nom.to_numpy(), tr.iloc[vi].w.to_numpy()
    for steps in [500, 1500, 3000, 6000]:
        t0 = time.time()
        pr = fit_ensemble(Vtr[fi], tr.iloc[fi].soh_nom.to_numpy(), tr.iloc[fi].cell_id.to_numpy(), steps)
        mu, sd = pr(Vtr[vi])
        mae = np.sum(wv * np.abs(mu - yv)) / wv.sum()
        print(f'pilot steps {steps}: validation MAE {mae:.3f}, {time.time() - t0:.0f} s', flush=True)


def run(ok, ds, steps):
    d, V = prepare(ok, ds)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    out = []
    for k, (tr_i, te_i) in enumerate(outer.split(d, d.cond, d.cell_id)):
        t0 = time.time()
        tr, te, Vtr = d.iloc[tr_i], d.iloc[te_i], V[tr_i]
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
        oof = []
        for j, (fi, ci) in enumerate(inner.split(tr, tr.cond, tr.cell_id)):
            pr = fit_ensemble(Vtr[fi], tr.iloc[fi].soh_nom.to_numpy(), tr.iloc[fi].cell_id.to_numpy(), steps, seed=10 * k + j)
            mu, sd = pr(Vtr[ci])
            oof.append(pd.DataFrame({'y': tr.iloc[ci].soh_nom.to_numpy(), 'w': tr.iloc[ci].w.to_numpy(), 'm': mu, 's': sd}))
        oof = pd.concat(oof)
        s = scale_factor(oof['y'].to_numpy(), oof['m'].to_numpy(), oof['s'].to_numpy(), oof['w'].to_numpy())
        pr = fit_ensemble(Vtr, tr.soh_nom.to_numpy(), tr.cell_id.to_numpy(), steps, seed=10 * k + 9)
        mu, sd = pr(V[te_i])
        o = te[['cell_id', 'cycle', 'cond', 'soh_nom', 'w']].copy()
        o['fold'], o['cnn_mu'], o['cnn_sd'], o['cnn_s'] = k, mu, sd, s
        out.append(o)
        print(f'D{ds} fold {k}: scale {s:.2f}; {time.time() - t0:.0f} s', flush=True)
    return pd.concat(out)


if __name__ == '__main__':
    ok = pd.read_pickle('data/clean.pkl')
    if len(sys.argv) > 1 and sys.argv[1] == 'pilot':
        pilot(ok)
    else:
        steps = int(sys.argv[1])
        for ds in [1, 2]:
            run(ok, ds, steps).to_pickle(f'data/preds_step4_D{ds}.pkl')
        print('EXIT OK')
