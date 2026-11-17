"""Paper figures (Figures 2-6 of the article) from the saved results, as PNG 300 dpi, PDF and SVG, into figures/."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)
C = {'D1': '#2a78d6', 'D2': '#eb6834', 'D3': '#1baf7a'}          # validated categorical slots 1-3
MK = {'D1': 'o', 'D2': 's', 'D3': '^'}                            # second encoding for print / CVD
NAME = {'D1': 'Dataset 1 (NCA)', 'D2': 'Dataset 2 (NCM)', 'D3': 'Dataset 3 (NCM+NCA)'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 1.6,
    'lines.markersize': 5, 'savefig.dpi': 300, 'figure.dpi': 150,
})


def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def panel_label(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=10, color=INK)


# ---- Figure 2: routing errors sit at the threshold (step 2, Gaussian process, point rule) ----
bins = [0, 0.5, 1, 2, 3, 5, 100]
labels = ['<0.5', '0.5–1', '1–2', '2–3', '3–5', '>5']
fig, ax = plt.subplots(figsize=(3.4, 2.4))
x = np.arange(len(labels)); wbar = 0.38
for j, ds in enumerate(['D1', 'D2']):
    P = pd.read_pickle(f'data/preds_step2_{ds}.pkl')
    dist = (P.soh_nom - 80).abs()
    wrong = ((P.soh_nom >= 80) != (P.gp >= 80)).astype(float)
    grp = pd.cut(dist, bins, right=False)
    share = (wrong * P.w).groupby(grp, observed=False).sum() / P.w.groupby(grp, observed=False).sum()
    ax.bar(x + (j - 0.5) * wbar, 100 * share.to_numpy(), wbar - 0.04, color=C[ds], label=NAME[ds])
ax.set_xticks(x, labels)
ax.set_xlabel('Distance of true SoH from the 80% threshold (points)')
ax.set_ylabel('Batteries misrouted (%)')
ax.legend(loc='upper right')
save(fig, 'Figure2_errors_at_threshold')

# ---- Figure 3: policies at the illustrative costs (step 3, Gaussian process) ----
R = pd.read_csv('data/results_step3_grid.csv')
R = R[(R.model == 'gp') & (R.C_FN == 5) & (R.C_FP == 20)]
pols = ['point', 'two-way by cost', 'screen then test', 'three-way (ours)', 'retest all']
plabels = ['Point\nrule', 'Two-way\nby cost', 'Screen\nthen test', 'Three-way\n(ours)', 'Retest\nall']
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, metric, ylab, scale in [(axes[0], 'cost_per_battery', 'Cost per battery (full tests)', 1),
                                (axes[1], 'retest_share', 'Batteries retested (%)', 100)]:
    x = np.arange(len(pols))
    for j, ds in enumerate(['D1', 'D2']):
        v = [scale * R[(R.data == ds) & (R.policy == p)][metric].iloc[0] for p in pols]
        ax.bar(x + (j - 0.5) * wbar, v, wbar - 0.04, color=C[ds], label=NAME[ds])
    i = pols.index('three-way (ours)')
    vals = [scale * R[(R.data == ds) & (R.policy == pols[i])][metric].iloc[0] for ds in ['D1', 'D2']]
    txt = ' / '.join(f'{v:.2f}' if scale == 1 else f'{v:.0f}%' for v in vals)
    ax.text(i, max(vals) + 0.03 * ax.get_ylim()[1], txt, ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.set_xticks(x, plabels, fontsize=7.8)
    ax.set_ylabel(ylab)
panel_label(axes[0], 'a'); panel_label(axes[1], 'b')
axes[0].legend(loc='upper left')
fig.tight_layout(w_pad=2.5)
save(fig, 'Figure3_policies')

# ---- Figure 4: rest length (step 5a) ----
T = pd.read_csv('data/results_step5_time.csv')
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ds in ['D1', 'D2', 'D3']:
    t = T[T.data == ds]
    axes[0].plot(t.rest_min, t.MAE, color=C[ds], marker=MK[ds], label=NAME[ds])
    axes[1].plot(t.rest_min, t.cost_ours_5_20, color=C[ds], marker=MK[ds], label=NAME[ds] + ', three-way')
    axes[1].plot(t.rest_min, t.cost_point_5_20, color=C[ds], marker=MK[ds], ls='--', lw=1.1, mfc='white', alpha=0.9)
axes[1].axhline(1.0, color=MUTED, lw=1, ls=':')
axes[1].text(0.52, 1.03, 'retest every battery', color=MUTED, fontsize=7.5, va='bottom')
for ax in axes:
    ax.set_xscale('log'); ax.set_xticks([0.5, 1, 2, 4, 10, 26, 59], ['0.5', '1', '2', '4', '10', '26', '59'])
    ax.minorticks_off(); ax.set_xlabel('Rest after full charge (min)')
axes[0].set_ylabel('SoH error, MAE (points)'); axes[0].set_ylim(0, None)
axes[1].set_ylabel('Cost per battery (full tests)'); axes[1].set_ylim(0, 1.5)
axes[0].legend(loc='lower left')
from matplotlib.lines import Line2D  # noqa: E402
axes[1].legend(handles=[Line2D([], [], color=MUTED, lw=1.6, label='Three-way (ours)'),
                        Line2D([], [], color=MUTED, lw=1.1, ls='--', label='Point rule')], loc='upper right')
panel_label(axes[0], 'a'); panel_label(axes[1], 'b')
fig.tight_layout(w_pad=2.5)
save(fig, 'Figure5_rest_length')

# ---- Figure 5: new chemistry (step 5b) ----
L = pd.read_csv('data/results_step5_loco.csv')
same = T[T.rest_min == 26].set_index('data')
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
off = {'D1': -0.12, 'D2': 0.0, 'D3': 0.12}
for ds in ['D1', 'D2', 'D3']:
    g = L[L.target == ds].groupby('k_target_cells')
    for ax, col, ref in [(axes[0], 'MAE', 'MAE'), (axes[1], 'cost_ours_5_20', 'cost_ours_5_20')]:
        m, lo, hi = g[col].mean(), g[col].min(), g[col].max()
        xk = np.arange(len(m)) + off[ds]
        ax.errorbar(xk, m, yerr=[m - lo, hi - m], color=C[ds], marker=MK[ds], capsize=2, lw=1.6, elinewidth=1, label=NAME[ds])
        ax.axhline(same.loc[ds, ref], color=C[ds], lw=1, ls=':')
axes[1].axhline(1.0, color=MUTED, lw=1, ls='--')
axes[1].text(2.95, 1.04, 'retest every battery', color=MUTED, fontsize=7.5, ha='right', va='bottom')
for ax in axes:
    ax.set_xticks(range(4), ['0', '1', '2', '4']); ax.set_xlabel('Cells of the new chemistry in training')
axes[0].set_ylabel('SoH error, MAE (points)'); axes[0].set_ylim(0, None)
axes[1].set_ylabel('Cost per battery (full tests)'); axes[1].set_ylim(0, 2.1)
axes[0].legend(loc='upper right')
axes[1].text(0.99, 0.97, 'dotted lines: trained on the same chemistry', transform=axes[1].transAxes, ha='right', va='top', fontsize=7.2, color=MUTED)
panel_label(axes[0], 'a'); panel_label(axes[1], 'b')
fig.tight_layout(w_pad=2.5)
save(fig, 'Figure6_new_chemistry')
print(sorted(os.listdir(OUT)))

# ---- Figure 6: savings map over error costs, with the literature-based region (case_study.py) ----
import json  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
CS = json.load(open('data/results_case_study.json'))
cmap = LinearSegmentedColormap.from_list('div', [(0, '#b8312f'), (0.25, '#e98e8c'), (0.5, '#f0efec'), (0.62, '#b7d3f6'), (0.75, '#5598e7'), (0.88, '#1c5cab'), (1, '#0d366b')])
norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=70)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, ds in zip(axes, ['D1', 'D2']):
    m = CS[ds]['map']; fn, fp, S = np.array(m['C_FN']), np.array(m['C_FP']), np.array(m['saving_pct'])
    pc = ax.pcolormesh(fn, fp, S, cmap=cmap, norm=norm, shading='nearest', rasterized=True)
    ax.contour(fn, fp, S, levels=[25, 50], colors=['white', 'white'], linewidths=[0.9, 0.9], linestyles=['--', '-'])
    ax.add_patch(Rectangle((0.5, 2.85), 4.62 - 0.5, 9.46 - 2.85, fill=False, ec=INK, lw=1.2, ls='--'))
    ax.text(0.55, 10.1, 'literature range', fontsize=7.2, color=INK, va='bottom')
    ax.axvline(1.0, color=INK, lw=0.8, ls=':')
    for (x, y, lab) in [(0.62, 5.08, 'central'), (4.62, 9.46, 'high margin'), (5, 20, 'illustrative')]:
        ax.plot(x, y, marker='o', ms=5, mfc='white', mec=INK, mew=1.2)
        ax.text(x * 1.12, y * 1.06, lab, fontsize=7.2, color=INK if lab != 'illustrative' else 'white')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(False)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20], ['0.5', '1', '2', '5', '10', '20'])
    ax.set_yticks([1, 2, 5, 10, 20, 50], ['1', '2', '5', '10', '20', '50'])
    ax.set_xlabel('C_FN: recycling an eligible battery\n(in full tests)')
    ax.set_title(NAME[ds], fontsize=8.5, color=INK)
axes[0].set_ylabel('C_FP: second-life failure (in full tests)')
cb = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.02, ticks=[-10, 0, 20, 40, 60])
cb.set_label('Saving against the best other policy (%)')
panel_label(axes[0], 'a'); panel_label(axes[1], 'b')
save(fig, 'Figure4_cost_map')
print('fig6 done')
