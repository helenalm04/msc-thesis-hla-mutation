# Chapter 6, the B*51:01 corridor figure. Each panel draws the direct barrier
# as a flat line and the indirect barrier as a curve, so the gap between them
# is the gain from being allowed to detour. Reads the --bulk tables from
# src/b51_either_sweep.py and src/b51/b51_position_diff.py.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND

COL = {'HLA-B*27:05': '#5C82D6', 'HLA-B*27:07': '#EA9542'}  # separated by OKLab dE 23.7 under tritanopia
LABEL = {'HLA-B*27:05': 'B*27:05 (corridor)', 'HLA-B*27:07': 'B*27:07 (valley)'}
ORDER = ['HLA-B*27:05', 'HLA-B*27:07']


def _levels(ax, df, xcol, xvals, xlabel, panel):
    '''Draw the direct barrier as a flat line and the indirect barrier as a curve.'''
    for target in ORDER:
        g = df[(df['target'] == target) & (df[xcol].isin(xvals))].sort_values(xcol)
        if g.empty:
            continue
        c = COL[target]
        # the direct barrier does not depend on the swept variable, so one
        # value serves for the whole panel and is drawn as a level
        direct = g['direct_worst'].iloc[0]

        ax.axhline(direct, color=c, lw=2.0, ls=(0, (5, 3)), zorder=2)
        ax.fill_between(g[xcol], direct, g['indirect_worst'],
                        color=c, alpha=0.13, lw=0, zorder=1)
        ax.errorbar(g[xcol], g['indirect_worst'], yerr=g['indirect_worst_std'],
                    marker='o', ms=8, lw=2.0, capsize=3, color=c,
                    ecolor='0.55', elinewidth=0.8, mec='black', mew=0.6,
                    label=LABEL[target], zorder=3)

    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_title(panel, fontsize=FS_TITLE, loc='left')
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis='y', color='0.9', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _posdiff(ax, pdf, panel):
    positions = sorted(pdf['pos'].unique())
    x = np.arange(len(positions))
    w = 0.38
    for target, sign in [(ORDER[0], -1), (ORDER[1], +1)]:
        g = pdf[pdf['target'] == target].sort_values('pos')
        ax.bar(x + sign * w / 2, 100 * g['off_frac'], w, color=COL[target],
               edgecolor='white', linewidth=1.0, label=LABEL[target], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{p}' for p in positions])
    ax.set_xlabel('peptide position', fontsize=FS_LABEL)
    ax.set_ylabel('paths off direct line (%)', fontsize=FS_LABEL)
    ax.set_title(panel, fontsize=FS_TITLE, loc='left')
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis='y', color='0.9', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main():
    sweep_csv = os.path.join(RESULTS_DIR, 'b51_either_sweep_bulk.csv')
    if not os.path.exists(sweep_csv):
        sweep_csv = os.path.join(RESULTS_DIR, 'b51_either_sweep.csv')
    pos_csv = os.path.join(RESULTS_DIR, 'b51_position_diff.csv')
    for f in (sweep_csv, pos_csv):
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
    df = pd.read_csv(sweep_csv)
    pdf = pd.read_csv(pos_csv)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.6))

    _levels(axes[0], df[df['k'] == 4], 'lambda', [0.25, 0.5, 1.0],
            r'elastic penalty  $\Lambda$', r'(a)  $T = D+4$')
    _levels(axes[1], df[df['lambda'] == 0.5], 'k', [2, 4, 6],
            r'slack  $T - D$', r'(b)  $\Lambda = 0.5$')

    lo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    hi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes[:2]:
        ax.set_ylim(lo, hi)
        ax.set_xmargin(0.16)
    axes[0].set_ylabel('worst-either barrier (nats)', fontsize=FS_LABEL)
    axes[1].tick_params(labelleft=False)
    axes[1].set_xticks([2, 4, 6])

    _posdiff(axes[2], pdf, '(c)')

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=COL[ORDER[0]], lw=2.0, marker='o', ms=8, mec='black',
               mew=0.6, label=LABEL[ORDER[0]]),
        Line2D([], [], color=COL[ORDER[1]], lw=2.0, marker='o', ms=8, mec='black',
               mew=0.6, label=LABEL[ORDER[1]]),
        Line2D([], [], color='0.35', lw=2.0, ls=(0, (5, 3)), label='direct  ($T = D$)'),
        Line2D([], [], color='0.35', lw=2.0, marker='o', ms=8, mec='black',
               mew=0.6, label='indirect'),
    ]
    fig.legend(handles=handles, fontsize=FS_LEGEND - 1, frameon=False, ncol=4,
               loc='lower center', bbox_to_anchor=(0.5, -0.07))  # outside the axes to clear the error bars

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = os.path.join(FIGURES_DIR, f'b51_either_levels.{ext}')
        fig.savefig(out, bbox_inches='tight', dpi=160)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
