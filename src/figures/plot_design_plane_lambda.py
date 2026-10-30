# Chapter 4, designing peptides instead of sampling paths between them.
# Lambda sets what is asked for: below zero a peptide that binds one allele
# and not the other, above zero one that binds both. Each design is plotted by
# its two energies, so the lambda series sweeps out a front across the plane.
# Reads the designs from src/scans/design_lambda.py.

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import RESULTS_DIR, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK

CACHE = os.path.join(RESULTS_DIR, 'design_lambda.npz')
OUT = os.path.join(FIGURES_DIR, 'design_superbinder')

C_A, C_B = '#5C82D6', '#EA9542'  # allele A and allele B ligands

LAMBDA_COLOUR = {  # adjacent steps are 20 to 32 points of L* apart
    -0.50: '#4A1D6B',
    -0.25: '#9B6BC4',
     0.00: '#CFCFCF',
     0.50: '#5FAD56',
     1.00: '#14532D',
}
LAMS = [1.00, 0.50, 0.00, -0.25, -0.50]  # draw order, +1 ends up on top

PAD = 1.5  # limits are the data range plus a pad

PANEL = {  # where the key fits without covering designs
    'overlapping': dict(tag='overlapping motifs, JSD $0.31$', leg='upper right'),
    'orthogonal':  dict(tag='conflicting motifs, JSD $0.60$', leg='upper right'),
}


# legend text for one value of the regime parameter
def lam_label(lam):
    return rf'$\lambda={lam:+.2f}$'.replace('+0.00', '0')  # a signed zero reads wrong in the key


def panel(ax, z, label):
    '''Draw the design plane for one allele pair.'''
    g = lambda k: z[f'{label}__{k}']
    a, b = str(g('a')), str(g('b'))
    cfg = PANEL[label]

    des = {lam: (z[f'{label}__lam{lam}__eA'], z[f'{label}__lam{lam}__eB']) for lam in LAMS}
    lig = [g(k) for k in ('ligA_A', 'ligA_B', 'ligB_A', 'ligB_B')]
    # each panel is fitted to its own content
    xs_all = [lig[0], lig[2]] + [d[0] for d in des.values()]
    ys_all = [lig[1], lig[3]] + [d[1] for d in des.values()]
    xlim = (min(map(np.min, xs_all)) - PAD, max(map(np.max, xs_all)) + PAD)
    ylim = (min(map(np.min, ys_all)) - PAD, max(map(np.max, ys_all)) + PAD)

    ax.scatter(lig[0], lig[1], s=7, color=C_A, alpha=0.28, lw=0,
               label=f'{a} ligands', zorder=1)
    ax.scatter(lig[2], lig[3], s=7, color=C_B, alpha=0.28, lw=0,
               label=f'{b} ligands', zorder=1)

    cen = {}
    for lam in LAMS:
        eA, eB = des[lam]
        ax.scatter(eA, eB, s=15, color=LAMBDA_COLOUR[lam], alpha=0.55, lw=0,
                   zorder=3, label=lam_label(lam))
        cen[lam] = (float(np.median(eA)), float(np.median(eB)))

    for lam in LAMS:
        ax.plot(*cen[lam], 'o', ms=12, color=LAMBDA_COLOUR[lam], markeredgecolor='black',
                markeredgewidth=1.2, zorder=5,
                label='median design' if lam == LAMS[0] else None)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(f'{a} $+$ {b},  {cfg["tag"]}', fontsize=FS_TITLE - 1)
    ax.set_xlabel(f'energy under {a}  (nats)', fontsize=FS_LABEL)
    ax.set_ylabel(f'energy under {b}  (nats)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    # lambda is not labelled inside the axes
    h, lb = ax.get_legend_handles_labels()
    med = Line2D([], [], ls='none', marker='o', ms=9, color='0.6',
                 markeredgecolor='black', markeredgewidth=1.2)
    key = ax.legend(h[:2] + [med], lb[:2] + ['median of a $\\lambda$ cloud'],
                    fontsize=FS_LEGEND - 3, loc=cfg['leg'], framealpha=0.9,
                    handletextpad=0.4, labelspacing=0.5)
    for hd in key.legend_handles[:2]:
        hd.set_sizes([46]); hd.set_alpha(0.85)
    ax.add_artist(key)
    lam_i = [i for i, x in enumerate(lb) if x.startswith('$\\lambda')]
    return [h[i] for i in lam_i], [lb[i] for i in lam_i]


def main():
    '''Draw both panels and print a summary for each lambda.'''
    z = np.load(CACHE, allow_pickle=True)
    labels = [str(x) for x in z['labels']]

    fig, axes = plt.subplots(1, len(labels), figsize=(6.6 * len(labels), 6.6))
    axes = np.atleast_1d(axes)
    handles = None
    for ax, label in zip(axes, labels):
        h, lb = panel(ax, z, label)
        if handles is None:
            handles, hl = h, lb

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles, hl, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5,
               frameon=False, fontsize=FS_LEGEND, markerscale=2.4,
               handletextpad=0.3, columnspacing=1.8)

    for label in labels:
        g = lambda k: z[f'{label}__{k}']
        print(f'{label}: {g("a")} + {g("b")}  bars {float(g("thrA")):+.2f}/{float(g("thrB")):+.2f}')
        for lam in LAMS:
            eA, eB = z[f'{label}__lam{lam}__eA'], z[f'{label}__lam{lam}__eB']
            print(f'   lam {lam:>+5.2f}  n {eA.size:>4d}  median ({np.median(eA):+6.2f},'
                  f' {np.median(eB):+6.2f})  dual-presented '
                  f'{100*float(z[f"{label}__lam{lam}__f_super"]):>3.0f}%')

    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', dpi=200, bbox_inches='tight')
    print(f'Wrote {OUT}.pdf and {OUT}.png')


if __name__ == '__main__':
    main()
