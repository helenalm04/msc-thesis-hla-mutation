# Chapter 6, the mean profile figure. Panel (a) averages over allele pairs
# across the JSD range, panel (b) over endpoint pairs within one transition.
# The arrays come from mean_profile.py; this only draws them.

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

from config import FIGURES_DIR, FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND

BLUE, ORANGE = '#5C82D6', '#EA9542'
PANELS = [
    ('poster_meanprofile_pairs.npz',
     '(a) 40 allele pairs across the JSD range'),
    ('poster_meanprofile_endpoints_b5101_b2705.npz',
     '(b) 12 endpoint pairs, B*51:01 to B*27:05'),
]


# standard error of the mean over the entries along one axis
def _se(a, axis=0):
    return a.std(axis=axis, ddof=1) / np.sqrt(a.shape[axis])


def main():
    '''Draw both panels of the mean profile figure from the saved arrays.'''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (fname, title) in zip(axes, PANELS):
        z = np.load(os.path.join(FIGURES_DIR, fname))
        grid, direct, globl = z['grid'], z['direct'], z['globl']
        slack, lam = int(z['slack']), float(z['lam'])

        for curve, colour, label in ((direct, BLUE, r'direct  ($T = D$)'),
                                     (globl, ORANGE, rf'detour  ($T = D + {slack}$)')):
            m, s = curve.mean(axis=0), _se(curve)
            ax.plot(grid, m, '-', color=colour, lw=2.4, label=label, zorder=3)
            ax.fill_between(grid, m - s, m + s, color=colour, alpha=0.22, lw=0)

        gap = direct.max(axis=1) - globl.max(axis=1)  # goes in the caption, not on the axes
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xlabel('position along the path', fontsize=FS_LABEL)
        ax.set_ylabel(r'$\langle\,\min(E_A, E_B)\,\rangle$  (nats)', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.margins(y=0.14)
        print(f"{fname}: slack={slack} Lambda={lam} n={len(direct)} "
              f"gap={gap.mean():+.2f}+/-{_se(gap):.2f} lower={100 * (gap > 0).mean():.0f}%")

    # an in-axes legend covers the peak of the direct curve
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FS_LEGEND, frameon=False, ncol=2,
               loc='lower center', bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout(w_pad=2.4)
    out = os.path.join(FIGURES_DIR, 'mean_profile.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=180, bbox_inches='tight')
    print(f"figure -> {out}")


if __name__ == '__main__':
    main()
