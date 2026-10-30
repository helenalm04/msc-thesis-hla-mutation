# Chapter 6, where paths stop going straight. Lambda is the cost of a
# mutation, so sweeping it trades the two routes off: cheap mutations let a
# path detour, expensive ones force it straight. The figure marks the last
# cost at which no path stays direct and the first at which none detours.
# Reads both tables from src/scans/scan_direct_global_transition.py.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (RESULTS_DIR, FIGURES_DIR, FS_LABEL, FS_TITLE,
                    FS_LEGEND, FS_TICK)

CSV = os.path.join(RESULTS_DIR, 'direct_global_transition.csv')
CSV_FIXED = os.path.join(RESULTS_DIR, 'direct_global_transition_fixedbeta.csv')

STYLE = {
    'anchor-incompatible': ('#EA9542', 'o', 'anchor-incompatible\nB*51:01 $\\to$ B*27:05'),
    'same-supertype':      ('#5C82D6', 's', 'same-supertype\nA*02:01 $\\to$ A*02:04'),
}


def phase_label_anchors(df, tol=0.01):
    '''Return the last tension with no direct path and the first with no indirect path.'''
    # take the most favourable endpoint pair at each cost, so a boundary is
    # only crossed once no pair at all manages that route
    g = df.groupby('Lambda').agg(f=('frac_direct', 'max'),
                                 d=('dds_mean', 'max')).sort_index()
    glob = g.index[g['f'] <= 0.0]
    dirc = g.index[g['d'] <= tol]
    return (float(glob[-1]) if len(glob) else None,
            float(dirc[0]) if len(dirc) else None)


def main():
    df = pd.read_csv(CSV)
    df = df[df['system'].isin(STYLE)]  # the scan also writes a single-allele system
    fig, (ax, axf) = plt.subplots(1, 2, figsize=(11.6, 4.9))
    ctrl = pd.read_csv(CSV_FIXED) if os.path.exists(CSV_FIXED) else df
    ctrl = ctrl[ctrl['system'].isin(STYLE)]

    for sysname, (col, mk, lab) in STYLE.items():
        d = df[df['system'] == sysname].sort_values('Lambda')
        if d.empty:
            continue
        ax.plot(d['Lambda'], d['dds_mean'], marker=mk, color=col, lw=2.0, ms=7,
                label=lab, zorder=3)

    lams = sorted(df['Lambda'].unique())
    keep, last = [lams[0]], np.log10(lams[0])
    for v in lams[1:-1]:
        if np.log10(v) - last >= 0.15:  # the refined cells sit a few percent apart
            keep.append(v)
            last = np.log10(v)
    keep.append(lams[-1])
    pad = (lams[1] / lams[0]) ** 0.35  # keep the end points off the spines

    def style_x(a):
        '''Apply the shared log tension axis to one panel.'''
        a.set_xscale('log')
        a.set_xticks(lams)
        a.set_xticklabels([f'{v:g}' if v in keep else '' for v in lams])  # label only the ticks far enough apart in log space
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a.set_xlim(lams[0] / pad, lams[-1] * pad)
        a.set_xlabel(r'elastic tension  $\beta\Lambda$', fontsize=FS_LABEL)
        a.tick_params(labelsize=FS_TICK)

    style_x(ax)
    ax.set_ylim(bottom=-0.05, top=ax.get_ylim()[1] + 0.30)  # headroom for the labels

    lam_glob, lam_dir = phase_label_anchors(df)
    xlo, xhi = ax.get_xlim()
    blend = ax.get_xaxis_transform()
    if lam_glob is not None:
        ax.text(np.sqrt(xlo * lam_glob), 0.965, 'global space', transform=blend,
                ha='center', va='top', fontsize=FS_TICK - 1, color='0.45')
    if lam_dir is not None:
        ax.text(np.sqrt(lam_dir * lams[-1]), 0.965, 'direct\nspace', transform=blend,
                ha='center', va='top', fontsize=FS_TICK - 1, color='0.45')
    ax.set_xlim(xlo, xhi)
    ax.set_ylabel(r'$\langle d_{\mathrm{DS}}\rangle$', fontsize=FS_LABEL)
    ax.set_title('(a)', loc='left', fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND - 4, frameon=False, loc='lower left')

    for sysname, (col, mk, _) in STYLE.items():
        d = df[df['system'] == sysname].sort_values('Lambda')
        c = ctrl[ctrl['system'] == sysname].sort_values('Lambda')
        if not d.empty:
            axf.plot(d['Lambda'], d['frac_direct'], marker=mk, color=col,
                     lw=2.0, ms=6, zorder=3)
        if not c.empty:
            axf.plot(c['Lambda'], c['frac_direct'], marker=mk, color=col,
                     lw=1.5, ms=6, ls='--', mfc='white', mew=1.5, zorder=2)
    axf.plot([], [], color='0.35', lw=2.0, label='annealed')
    axf.plot([], [], color='0.35', lw=1.5, ls='--', label=r'fixed $\beta$')
    style_x(axf)
    axf.set_ylim(-0.04, 1.04)
    axf.set_ylabel('fraction of paths with $d_{\\mathrm{DS}} = 0$',
                   fontsize=FS_LABEL - 2)
    axf.set_title('(b)', loc='left', fontsize=FS_TITLE)
    axf.legend(fontsize=FS_LEGEND - 4, frameon=False, loc='upper left')

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, 'direct_global_transition.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")

    # reported, not plotted
    both = df.dropna(subset=['barrier_direct', 'barrier_global'])
    for sysname, (_, _, lab) in STYLE.items():
        d = both[both['system'] == sysname]
        if d.empty:
            continue
        cell = d.iloc[(d['frac_direct'] - 0.5).abs().argsort().iloc[0]]
        gap = float(cell['barrier_direct'] - cell['barrier_global'])
        print(f"  {lab.splitlines()[0]:>22}  coexistence cell "
              f"beta*Lambda={cell['Lambda']:g}  gap {gap:+.3f}  "
              f"<d_DS>={cell['dds_mean']:.2f}")


if __name__ == '__main__':
    main()
