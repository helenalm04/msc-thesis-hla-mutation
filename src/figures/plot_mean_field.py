# Chapter 6, the mean-field path drawn over the sampled ensemble, with
# ellipses for the spread the solver predicts, so the figure shows where the
# approximation tracks the chain. Reads src/scans/mean_field_corridor.py.

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.append(_p)

from config import (FIGURES_DIR, FS_LABEL, FS_TITLE, FS_LEGEND,
                    FS_TICK)

CACHE = os.path.join(FIGURES_DIR, '_mean_field_corridor.npz')
OUT = os.path.join(FIGURES_DIR, 'mean_field_paths.pdf')

SYSTEMS = [  # key, panel title, regime, source allele, target allele
    ('HLA-B5101__HLA-B2705', 'B*51:01 $\\rightarrow$ B*27:05', 'corridor',
     'B*51:01', 'B*27:05'),
    ('HLA-B5101__HLA-B2707', 'B*51:01 $\\rightarrow$ B*27:07', 'valley',
     'B*51:01', 'B*27:07'),
    ('HLA-A0201__HLA-A0204', 'A*02:01 $\\rightarrow$ A*02:04', 'no barrier',
     'A*02:01', 'A*02:04'),
]

C_MF = '#000000'
C_MCMC = '#EA9542'


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(f'missing {CACHE}; run src/scans/mean_field_corridor.py')
    d = np.load(CACHE, allow_pickle=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    top_extent = []  # a shared span was tried and dropped

    for col, (key, title, label, src, tgt) in enumerate(SYSTEMS):
        ax = axes[0, col]
        xy_mf = d[f'{key}__xy_mf']
        xy_mc = d[f'{key}__xy_mcmc']
        sd = d[f'{key}__xy_mcmc_sd']

        assert len(xy_mf) == len(xy_mc) - 2, 'unexpected interior-node count'
        assert np.abs(sd[[0, -1]]).max() < 1e-3, 'endpoints are not fixed'  # float32, so about 1e-5 and not zero
        xy_mf = np.vstack([xy_mc[0], xy_mf, xy_mc[-1]])

        ax.plot(xy_mc[:, 0], xy_mc[:, 1], '-o', color=C_MCMC, ms=4, lw=2,
                label='sampler, mean path', zorder=3)
        for k in range(len(xy_mc)):  # overlapping ellipses darken
            ax.add_patch(Ellipse((xy_mc[k, 0], xy_mc[k, 1]),
                                 width=2 * sd[k, 0], height=2 * sd[k, 1],
                                 facecolor=C_MCMC, edgecolor='none', alpha=0.16,
                                 zorder=2,
                                 label='sampler, $\\pm 1$ s.d. across paths'
                                 if (k == 0 and col == 0) else None))
        mf_sd_key = f'{key}__xy_mf_sd'
        if mf_sd_key in d.files:  # older caches do not have it
            mf_sd = np.vstack([np.zeros(2), d[mf_sd_key], np.zeros(2)])
            for k in range(1, len(xy_mf) - 1):  # the endpoints have no spread
                ax.add_patch(Ellipse((xy_mf[k, 0], xy_mf[k, 1]),
                                     width=2 * mf_sd[k, 0], height=2 * mf_sd[k, 1],
                                     facecolor=C_MF, edgecolor='none', alpha=0.10,
                                     zorder=1,
                                     label='mean field, $\\pm 1$ s.d. across paths'
                                     if (k == 1 and col == 0) else None))
        ax.plot(xy_mf[:, 0], xy_mf[:, 1], '-s', color=C_MF, ms=4, lw=2,
                label='mean field', zorder=4)
        ax.plot(xy_mc[0, 0], xy_mc[0, 1], '*', color='k', ms=16, zorder=5,
                label='start $v_0$' if col == 0 else None)
        ax.plot(xy_mc[-1, 0], xy_mc[-1, 1], 'X', color='k', ms=11, zorder=5,
                label='end $v_T$' if col == 0 else None)

        ex = np.concatenate([xy_mc[:, 0] - sd[:, 0], xy_mc[:, 0] + sd[:, 0], xy_mf[:, 0]])
        ey = np.concatenate([xy_mc[:, 1] - sd[:, 1], xy_mc[:, 1] + sd[:, 1], xy_mf[:, 1]])
        if mf_sd_key in d.files:
            ex = np.concatenate([ex, xy_mf[:, 0] - mf_sd[:, 0], xy_mf[:, 0] + mf_sd[:, 0]])
            ey = np.concatenate([ey, xy_mf[:, 1] - mf_sd[:, 1], xy_mf[:, 1] + mf_sd[:, 1]])
        top_extent.append((ex.min(), ex.max(), ey.min(), ey.max()))

        ax.set_title(title, fontsize=FS_TITLE)  # the regime word is not shown
        ax.set_xlabel(f'$I_\\mu$, {src} unit', fontsize=FS_LABEL)
        ax.set_ylabel(f'$I_\\nu$, {tgt} unit', fontsize=FS_LABEL)  # each column has its own target allele
        ax.tick_params(labelsize=FS_TICK)

        ax = axes[1, col]
        mf_full = d[f'{key}__dDS_mf']
        mc = d[f'{key}__dDS_mcmc']
        sd_paths = d[f'{key}__dDS_mcmc_sd']  # not the top row's sd
        t = np.arange(1, len(mc) + 1)

        ax.fill_between(t, mc - sd_paths, mc + sd_paths, color=C_MCMC, alpha=0.22,
                        lw=0, zorder=1,
                        label='sampler, $\\pm 1$ s.d. across paths')
        mf_sd_key = f'{key}__dDS_mf_sd'
        if mf_sd_key in d.files:
            mf_sd = d[mf_sd_key]  # from mean_field.d_direct_subspace_sd
            ax.fill_between(t, mf_full - mf_sd, mf_full + mf_sd, color=C_MF,
                            alpha=0.13, lw=0, zorder=1,
                            label='mean field, $\\pm 1$ s.d. across paths')
        ax.plot(t, mc, '-o', color=C_MCMC, ms=5, lw=2, label='sampler (MCMC)',
                zorder=3)
        ax.plot(t, mf_full, '-s', color=C_MF, ms=5, lw=2, label='mean field',
                zorder=4)

        ax.set_xlabel('step $t$ along the path', fontsize=FS_LABEL)
        if col == 0:
            ax.set_ylabel('$d_{\\mathrm{DS}}$, sites off the\ndirect subspace',
                          fontsize=FS_LABEL)
            pass
        ax.tick_params(labelsize=FS_TICK)
        top = max((mc + sd_paths).max(), mf_full.max())
        ax.set_ylim(0, top * (1.30 if col == 0 else 1.08))
        ax.grid(alpha=0.25)

    handles, labels, seen = [], [], set()  # both rows draw the same two series
    for ax in (axes[0, 0], axes[1, 0]):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, fontsize=FS_LEGEND - 3, loc='lower center',
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
