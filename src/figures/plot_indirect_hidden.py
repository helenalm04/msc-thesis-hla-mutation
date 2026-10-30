# Chapter 6, what the detour is doing in the models' own terms. The hidden
# unit that best separates the two repertoires is picked out, the paths are
# projected onto it, and its weights drawn underneath as a logo, which is what
# gives the detour a reading. Reads the cache from plot_indirect_pca.py.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logomaker

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (IEDB_MS_CSV, FIGURES_DIR, L, AA, Q_PEP,
                    FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK)
from b51_either_sweep import load_rbm, rbm_energy_batch
from encoding import convert_number
from plot_indirect_pca import SYSTEMS, syskey
N_SHOW = 30
N_CLOUD = 450
C_DIRECT = '#5C82D6'
C_INDIRECT = '#EA9542'
C_SRC = '#7FAF6B'
C_TGT = '#B07AA1'
CACHE = os.path.join(FIGURES_DIR, '_indirect_pca_paths.npz')


# the allele's measured nonamer ligands in the table, encoded as amino acid indices
def repertoire(df, allele):
    return convert_number([s for s in df[df['MHC allele'] == allele]['Peptide']
                           if len(s) == 9])


# the input each peptide sends to each of the model's hidden units
def hid(rbm, seqs):
    return rbm.input_hiddens(np.atleast_2d(np.asarray(seqs)).astype(np.int16))


# the unit whose input separates the two repertoires best, measured as a
# standardised difference of means so units of different scale compare fairly
def pick_unit(rbm, own, other):
    '''Return the hidden unit that best separates the two alleles' peptides.'''
    Io, Ix = hid(rbm, own), hid(rbm, other)
    d = (Io.mean(0) - Ix.mean(0)) / np.sqrt(0.5 * (Io.var(0) + Ix.var(0)) + 1e-12)
    mu = int(np.argmax(np.abs(d)))
    return mu, float(np.sign(d[mu])), float(abs(d[mu]))


def logo_panel(ax, rbm, mu, sign, title):
    '''Draw the unit's weights as a signed sequence logo.'''
    w = sign * rbm.weights[mu]
    dfw = pd.DataFrame(w, columns=list(AA[:Q_PEP]))
    dfw.index = list(range(L))
    logomaker.Logo(dfw, ax=ax, color_scheme='chemistry', flip_below=False)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f'P{i + 1}' for i in range(L)], fontsize=FS_TICK - 3)
    ax.tick_params(labelsize=FS_TICK - 3)
    ax.set_ylabel('weight', fontsize=FS_TICK - 2)
    ax.set_title(title, fontsize=FS_TICK)


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(f"missing {CACHE}; run plot_indirect_pca.py first")
    z = np.load(CACHE, allow_pickle=True)
    df = pd.read_csv(IEDB_MS_CSV)

    show = SYSTEMS
    n = len(show)
    fig = plt.figure(figsize=(4.7 * n, 11.6))
    gs = fig.add_gridspec(3, n, height_ratios=[2.7, 0.85, 0.85],
                          hspace=0.55, wspace=0.34)
    rng = np.random.default_rng(0)
    tags = 'abcdefghi'
    summary, src_logos, tgt_logos = [], [], []

    for j, (src, tgt, _, _, outcome) in enumerate(show):
        key = syskey(src, tgt)
        ax = fig.add_subplot(gs[0, j])
        rbmA, rbmB = load_rbm(src), load_rbm(tgt)
        EA = lambda s, m=rbmA: rbm_energy_batch(np.atleast_2d(s), m)
        EB = lambda s, m=rbmB: rbm_energy_batch(np.atleast_2d(s), m)
        offA, offB = float(z[f'{key}__offA'][0]), float(z[f'{key}__offB'][0])
        cA = lambda s, e=EA, o=offA: e(s) - o
        cB = lambda s, e=EB, o=offB: e(s) - o
        repA, repB = repertoire(df, src), repertoire(df, tgt)

        muA, sgA, dA = pick_unit(rbmA, repA, repB)
        muB, sgB, dB = pick_unit(rbmB, repB, repA)
        proj = lambda s, a=rbmA, b=rbmB, ia=muA, ib=muB, ga=sgA, gb=sgB: \
            np.column_stack([ga * hid(a, s)[:, ia], gb * hid(b, s)[:, ib]])

        sA = repA[rng.choice(len(repA), min(N_CLOUD, len(repA)), replace=False)]
        sB = repB[rng.choice(len(repB), min(N_CLOUD, len(repB)), replace=False)]
        PA, PB = proj(sA), proj(sB)
        ax.scatter(PA[:, 0], PA[:, 1], s=8, c=C_SRC, alpha=0.55, edgecolor='none',
                   rasterized=True, zorder=1, label='source-allele peptides')
        ax.scatter(PB[:, 0], PB[:, 1], s=8, c=C_TGT, alpha=0.55, marker='^',
                   edgecolor='none', rasterized=True, zorder=1,
                   label='target-allele peptides')

        v0, vT = z[f'{key}__v0'], z[f'{key}__vT']
        rx, ry = PA[:, 0], PB[:, 1]  # each repertoire on its own axis
        row = {'system': f'{src} -> {tgt}', 'outcome': outcome,
               "d'_source": dA, "d'_target": dB}  # goes to the summary CSV only
        for cls, col, lab in [('direct', C_DIRECT, 'direct  ($T=D$)'),
                              ('indirect', C_INDIRECT, r'indirect  ($T=D+6$)')]:
            paths = z[f'{key}__{cls}']
            idx = rng.choice(len(paths), min(N_SHOW, len(paths)), replace=False)
            allx, ally, bars = [], [], []  # one barrier node from each path
            for i in range(len(paths)):
                k = int(np.minimum(cA(paths[i]), cB(paths[i])).argmax())
                p = proj(paths[i][k][None])[0]
                allx.append(p[0]); ally.append(p[1])
                bars.append(float(np.minimum(cA(paths[i]), cB(paths[i])).max()))
            for m, i in enumerate(idx):
                P = proj(paths[i])
                ax.plot(P[:, 0], P[:, 1], color=col, lw=1.0, alpha=0.32, zorder=3,
                        solid_capstyle='round', label=lab if m == 0 else None)
            row[f'{cls}_barrier'] = float(np.mean(bars))
            row[f'{cls}_z_target'] = (np.mean(ally) - ry.mean()) / ry.std()

        for v, fc in [(v0, 'white'), (vT, 'black')]:
            p = proj(np.asarray(v)[None])[0]
            ax.scatter(p[0], p[1], marker='*', s=380, facecolor=fc,
                       edgecolor='black', lw=1.4, zorder=7)
            ax.annotate(r'$\mathbf{v}_0$' if fc == 'white' else r'$\mathbf{v}_T$',
                        p, textcoords='offset points', xytext=(8, 5),
                        fontsize=FS_LABEL - 1, fontweight='bold', zorder=8)

        ax.set_xlabel(f'$I_{{{muA + 1}}}$ of {src[4:]}', fontsize=FS_LABEL - 1)
        ax.set_ylabel(f'$I_{{{muB + 1}}}$ of {tgt[4:]}', fontsize=FS_LABEL - 1)
        ax.set_title(f"({tags[j]})  {src[4:]} $\\to$ {tgt[4:]}",
                     fontsize=FS_TITLE - 1)
        ax.tick_params(labelsize=FS_TICK - 1)

        src_logos.append((src, rbmA, muA, sgA))  # one logo for each column
        tgt_logos.append((tgt, rbmB, muB, sgB))
        summary.append(row)

    for j, (allele, rbm_, mu_, sg_) in enumerate(src_logos):
        logo_panel(fig.add_subplot(gs[1, j]), rbm_, mu_, sg_,
                   f'({tags[n + j]})  {allele[4:]} unit $I_{{{mu_ + 1}}}$')
    for j, (allele, rbm_, mu_, sg_) in enumerate(tgt_logos):
        logo_panel(fig.add_subplot(gs[2, j]), rbm_, mu_, sg_,
                   f'({tags[2 * n + j]})  {allele[4:]} unit $I_{{{mu_ + 1}}}$')

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FS_LEGEND - 2, frameon=False,
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.025),
               columnspacing=2.0, handletextpad=0.5, markerscale=1.6)
    out = os.path.join(FIGURES_DIR, 'indirect_hidden_b51.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")
    s = pd.DataFrame(summary)
    s['gap'] = s['direct_barrier'] - s['indirect_barrier']
    print(s.to_string(index=False, float_format=lambda v: f'{v:+.2f}'))
    s.to_csv(os.path.join(FIGURES_DIR, 'indirect_hidden_summary.csv'), index=False)


if __name__ == '__main__':
    main()
