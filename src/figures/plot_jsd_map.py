# Chapter 3, the pairwise JSD as an MDS scatter and as a supertype-ordered
# heatmap. Both show the landscapes grouping by supertype, not by locus.

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

from config import ENTROPY_CSV, JSD_CSV, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK

LOCUS_C = {'A': '#5C82D6', 'B': '#EA9542', 'C': '#4CA64C'}


# the alleles, their pairwise JSD matrix, and their locus and supertype labels
def load():
    E = pd.read_csv(ENTROPY_CSV)
    J = pd.read_csv(JSD_CSV)
    alleles = list(E['HLA allele'])
    idx = {a: i for i, a in enumerate(alleles)}
    # the table holds each pair once, so fill both triangles of the matrix
    n = len(alleles)
    M = np.zeros((n, n))
    for _, r in J.iterrows():
        i, j = idx[r['HLA allele1']], idx[r['HLA allele2']]
        # the finite size correction can push a small JSD below zero
        v = max(float(r['JSD_corr_rbm']), 0.0)
        M[i, j] = M[j, i] = v
    # the locus is the letter after the dash, so HLA-B*51:01 gives B
    loc = np.array([a.split('-')[1][0] for a in alleles])
    st = np.array(E['Supertype_new'].values, dtype=object)
    return np.array(alleles, dtype=object), M, loc, st


def cmds(D, k=2):
    '''Return k coordinates for a distance matrix by classical multidimensional scaling.'''
    # double centre the squared distances to get the Gram matrix of the points
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    # its leading eigenvectors, scaled by the square root of the eigenvalues,
    # are the coordinates; clip because a non-Euclidean D gives negative ones
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    return V[:, :k] * np.sqrt(np.clip(w[:k], 0, None))


def _save(fig, name):
    out = os.path.join(FIGURES_DIR, name)
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def _st_colours(st):
    sts = sorted(s for s in set(st) if s != 'Unclassified')
    cmap = plt.cm.tab20(np.linspace(0, 1, len(sts)))
    col = {s: tuple(cmap[i][:3]) for i, s in enumerate(sts)}
    col['Unclassified'] = (0.7, 0.7, 0.7)
    return col


# the same scatter twice, coloured by locus and then by supertype
def fig_mds(M, loc, st):
    # the square root of the divergence is what satisfies the triangle
    # inequality, so MDS gets that rather than the JSD itself
    coords = cmds(np.sqrt(M))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)

    ax = axes[0]
    for l, c in LOCUS_C.items():
        m = loc == l
        ax.scatter(coords[m, 0], coords[m, 1], s=48, color=c, edgecolors='0.3',
                   linewidths=0.4, alpha=0.85, label=f'HLA-{l}')
    ax.legend(title='locus', fontsize=FS_LEGEND)
    ax.set_title('coloured by locus  (intermingled)', fontsize=FS_TITLE)

    ax = axes[1]
    stcol = _st_colours(st)
    for s in sorted(set(st)):
        m = st == s
        ax.scatter(coords[m, 0], coords[m, 1], s=48, color=stcol[s],
                   edgecolors='0.3', linewidths=0.4, alpha=0.85)
        if s != 'Unclassified' and m.sum() >= 2:
            ax.annotate(s, (coords[m, 0].mean(), coords[m, 1].mean()),
                        fontsize=FS_TICK, fontweight='bold', ha='center', va='center')
    ax.set_title('coloured by supertype  (clusters)', fontsize=FS_TITLE)

    for ax in axes:
        ax.set_xlabel('MDS axis 1', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
    axes[0].set_ylabel('MDS axis 2', fontsize=FS_LABEL)
    fig.suptitle('Allele landscapes cluster by supertype, not locus  '
                 '(MDS of pairwise JSD)', fontsize=FS_TITLE)
    plt.tight_layout()
    _save(fig, 'jsd_mds.pdf')


# the full matrix, with the supertype blocks on the diagonal
def fig_heatmap(alleles, M, loc, st):
    # grouping the rows by supertype is what makes the blocks visible
    order = sorted(range(len(alleles)), key=lambda i: (st[i], loc[i], alleles[i]))
    o = np.array(order)
    Mo = M[np.ix_(o, o)]
    sto = st[o]
    n = len(o)
    stcol = _st_colours(st)

    fig = plt.figure(figsize=(10.5, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 14],
                          width_ratios=[24, 0.5], hspace=0.04, wspace=0.03)
    ax_st = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    ax_st.imshow(np.array([[stcol[s] for s in sto]]), aspect='auto')
    ax_st.set_xticks([]); ax_st.set_yticks([])
    ax_st.set_ylabel('supertype', rotation=0, ha='right', va='center', fontsize=FS_TICK)

    # the limits are narrowed to the range the data actually covers, otherwise
    # the blocks wash out
    im = ax.imshow(Mo, aspect='auto', cmap='viridis_r', vmin=0.30, vmax=0.68)
    # rule off where one supertype ends and the next begins
    bnd = [i for i in range(1, n) if sto[i] != sto[i - 1]]
    for b in bnd:
        ax.axhline(b - 0.5, color='white', lw=0.5)
        ax.axvline(b - 0.5, color='white', lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('68 alleles, ordered by supertype', fontsize=FS_LABEL)
    fig.colorbar(im, cax=cax).set_label('pairwise JSD (RBM, corrected)', fontsize=FS_LABEL)
    cax.tick_params(labelsize=FS_TICK)
    _save(fig, 'jsd_heatmap_supertype.pdf')


def main():
    alleles, M, loc, st = load()
    fig_mds(M, loc, st)
    fig_heatmap(alleles, M, loc, st)


if __name__ == '__main__':
    main()
