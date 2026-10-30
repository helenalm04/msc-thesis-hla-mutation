# Chapter 3, where the divergence between two alleles actually sits. The JSD
# is taken apart column by column, the anchors against the rest, and each part
# set against how hard the transitions turned out to be. The correlations are
# partial throughout: both parts rise with the total divergence, so a plain
# correlation cannot say which carries the difficulty.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (IEDB_MS_CSV, JSD_CSV, ENTROPY_CSV, MODELS_DIR, RESULTS_DIR,
                    FIGURES_DIR, ANCHOR_POSITIONS, NON_ANCHOR_POSITIONS, L,
                    FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK)
from entropy import convert_number, pwm_train

DIFF_CSV = os.path.join(RESULTS_DIR, 'path_stats_either.csv')  # 40 sampled pairs
PAIRS_OUT = os.path.join(RESULTS_DIR, 'anchor_jsd_pairs.csv')
CACHE = os.path.join(FIGURES_DIR, 'anchor_jsd.npz')
FIGOUT = os.path.join(FIGURES_DIR, 'anchor_jsd.pdf')
FIGOUT_COLUMNS = os.path.join(FIGURES_DIR, 'anchor_jsd_columns.pdf')
FIGOUT_DIFFICULTY = os.path.join(FIGURES_DIR, 'anchor_jsd_difficulty.pdf')

C_SAME = '#4CA64C'
C_DIFF = '#B07AA1'


# allele name in the compact form that appears in the model filenames
def _model_key(allele):
    return allele.replace('HLA-', '').replace('*', '').replace(':', '')


def _valid_alleles(df_ms, df_E):
    '''Alleles with a pretrained RBM, IEDB ligands and an entropy row.'''
    model_files = os.listdir(MODELS_DIR)
    has_model = {a for a in df_E['HLA allele']
                 if any(_model_key(a) in f for f in model_files)}
    return has_model & set(df_ms['MHC allele'].unique()) & set(df_E['HLA allele'])


def build_pwms(df_ms, alleles):
    '''Map each allele to its (9,20) empirical PWM.'''
    pwms = {}
    for a in alleles:
        seqs = [s for s in df_ms[df_ms['MHC allele'] == a]['Peptide'] if len(s) == L]
        if len(seqs) >= 20:
            pwms[a] = pwm_train(convert_number(seqs))
    return pwms


def _pos_jsd(pA, pB):
    '''Jensen-Shannon divergence at each position between two PWMs, in nats.'''
    m = 0.5 * (pA + pB)
    def _kl(p, q):
        return np.sum(np.where(p > 0, p * np.log(p / q), 0.0), axis=1)
    return 0.5 * _kl(pA, m) + 0.5 * _kl(pB, m)  # (9,)


def decompose(pA, pB):
    j = _pos_jsd(pA, pB)
    return (float(j[ANCHOR_POSITIONS].mean()),
            float(j[NON_ANCHOR_POSITIONS].mean()),
            float(j.mean()))


def _partial_spearman(x, y, Z):
    '''Partial Spearman of x and y controlling for the columns of Z.'''
    x, y = rankdata(x), rankdata(y)
    Z = np.atleast_2d(np.asarray(Z, float))
    if Z.shape[0] != len(x):
        Z = Z.T
    Zr = np.column_stack([rankdata(Z[:, k]) for k in range(Z.shape[1])])
    A = np.column_stack([Zr, np.ones(len(x))])
    ex = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ey = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return spearmanr(ex, ey).correlation


def compute():
    df_ms = pd.read_csv(IEDB_MS_CSV)
    df_jsd = pd.read_csv(JSD_CSV, index_col=0)
    df_E = pd.read_csv(ENTROPY_CSV, index_col=0)

    st = dict(zip(df_E['HLA allele'], df_E['Supertype_new'].astype(str)))
    Nof = dict(zip(df_E['HLA allele'], df_E['N']))
    valid = _valid_alleles(df_ms, df_E)
    pwms = build_pwms(df_ms, valid)
    print(f"{len(pwms)} alleles with a PWM")

    rows = []
    for _, r in df_jsd.iterrows():
        a, b = r['HLA allele1'], r['HLA allele2']
        if a not in pwms or b not in pwms or a == b:
            continue
        aj, nj, pj = decompose(pwms[a], pwms[b])
        rows.append(dict(
            allele_A=a, allele_B=b,
            anchor_jsd=aj, nonanchor_jsd=nj, pos_jsd=pj,
            jsd_table_pwm=float(r.get('JSD_corr_pwm', np.nan)),
            jsd_table_rbm=float(r.get('JSD_corr_rbm', np.nan)),
            same_supertype=int(st.get(a, 'A') == st.get(b, 'B')),
            same_locus=int(a.split('-')[1][0] == b.split('-')[1][0]),
            min_N=float(min(Nof.get(a, np.nan), Nof.get(b, np.nan))),
        ))
    pop = pd.DataFrame(rows)
    print(f"population: {len(pop)} pairs "
          f"({pop.same_supertype.sum()} same-supertype)")

    diff = pd.read_csv(DIFF_CSV)
    key = lambda a, b: frozenset((a, b))
    pop_by_key = {key(r.allele_A, r.allele_B): r for r in pop.itertuples()}
    drows = []
    for _, d in diff.iterrows():
        k = key(d['allele_A'], d['allele_B'])
        if k not in pop_by_key:
            continue
        p = pop_by_key[k]
        drows.append(dict(
            allele_A=d['allele_A'], allele_B=d['allele_B'],
            difficulty=float(d['worst_either_centered']),
            barrier_both=float(d.get('barrier_both_centered', np.nan)),
            T=float(d['T']),
            anchor_jsd=p.anchor_jsd, nonanchor_jsd=p.nonanchor_jsd,
            pos_jsd=p.pos_jsd, jsd_table_rbm=p.jsd_table_rbm,
            same_supertype=p.same_supertype, min_N=p.min_N,
        ))
    dm = pd.DataFrame(drows)
    pop.to_csv(PAIRS_OUT, index=False)
    print(f"difficulty: matched {len(dm)}/{len(diff)} sampled pairs")

    np.savez(CACHE,
             pop_anchor=pop.anchor_jsd.values, pop_non=pop.nonanchor_jsd.values,
             pop_pos=pop.pos_jsd.values, pop_table=pop.jsd_table_rbm.values,
             pop_same=pop.same_supertype.values, pop_locus=pop.same_locus.values,
             pop_minN=pop.min_N.values,
             d_anchor=dm.anchor_jsd.values, d_non=dm.nonanchor_jsd.values,
             d_pos=dm.pos_jsd.values, d_table=dm.jsd_table_rbm.values,
             d_diff=dm.difficulty.values, d_T=dm['T'].values,
             d_same=dm.same_supertype.values, d_minN=dm.min_N.values)
    print(f"Cached -> {CACHE}")


def _report(z):
    '''Print the correlation table and return its numbers.'''
    d_diff = z['d_diff']
    same = z['d_same'].astype(bool)
    Z_full = np.column_stack([z['d_minN'], z['d_T'], z['d_same']])
    out = {}
    print("\nLink B: JSD flavour -> transition difficulty (n ="
          f"{len(d_diff)} sampled pairs, {same.sum()} same-supertype) ===")
    print(f"{'predictor':<16}{'rho':>7}{'partial|N,T':>13}"
          f"{'partial|N,T,ST':>16}{'within-ST rho':>15}")
    for name, key in [('anchor-JSD', 'd_anchor'), ('non-anchor-JSD', 'd_non'),
                      ('pos-JSD(all)', 'd_pos'), ('table JSD', 'd_table')]:
        x = z[key]
        rho = spearmanr(x, d_diff).correlation
        pr_nt = _partial_spearman(x, d_diff, np.column_stack([z['d_minN'], z['d_T']]))
        pr_full = _partial_spearman(x, d_diff, Z_full)
        wrho = (spearmanr(x[same], d_diff[same]).correlation
                if same.sum() >= 5 else np.nan)
        out[name] = (rho, pr_nt, pr_full, wrho)
        print(f"{name:<16}{rho:>7.2f}{pr_nt:>13.2f}{pr_full:>16.2f}{wrho:>15.2f}")

    psame = z['pop_same'].astype(bool)
    for name, key in [('anchor-JSD', 'pop_anchor'), ('non-anchor-JSD', 'pop_non')]:
        a = z[key]
        delta = a[psame].mean() - a[~psame].mean()
        rN = spearmanr(a, z['pop_minN']).correlation
        print(f"[population] {name}: same-ST {a[psame].mean():.3f} vs "
              f"cross-ST {a[~psame].mean():.3f} (delta {delta:+.3f}); "
              f"rho(.,min N) = {rN:+.2f}")
    return out


def _tidy(ax, axis='both'):
    '''Soften the grid and axes.'''
    ax.grid(True, axis=axis, lw=0.4, color='#DDDDDD')
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)


def _panel_difficulty(ax, z, same, tag='(a)'):
    '''Plot transition difficulty against anchor-column divergence.'''
    for m, c, lab in [(same, C_SAME, 'same supertype'),
                      (~same, C_DIFF, 'cross supertype')]:
        ax.scatter(z['d_anchor'][m], z['d_diff'][m], s=48, color=c, alpha=0.9,
                   edgecolor='white', linewidth=0.8, label=lab, zorder=3)
    ax.set_xlabel('anchor-column JSD  (nats)', fontsize=FS_LABEL)
    ax.set_ylabel('transition difficulty  (nats)', fontsize=FS_LABEL)
    if tag:
        ax.set_title(tag, loc='left', fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND - 2, loc='upper left', framealpha=0.9)
    _tidy(ax)


def _panel_columns(ax, z, psame, tag='(b)'):
    '''Plot divergence at each position by column set and supertype.'''
    for meas, y_same, y_cross in [('pop_anchor', 3.4, 2.8), ('pop_non', 1.7, 1.1)]:
        for m, c, ypos in [(psame, C_SAME, y_same), (~psame, C_DIFF, y_cross)]:
            ax.boxplot(z[meas][m], positions=[ypos], vert=False, widths=0.52,
                       whis=(5, 95), showfliers=False, patch_artist=True,
                       showmeans=True, zorder=3,
                       medianprops=dict(color='white', lw=1.6),
                       boxprops=dict(facecolor=c, edgecolor=c, lw=1.2),
                       whiskerprops=dict(color=c, lw=1.4),
                       capprops=dict(color=c, lw=1.4),
                       meanprops=dict(marker='D', markersize=5,
                                      markerfacecolor='white',
                                      markeredgecolor=c, markeredgewidth=1.4))
    ax.set_yticks([3.1, 1.4])
    ax.set_yticklabels(['anchor\n(P2, P9)', 'non-anchor\n(P1, P3\u2013P8)'],
                       fontsize=FS_LABEL)
    ax.set_ylim(0.65, 3.85)
    ax.set_xlim(left=-0.01)
    ax.set_xlabel('per-position JSD  (nats)', fontsize=FS_LABEL)
    # boxplot artists take no label argument
    ax.legend(handles=[Patch(facecolor=C_SAME, edgecolor=C_SAME, label='same supertype'),
                       Patch(facecolor=C_DIFF, edgecolor=C_DIFF, label='cross supertype')],
              fontsize=FS_LEGEND - 2, loc='lower right', framealpha=0.9)
    if tag:
        ax.set_title(tag, loc='left', fontsize=FS_TITLE)
    _tidy(ax, axis='x')


def _save(fig, path):
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=200)
    fig.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def plot():
    z = np.load(CACHE, allow_pickle=True)
    _report(z)
    same = z['d_same'].astype(bool)
    psame = z['pop_same'].astype(bool)

    # combined two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    _panel_difficulty(axes[0], z, same)
    _panel_columns(axes[1], z, psame)
    _save(fig, FIGOUT)

    fig, ax = plt.subplots(figsize=(5.9, 4.6))
    _panel_columns(ax, z, psame, tag=None)
    _save(fig, FIGOUT_COLUMNS)

    fig, ax = plt.subplots(figsize=(5.9, 4.6))
    _panel_difficulty(ax, z, same, tag=None)
    _save(fig, FIGOUT_DIFFICULTY)


if __name__ == '__main__':
    if '--plot' not in sys.argv:
        compute()
    plot()
