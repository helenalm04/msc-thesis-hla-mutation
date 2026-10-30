# Chapter 4, the regime parameter against how far apart the alleles are.
# Pairs are drawn per JSD bin rather than at random, so the range is covered
# evenly, and no allele is used more than twice, so one allele with many
# ligands cannot carry the result.

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (IEDB_MS_CSV, DATA_DIR, RESULTS_DIR, FIGURES_DIR,
                    FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK)
from entropy import convert_number
from sampler import run_sampler, hamming
from path_stats import _node_energies

_r = importlib.util.spec_from_file_location("src_rbm", os.path.join(SRC, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_r); _r.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

JSD_CSV = os.path.join(DATA_DIR, 'df_jsd_total.csv')
JSD_COL = 'JSD_corr_rbm'  # RBM JSD, corrected for sample size
JSD_BINS = [0.00, 0.10, 0.20, 0.35, 0.50, 0.62, 0.67, 0.70]
PER_BIN = 2  # allele pairs drawn from each JSD bin
MAX_USES = 2  # how many times one allele can appear
LAMS = [-0.5, -0.25, 0.0, 0.5, 1.0]
LAM_BOTH, LAM_EITHER, LAM_SWITCH = 1.0, 0.0, -0.5
N_PAIRS, SEEDS = 6, [1, 2]
TAU = 2.0
GRID = np.linspace(0.0, 1.0, 21)
COMMON = dict(beta=1.0, beta_init=0.1, anneal='linear', n_burn=3000, n_iter=15000, n_thin=40)
CACHE = os.path.join(RESULTS_DIR, 'lambda_vs_jsd.npz')
FIGOUT = os.path.join(FIGURES_DIR, 'lambda_vs_jsd.pdf')
C_A, C_B = '#5C82D6', '#4CA64C'
C_SPREAD = '#832f0e'
FIGSIZE = (11.0, 9.2)  # labels come out near 10 pt at 180 mm \linewidth


def available_alleles():
    '''Alleles with a pretrained RBM, named the biology way.'''
    out = set()
    for f in os.listdir(os.path.join(DATA_DIR, 'models')):
        if not f.startswith('model_HLA-'):
            continue
        tag = f.split('_')[1]  # tag looks like HLA-A0201
        out.add(f'{tag[:5]}*{tag[5:7]}:{tag[7:]}')
    return out


def select_pairs():
    df = pd.read_csv(JSD_CSV)
    have = available_alleles()
    df = df[df['HLA allele1'].isin(have) & df['HLA allele2'].isin(have)].copy()
    df['N_min'] = df[['N1', 'N2']].min(axis=1)
    picks, uses = [], {}
    for lo, hi in zip(JSD_BINS[:-1], JSD_BINS[1:]):
        sub = df[(df[JSD_COL] >= lo) & (df[JSD_COL] < hi)].sort_values('N_min', ascending=False)
        taken = 0
        for _, r in sub.iterrows():
            a, b = r['HLA allele1'], r['HLA allele2']
            if uses.get(a, 0) >= MAX_USES or uses.get(b, 0) >= MAX_USES:
                continue
            picks.append((a, b, float(r[JSD_COL])))
            uses[a] = uses.get(a, 0) + 1; uses[b] = uses.get(b, 0) + 1
            taken += 1
            if taken >= PER_BIN:
                break
    return sorted(picks, key=lambda t: t[2])


# true when a peptide carries at least k distinct residues
def _complex(enc, k=6):
    return len(set(int(x) for x in enc)) >= k


def top_binders(allele, df, E_fn, top_frac=0.3, n_min=60):
    '''The ligands this allele presents best, encoded and sorted best first.'''
    enc = convert_number(df[df['MHC allele'] == allele]['Peptide'].tolist())
    order = np.argsort(E_fn(enc))
    cut = max(int(len(order) * top_frac), n_min)
    return [enc[i] for i in order[:cut] if _complex(enc[i])]


def pick_pairs(candA, candB, n_pairs, min_d=5):
    '''Endpoint pairs near the top of both rankings, far apart and not reused.'''
    scored = []
    for i, a in enumerate(candA[:40]):
        for j, b in enumerate(candB[:40]):
            d = hamming(a, b)
            if d >= min_d:
                scored.append((i + j, d, a, b))
    scored.sort(key=lambda x: x[0])
    pairs, usedA, usedB = [], set(), set()
    for _, d, a, b in scored:
        ka, kb = a.tobytes(), b.tobytes()
        if ka in usedA or kb in usedB:
            continue
        pairs.append((a, b, d)); usedA.add(ka); usedB.add(kb)
        if len(pairs) >= n_pairs:
            break
    return pairs


# a curve of any length read off at the shared fractional positions along a path
def _interp(curve):
    return np.interp(GRID, np.linspace(0.0, 1.0, len(curve)), curve)


def _pool(v0, vT, cA, cB, lam):
    '''Direct paths at this lam, pooled over the seeds.'''
    paths = []
    for seed in SEEDS:
        s, *_ = run_sampler(v0, vT, cA, cB, T=None, score='lambda', lam=lam, seed=seed, **COMMON)
        paths.extend(np.stack(p) for p in s)
    return paths


def compute():
    df = pd.read_csv(IEDB_MS_CSV)
    sel = select_pairs()
    print(f"{len(sel)} allele pairs, JSD "
          f"{sel[0][2]:.3f} .. {sel[-1][2]:.3f}; lam sweep {LAMS}\n")

    n_ap, n_lam = len(sel), len(LAMS)
    keys = ('spread', 'corridor', 'ortho', 'balanced')
    store = {k: np.full((n_lam, n_ap, N_PAIRS), np.nan) for k in keys}
    prof = {s: np.full((n_lam, n_ap, N_PAIRS, len(GRID)), np.nan) for s in ('eA', 'eB')}
    dists = np.full((n_ap, N_PAIRS), np.nan)

    for ai, (A, B, jsd) in enumerate(sel):
        rbmA, rbmB = load_rbm(A), load_rbm(B)
        EA = lambda s, m=rbmA: rbm_energy_batch(np.atleast_2d(s), m)
        EB = lambda s, m=rbmB: rbm_energy_batch(np.atleast_2d(s), m)
        seqA = convert_number(df[df['MHC allele'] == A]['Peptide'].tolist())
        seqB = convert_number(df[df['MHC allele'] == B]['Peptide'].tolist())
        offA, offB = float(np.mean(EA(seqA))), float(np.mean(EB(seqB)))
        cA = lambda s: EA(s) - offA
        cB = lambda s: EB(s) - offB
        pairs = pick_pairs(top_binders(A, df, EA), top_binders(B, df, EB), N_PAIRS)
        dists[ai, :len(pairs)] = [d for *_, d in pairs]
        print(f"[{ai+1}/{n_ap}] {A} -> {B}  JSD={jsd:.3f}  "
              f"{len(pairs)} endpoint pairs, D={[d for *_, d in pairs]}", flush=True)

        for li, lam in enumerate(LAMS):
            for pi, (v0, vT, _) in enumerate(pairs):
                paths = _pool(v0, vT, cA, cB, lam)
                EAn, EBn = _node_energies(paths, cA, cB)  # shape (n_paths, T+1)
                mx, mn = np.maximum(EAn, EBn), np.minimum(EAn, EBn)
                inner = slice(1, -1)
                store['spread'][li, ai, pi] = float(np.abs(EAn - EBn)[:, inner].mean())
                store['corridor'][li, ai, pi] = float(mn[:, inner].mean())
                store['ortho'][li, ai, pi] = float(mx[:, inner].mean())
                store['balanced'][li, ai, pi] = float((np.abs(EAn - EBn)[:, inner] < TAU).mean())
                prof['eA'][li, ai, pi] = _interp(EAn.mean(0))
                prof['eB'][li, ai, pi] = _interp(EBn.mean(0))
            print(f"    lam={lam:+.2f}  spread={np.nanmean(store['spread'][li, ai]):.2f}  "
                  f"corridor={np.nanmean(store['corridor'][li, ai]):+.2f}  "
                  f"ortho={np.nanmean(store['ortho'][li, ai]):+.2f}  "
                  f"balanced={np.nanmean(store['balanced'][li, ai]):.2f}", flush=True)

    out = {'lams': np.array(LAMS), 'grid': GRID, 'D': dists,
           'jsd': np.array([j for *_, j in sel]),
           'alleleA': np.array([a for a, _, _ in sel]),
           'alleleB': np.array([b for _, b, _ in sel])}
    out.update(store); out.update(prof)
    np.savez(CACHE, **out)
    print(f"\nCached -> {CACHE}")


def _contrasts(z):
    '''Lam contrasts for each allele pair: (both-cost, switch-gain).'''
    lams = list(np.round(z['lams'], 6))
    ib, ie, isw = (lams.index(round(v, 6)) for v in (LAM_BOTH, LAM_EITHER, LAM_SWITCH))
    both = z['corridor'][ib] - z['corridor'][ie]  # shape (n_ap, N_PAIRS)
    switch = z['ortho'][isw] - z['ortho'][ie]
    se = lambda a: np.nanstd(a, axis=1) / np.sqrt(np.sum(~np.isnan(a), axis=1))
    return (np.nanmean(both, axis=1), se(both)), (np.nanmean(switch, axis=1), se(switch))


# panel label set just outside the top left corner of an axes
def _letter(ax, s):
    ax.annotate(s, xy=(0, 1), xycoords='axes fraction', xytext=(-52, 10),
                textcoords='offset points', fontsize=FS_TITLE, fontweight='bold',
                va='bottom', ha='left')


def _exemplars(ax_lo, ax_hi, z, jsd, lams, A, B, lo_i, hi_i):
    '''(a, b) example routes at the both and the switch settings.'''
    def band(key, li, i):
        '''Mean and standard error over the endpoint pairs, as -energy.'''
        a = -z[key][li, i]
        if a.ndim == 1:
            return a, np.zeros_like(a)
        n = np.sum(~np.isnan(a[:, 0]))
        return np.nanmean(a, axis=0), np.nanstd(a, axis=0) / max(np.sqrt(n), 1.0)

    li_both, li_sw = (lams.index(round(v, 6)) for v in (LAM_BOTH, LAM_SWITCH))
    lim = []
    for i in (lo_i, hi_i):
        for li in (li_both, li_sw):
            for key in ('eA', 'eB'):
                m, se = band(key, li, i)
                lim.extend([m - se, m + se])
    ymin, ymax = float(np.min(lim)), float(np.max(lim))
    pad = 0.06 * (ymax - ymin)

    for ax, i, name in [(ax_lo, lo_i, 'low JSD'), (ax_hi, hi_i, 'high JSD')]:
        for lam, ls in [(LAM_BOTH, '-'), (LAM_SWITCH, '--')]:
            li = lams.index(round(lam, 6))
            for key, col in (('eA', C_A), ('eB', C_B)):
                m, se = band(key, li, i)
                ax.fill_between(GRID, m - se, m + se, color=col, alpha=0.18,
                                lw=0, zorder=2)
                ax.plot(GRID, m, ls, color=col, lw=2.3, zorder=3)
        ax.axhline(0, color='0.6', lw=0.8, ls=':', zorder=2)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(f"{name}:  {str(A[i])[4:]} $\\rightarrow$ {str(B[i])[4:]} "
                     f"  $({jsd[i]:.2f})$", fontsize=FS_TITLE, pad=10)
        ax.set_xlabel('path progress', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)

    ax_lo.set_ylabel('presentation log-likelihood\n(nats above a typical ligand)',
                     fontsize=FS_LABEL)
    h = [plt.Line2D([], [], color=C_A, lw=2.3), plt.Line2D([], [], color=C_B, lw=2.3),
         plt.Line2D([], [], color='0.35', lw=2.3, ls='-'),
         plt.Line2D([], [], color='0.35', lw=2.3, ls='--')]
    ax_lo.legend(h, ['start allele', 'target allele',
                     r'$\lambda=+1$ (both)', r'$\lambda=-0.5$ (switch)'],
                 fontsize=FS_LEGEND - 3, loc='center left', ncol=2, framealpha=0.92,
                 columnspacing=1.2, handlelength=1.9)


def _spread_vs_jsd(ax, z, jsd):
    lams = np.round(z['lams'], 6)
    sp = np.nanmean(z['spread'], axis=2)
    ib, isw = (list(lams).index(round(v, 6)) for v in (LAM_BOTH, LAM_SWITCH))
    ax.vlines(jsd, sp[ib], sp[isw], color=C_SPREAD, lw=5, alpha=0.28, zorder=1)
    for k, v in enumerate(lams):
        if v not in (LAM_BOTH, LAM_SWITCH):
            ax.plot(jsd, sp[k], '_', color=C_SPREAD, ms=5.5, mew=1.3, alpha=0.75,
                    zorder=2)
    ax.plot(jsd, sp[ib], 'o', color=C_SPREAD, ms=6, zorder=3)
    ax.plot(jsd, sp[isw], 'v', color=C_SPREAD, ms=6, zorder=3)

    h = [plt.Line2D([], [], color=C_SPREAD, marker='v', lw=0, ms=6),
         plt.Line2D([], [], color=C_SPREAD, marker='_', lw=0, ms=6, mew=1.3),
         plt.Line2D([], [], color=C_SPREAD, marker='o', lw=0, ms=6)]
    ax.legend(h, [r'$\lambda = -0.5$  (switch)', r'$\lambda = -0.25,\ 0,\ +0.5$',
                  r'$\lambda = +1$  (both)'],
              fontsize=FS_LEGEND - 3, loc='upper left', frameon=False,
              handletextpad=0.3, labelspacing=0.3, borderaxespad=0.6)
    ax.set_xlabel('pairwise JSD of the two alleles', fontsize=FS_LABEL)
    ax.set_ylabel(r'route mean $|E_A - E_B|$  (nats)', fontsize=FS_LABEL)
    ax.set_title(r'separation of the two alleles, and how much of it $\lambda$ '
                 'controls', fontsize=FS_TITLE, pad=10)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(jsd.min() - 0.04, jsd.max() + 0.04)
    lev = z['spread'][isw] - z['spread'][ib]
    return spearmanr(jsd, np.nanmean(lev, axis=1))


def plot():
    z = np.load(CACHE, allow_pickle=True)
    jsd, lams = z['jsd'], list(np.round(z['lams'], 6))
    A, B = z['alleleA'], z['alleleB']
    lo_i, hi_i = int(np.argmin(jsd)), int(np.argmax(jsd))
    (both_m, both_se), (sw_m, sw_se) = _contrasts(z)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.32)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _exemplars(ax_a, ax_b, z, jsd, lams, A, B, lo_i, hi_i)
    rho, p = _spread_vs_jsd(ax_c, z, jsd)
    print(f"  {'leverage':>12}: Spearman rho = {rho:+.2f}, p = {p:.4f}")
    for nm, y in [('both-cost', both_m), ('switch-gain', sw_m)]:
        r, pv = spearmanr(jsd, y)
        print(f"  {nm:>12}: Spearman rho = {r:+.2f}, p = {pv:.3f}   (text only)")

    for axp, L in zip((ax_a, ax_b, ax_c), ['(a)', '(b)', '(c)']):
        _letter(axp, L)
    fig.savefig(FIGOUT, bbox_inches='tight', dpi=200)
    fig.savefig(FIGOUT.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    _report(z, both_m, sw_m)
    print(f"Saved: {FIGOUT}")


def _report(z, both_m, sw_m):
    jsd = z['jsd']
    print(f"\ncombined score vs pairwise JSD ({len(jsd)} allele pairs)")
    print(f"{'allele A':>12} {'allele B':>12} {'JSD':>6} {'spread(0)':>10} "
          f"{'both-cost':>10} {'switch-gain':>12} {'bal(0)':>7}")
    ie = list(np.round(z['lams'], 6)).index(round(LAM_EITHER, 6))
    for i in range(len(jsd)):
        print(f"{str(z['alleleA'][i]):>12} {str(z['alleleB'][i]):>12} {jsd[i]:>6.3f} "
              f"{np.nanmean(z['spread'][ie, i]):>10.2f} {both_m[i]:>10.2f} "
              f"{sw_m[i]:>12.2f} {np.nanmean(z['balanced'][ie, i]):>7.2f}")
    lo, hi = jsd < np.median(jsd), jsd >= np.median(jsd)
    print(f"\nlow-JSD half  (JSD <= {np.median(jsd):.2f}): both-cost {both_m[lo].mean():+.2f}, "
          f"switch-gain {sw_m[lo].mean():+.2f}")
    print(f"high-JSD half (JSD >  {np.median(jsd):.2f}): both-cost {both_m[hi].mean():+.2f}, "
          f"switch-gain {sw_m[hi].mean():+.2f}")


if __name__ == '__main__':
    if '--plot' not in sys.argv:
        compute()
    plot()
