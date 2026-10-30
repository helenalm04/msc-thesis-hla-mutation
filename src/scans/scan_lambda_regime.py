# Chapter 4, the regime parameter on one allele pair in detail. Lambda is
# swept from negative, where only one allele is asked for, to positive, where
# both are, resampling at each value. A path counts as balanced when the two
# energies stay within TAU of each other, which finds the handover without
# having to pick a crossover point by hand.

import importlib.util
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

from config import IEDB_MS_CSV, RESULTS_DIR, FIGURES_DIR, AA_DICT, FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK
from entropy import convert_number
from sampler import run_sampler, hamming
from path_stats import switch_stats, _node_energies

_r = importlib.util.spec_from_file_location("src_rbm", os.path.join(SRC, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_r); _r.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

B51, TARGET = 'HLA-B*51:01', 'HLA-B*27:05'
ANCH_A, ANCH_B = {'P', 'A'}, {'R'}
N_PAIRS, SEEDS = 15, [1, 2, 3]
LAMS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.5, 1.0]
PANEL_LAMS = [-0.5, 0.0, 1.0]  # the ones drawn as panels
TAU = 2.0  # alleles count as balanced below this gap in nats
GRID = np.linspace(0.0, 1.0, 21)
COMMON = dict(beta=1.0, beta_init=0.1, anneal='linear', n_burn=3000, n_iter=15000, n_thin=40)
CACHE = os.path.join(RESULTS_DIR, 'lambda_regime.npz')
C_INIT, C_FIN = '#5C82D6', '#4CA64C'


def ligand_offset(allele, df, E_fn):
    '''Mean model energy of the peptides this allele presents.'''
    seqs = df[df['MHC allele'] == allele]['Peptide'].tolist()
    return float(np.mean(E_fn(convert_number(seqs))))


def _complex(enc, k=6):
    '''True when the peptide has at least k distinct residues.'''
    return len(set(int(x) for x in enc)) >= k


def natural_candidates(allele, df, E_fn, anchors, top_frac=0.3):
    '''Top scoring ligands of the allele with an anchor residue at P2.'''
    seqs = df[df['MHC allele'] == allele]['Peptide'].tolist()
    enc = convert_number(seqs)
    order = np.argsort(E_fn(enc))
    cut = max(int(len(order) * top_frac), 60)  # floor for alleles with few ligands
    codes = {AA_DICT[a] for a in anchors}
    return [enc[i] for i in order[:cut] if int(enc[i][1]) in codes and _complex(enc[i])]


def pick_pairs(candA, candB, n_pairs, min_d=5):
    '''Pair the two candidate lists using each peptide once.'''
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


# the curve resampled onto the common path-progress grid from 0 to 1
def _interp(curve):
    return np.interp(GRID, np.linspace(0.0, 1.0, len(curve)), curve)


def _pool(v0, vT, cA, cB, lam):
    '''Pool the paths for one endpoint pair across all seeds.'''
    paths = []
    for seed in SEEDS:
        s, *_ = run_sampler(v0, vT, cA, cB, T=None, score='lambda', lam=lam, seed=seed, **COMMON)
        paths.extend(np.stack(p) for p in s)
    return paths


def compute():
    df = pd.read_csv(IEDB_MS_CSV)
    rbmA, rbmB = load_rbm(B51), load_rbm(TARGET)
    EA = lambda s, m=rbmA: rbm_energy_batch(np.atleast_2d(s), m)
    EB = lambda s, m=rbmB: rbm_energy_batch(np.atleast_2d(s), m)
    offA, offB = ligand_offset(B51, df, EA), ligand_offset(TARGET, df, EB)
    cA = lambda s: EA(s) - offA
    cB = lambda s: EB(s) - offB
    pairs = pick_pairs(natural_candidates(B51, df, EA, ANCH_A),
                       natural_candidates(TARGET, df, EB, ANCH_B), N_PAIRS)
    print(f"{B51} -> {TARGET}: {len(pairs)} pairs, lambda sweep {LAMS}\n")

    out = {'lams': np.array(LAMS), 'grid': GRID, 'n_pairs': len(pairs)}
    store = {k: [] for k in ('ortho', 'cross', 'balanced', 'corridor')}
    curves = {lam: {'eA': [], 'eB': []} for lam in LAMS}
    for lam in LAMS:
        o, c, b, m = [], [], [], []
        for (v0, vT, D) in pairs:
            paths = _pool(v0, vT, cA, cB, lam)
            EAn, EBn = _node_energies(paths, cA, cB)  # (n_paths, T+1)
            mx, mn = np.maximum(EAn, EBn), np.minimum(EAn, EBn)
            o.append(float(mx[:, 1:-1].mean()))
            m.append(float(mn[:, 1:-1].mean()))
            b.append(float((np.abs(EAn - EBn)[:, 1:-1] < TAU).mean()))
            c.append(switch_stats(paths, cA, cB)['crossover_frac'])
            curves[lam]['eA'].append(_interp(EAn.mean(0)))
            curves[lam]['eB'].append(_interp(EBn.mean(0)))
        store['ortho'].append(o); store['cross'].append(c)
        store['balanced'].append(b); store['corridor'].append(m)
        print(f"  lam={lam:+.2f}: ortho(max)={np.mean(o):+.2f}  crossover={np.mean(c):.2f}  "
              f"balanced_frac={np.mean(b):.2f}  corridor(min)={np.mean(m):+.2f}", flush=True)

    for k, v in store.items():
        out[k] = np.array(v)  # (n_lam, n_pairs)
    for lam in LAMS:
        out[f'eA_{lam}'] = np.array(curves[lam]['eA'])
        out[f'eB_{lam}'] = np.array(curves[lam]['eB'])
    np.savez(CACHE, **out)
    _report()


def _report():
    z = np.load(CACHE, allow_pickle=True)
    lams = z['lams']
    print(f"\nE_lambda regime sweep (mean +/- between-pair sd, {int(z['n_pairs'])} pairs)")
    print(f"{'lambda':>7} {'ortho(max)':>18} {'crossover':>14} {'balanced_frac':>16} {'corridor(min)':>16}")
    for i, lam in enumerate(lams):
        def ms(key):
            a = z[key][i]; return a.mean(), a.std()
        print(f"{lam:>+7.2f} {ms('ortho')[0]:>+9.2f}+/-{ms('ortho')[1]:<5.2f} "
              f"{ms('cross')[0]:>7.2f}+/-{ms('cross')[1]:<4.2f} "
              f"{ms('balanced')[0]:>9.2f}+/-{ms('balanced')[1]:<4.2f} "
              f"{ms('corridor')[0]:>+9.2f}+/-{ms('corridor')[1]:<4.2f}")


def plot():
    z = np.load(CACHE, allow_pickle=True)
    lams = list(z['lams'])
    titles = {-0.5: r'$\lambda=-0.5$ (switch)', 0.0: r'$\lambda=0$ (either)',
              1.0: r'$\lambda=+1$ (both)'}
    fig, axes = plt.subplots(1, len(PANEL_LAMS) + 1, figsize=(5 * (len(PANEL_LAMS) + 1), 4.6))
    for ax, lam in zip(axes[:-1], PANEL_LAMS):
        eA, eB = -z[f'eA_{lam}'], -z[f'eB_{lam}']
        for curves, color, label in ((eA, C_INIT, 'initial B51:01'), (eB, C_FIN, 'final B27:05')):
            m, se = curves.mean(0), curves.std(0) / np.sqrt(len(curves))
            ax.plot(GRID, m, '-', color=color, lw=2.4, label=label)
            ax.fill_between(GRID, m - se, m + se, color=color, alpha=0.16, lw=0)
        cm = z['cross'][lams.index(lam)].mean()
        ax.axvline(cm, color='0.35', lw=1.3, ls='--')
        ax.axhline(0, color='0.6', lw=0.8, ls=':')
        ax.set_title(titles[lam], fontsize=FS_TITLE)
        ax.set_xlabel('path progress', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_ylim(-9, 7)
    axes[0].set_ylabel('presentation log-likelihood\n(nats above a typical ligand)', fontsize=FS_LABEL)
    axes[0].legend(fontsize=FS_LEGEND, loc='lower center', framealpha=0.92)

    ax = axes[-1]
    L = np.array(lams)
    se = lambda a: a.std(1) / np.sqrt(a.shape[1])
    C_BETTER, C_WORSE = '#4a3aa7', '#e34948'  # same colours as fig:lambda_vs_jsd
    better, worse = -z['corridor'].mean(1), -z['ortho'].mean(1)
    ax.fill_between(L, worse, better, color='0.55', alpha=0.16, lw=0, zorder=0)
    ax.annotate('separation\n$|E_A - E_B|$', xy=(0.12, 0.4), xycoords='data',
                ha='center', va='center', fontsize=FS_LEGEND - 4, color='0.3')
    ax.errorbar(L, better, yerr=se(z['corridor']), fmt='-o', color=C_BETTER,
                lw=2.2, ms=6, capsize=3, zorder=3, label=r'better allele  $-\min(E_A,E_B)$')
    ax.errorbar(L, worse, yerr=se(z['ortho']), fmt='-s', color=C_WORSE,
                lw=2.2, ms=6, capsize=3, zorder=3, label=r'worse allele  $-\max(E_A,E_B)$')
    ax.axhline(0, color='0.6', lw=0.8, ls=':')
    ax.set_xlabel(r'$\lambda$', fontsize=FS_LABEL)
    ax.set_ylabel('presentation log-likelihood\n(nats above a typical ligand)',
                  fontsize=FS_LABEL)
    ax.set_title('what $\\lambda$ trades', fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(-9, 7)  # same limits as the panels
    ax.legend(fontsize=FS_LEGEND - 4, loc='upper left', framealpha=0.92)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'lambda_regime.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    if '--plot' in sys.argv:  # replot from the cache without resampling
        _report()
    else:
        compute()
    plot()
