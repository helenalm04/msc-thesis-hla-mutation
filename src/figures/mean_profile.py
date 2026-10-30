# Chapter 6, the arrays behind the mean profile figure. Each system is sampled
# twice, direct and with slack to detour, and put on a shared grid so systems
# of different length can be averaged. The averaging is the point: one pair is
# too noisy to show a detour. Two modes, for the two panels.

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (IEDB_MS_CSV, ENTROPY_CSV, JSD_CSV,
                    FIGURES_DIR, FS_LABEL, FS_LEGEND, FS_TICK)
from scan_pairs import (valid_alleles, select_pairs_spread,
                        _model_key, _ligand_offset,
                        load_rbm, rbm_energy_batch)
from b51_either_sweep import (natural_candidates, B51,
                              ANCH_A, ANCH_B)
from sampler import (run_sampler, hamming,
                     choose_endpoints_transition)
from path_stats import _node_energies

B27 = 'HLA-B*27:05'
GRID = np.linspace(0.0, 1.0, 41)  # shared path grid
SLACK = 6
LAM = 0.35
DIRECTGREEN = '#2ca02c'
GLOBALRED = '#d62728'

SCHED_FULL = dict(score='either', beta=1.0, beta_init=0.1, anneal='linear',
                  n_burn=1000, n_iter=30000, n_thin=60)  # 30000/60 gives 500 paths
SCHED_QUICK = dict(score='either', beta=1.0, beta_init=0.1, anneal='linear',
                   n_burn=200, n_iter=3000, n_thin=60)


def _centred_energy_fns(rbm_A, rbm_B, off_A, off_B):
    '''Energy functions for each allele, centred on the mean over real ligands.'''
    cE_A = lambda s, m=rbm_A, o=off_A: rbm_energy_batch(np.atleast_2d(s), m) - o
    cE_B = lambda s, m=rbm_B, o=off_B: rbm_energy_batch(np.atleast_2d(s), m) - o
    return cE_A, cE_B


def _ensemble_profile(samples, cE_A, cE_B):
    '''Mean of min(E_A, E_B) at each node of an ensemble, on the shared grid.'''
    EA, EB = _node_energies(samples, cE_A, cE_B)  # shape (n, T+1)
    either = np.minimum(EA, EB)
    per_node = either.mean(axis=0)
    x = np.linspace(0.0, 1.0, len(per_node))
    return np.interp(GRID, x, per_node)


def _unit_profiles(v0, vT, cE_A, cE_B, sched, seed):
    '''Direct and global profiles for one endpoint pair.'''
    D = hamming(v0, vT)
    sd, _, _, _ = run_sampler(v0, vT, cE_A, cE_B, T=None, seed=seed, **sched)
    sg, _, _, _ = run_sampler(v0, vT, cE_A, cE_B, T=D + SLACK, Lambda=LAM,
                              seed=seed, **sched)
    return _ensemble_profile(sd, cE_A, cE_B), _ensemble_profile(sg, cE_A, cE_B)


def units_pairs(df_ms, df_E, df_jsd, n_pairs, top_k=40):
    '''One unit for each allele pair spread across the JSD range.'''
    valid = valid_alleles(df_ms, df_E)
    pairs = select_pairs_spread(df_jsd, valid, n_pairs=n_pairs)
    cache = {}
    get = lambda a: cache.setdefault(a, load_rbm(a)) if a not in cache else cache[a]
    for _, r in pairs.iterrows():
        A, B = r['HLA allele1'], r['HLA allele2']
        try:
            rbm_A, rbm_B = get(A), get(B)
            EA = lambda s, m=rbm_A: rbm_energy_batch(np.atleast_2d(s), m)
            EB = lambda s, m=rbm_B: rbm_energy_batch(np.atleast_2d(s), m)
            v0, vT, _ = choose_endpoints_transition(A, B, df_ms, EA, EB, top_k=top_k)
            cE_A, cE_B = _centred_energy_fns(
                rbm_A, rbm_B, _ligand_offset(A, df_ms, EA), _ligand_offset(B, df_ms, EB))
            yield f"{_model_key(A)}->{_model_key(B)}", v0, vT, cE_A, cE_B
        except Exception as e:
            print(f"[skip pair] {A} -> {B}: {type(e).__name__}: {e}")


def units_endpoints(df_ms, df_E, df_jsd, n_pairs, min_d=5):
    '''One unit for each endpoint peptide pair in B51:01 to B27:05.'''
    rbm_A, rbm_B = load_rbm(B51), load_rbm(B27)
    EA = lambda s, m=rbm_A: rbm_energy_batch(np.atleast_2d(s), m)
    EB = lambda s, m=rbm_B: rbm_energy_batch(np.atleast_2d(s), m)
    candA = natural_candidates(B51, df_ms, EA, ANCH_A)
    candB = natural_candidates(B27, df_ms, EB, ANCH_B)
    cE_A, cE_B = _centred_energy_fns(
        rbm_A, rbm_B, _ligand_offset(B51, df_ms, EA), _ligand_offset(B27, df_ms, EB))
    made, i, j = 0, 0, 0
    usedA, usedB = set(), set()
    while made < n_pairs and i < len(candA):  # next distinct pair from each ranked list
        a = candA[i]
        found = False
        for jj in range(j, len(candB)):
            b = candB[jj]
            if a.tobytes() in usedA or b.tobytes() in usedB:
                continue
            if hamming(a, b) >= min_d:
                usedA.add(a.tobytes()); usedB.add(b.tobytes())
                yield f"ep{made}", a, b, cE_A, cE_B
                made += 1; found = True; break
        i += 1
    if made == 0:
        raise RuntimeError("no B51->B27:05 endpoint pairs met the Hamming cut")


def collect(units, sched, seed):
    '''Run each unit and stack its direct and global profiles.'''
    direct, globl, labels = [], [], []
    for k, (label, v0, vT, cE_A, cE_B) in enumerate(units):
        try:
            dp, gp = _unit_profiles(v0, vT, cE_A, cE_B, sched, seed + k)
            direct.append(dp); globl.append(gp); labels.append(label)
            print(f"[ok] {label}: direct peak {dp.max():+.2f}  global peak {gp.max():+.2f}")
        except Exception as e:
            print(f"[skip] {label}: {type(e).__name__}: {e}")
    if not direct:
        raise RuntimeError("no units produced a profile")
    return np.array(direct), np.array(globl), labels


# mean and standard error across systems, which is what the figure shades
def _band(profiles):
    '''Mean and standard error of the mean across the rows.'''
    mean = profiles.mean(axis=0)
    n = len(profiles)
    se = profiles.std(axis=0, ddof=1) / np.sqrt(n) if n >= 2 else np.zeros_like(mean)
    return mean, se


def plot(direct, globl, out, n_units):
    '''Draw the mean direct and global profiles and save the pdf, png and npz.'''
    dmean, dse = _band(direct)
    gmean, gse = _band(globl)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for mean, se, col, lab in ((dmean, dse, DIRECTGREEN, 'direct  ($T=D$)'),
                               (gmean, gse, GLOBALRED, f'global  ($T=D+{SLACK}$)')):
        ax.plot(GRID, mean, color=col, lw=2.6, zorder=3)
        ax.fill_between(GRID, mean - se, mean + se, color=col, alpha=0.22, lw=0, zorder=2)

    ax.set_xlabel('position along the path', fontsize=FS_LABEL)
    ax.set_ylabel(r'$\langle\, \min(E_A, E_B)\, \rangle$', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    handles = [Line2D([], [], color=DIRECTGREEN, lw=3, label='direct  ($T=D$)'),
               Line2D([], [], color=GLOBALRED, lw=3, label=f'global  ($T=D+{SLACK}$)')]
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    np.savez(out + '.npz', grid=GRID, direct=direct, globl=globl, slack=SLACK, lam=LAM)
    plt.close(fig)
    print(f"saved {out}.pdf / .png / .npz  ({n_units} units)")


def run_mode(mode, sched, n_pairs, seed):
    '''Build the units for one mode, collect the profiles and plot them.'''
    df_ms = pd.read_csv(IEDB_MS_CSV)
    df_E = pd.read_csv(ENTROPY_CSV, index_col=0)
    df_jsd = pd.read_csv(JSD_CSV, index_col=0)
    if mode == 'pairs':
        units = units_pairs(df_ms, df_E, df_jsd, n_pairs)
    else:
        units = units_endpoints(df_ms, df_E, df_jsd, n_pairs)
    out = _out_path(mode)
    direct, globl, labels = collect(units, sched, seed)
    plot(direct, globl, out, len(labels))


# output path for a mode's figure and data files, stem only with no extension
def _out_path(mode):
    stem = 'poster_meanprofile_pairs' if mode == 'pairs' \
        else 'poster_meanprofile_endpoints_b5101_b2705'
    return os.path.join(FIGURES_DIR, stem)


def replot(mode):
    '''Redraw the figure from the saved .npz.'''
    out = _out_path(mode)
    d = np.load(out + '.npz')
    plot(d['direct'], d['globl'], out, len(d['direct']))


def main():
    p = argparse.ArgumentParser(description="Averaged direct-vs-global presentation profiles.")
    p.add_argument('--mode', choices=['pairs', 'endpoints', 'both'], default='both')
    p.add_argument('--n-pairs', type=int, default=None,
                   help="units to average (default 40 for pairs, 12 for endpoints)")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true',
                   help="short chains + few units, to sanity-check the pipeline")
    p.add_argument('--replot', action='store_true',
                   help="redraw from the saved .npz with current styling (no sampling)")
    args = p.parse_args()

    sched = SCHED_QUICK if args.quick else SCHED_FULL
    modes = ['pairs', 'endpoints'] if args.mode == 'both' else [args.mode]
    defaults = {'pairs': 40, 'endpoints': 12}
    if args.replot:
        for mode in modes:
            replot(mode)
        return
    for mode in modes:
        n = args.n_pairs or (3 if args.quick else defaults[mode])
        print(f"\n=== mode={mode}  n_units={n}  quick={args.quick} ===")
        run_mode(mode, sched, n, args.seed)


if __name__ == '__main__':
    main()
