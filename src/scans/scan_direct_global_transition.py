# Chapter 6, the sweep over what a mutation costs, to find where each route
# stops being available. Three systems: a pair whose anchors conflict, a
# same-supertype pair, and a single allele as a control with no second motif
# to cross to. Run twice, since the anneal could be shaping the result:
#
#     python -u src/scans/scan_direct_global_transition.py [--fixed-beta]

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, RESULTS_DIR, L
from b51_either_sweep import (natural_candidates, pick_pairs,
                              ligand_offset, load_rbm, rbm_energy_batch, B51)
from sampler import run_sampler
from path_stats import _node_energies

OUT = os.path.join(RESULTS_DIR, 'direct_global_transition.csv')

LAMBDAS = [0.35, 0.7, 1.5, 3.0, 6.0, 7.0, 8.5, 10.0, 12.0, 24.0]
SEEDS = [1, 2, 3]
K = 6  # extra steps above D
COMMON = dict(beta=1.0, beta_init=0.1, anneal='linear',
              n_burn=45000, n_iter=12000, n_thin=120)
FIXED = dict(COMMON, beta_init=None)  # no annealing ramp

SYSTEMS = [  # label, allele A, allele B, P2 anchors A, P2 anchors B
    ('anchor-incompatible', B51, 'HLA-B*27:05', {'P', 'A'}, {'R'}),
    ('same-supertype', 'HLA-A*02:01', 'HLA-A*02:04', {'L', 'M'}, {'L', 'M'}),
    ('single-allele', B51, None, {'P', 'A'}, {'P', 'A'}),
]


# for every node of a path, the number of positions carrying a residue found at neither endpoint
def d_direct_space(path, v0, vT):
    endset = [(int(a), int(b)) for a, b in zip(v0, vT)]
    return np.array([sum(int(node[i]) not in endset[i] for i in range(L))
                     for node in path])


def energies(system, df):
    '''Return the energy functions and candidate ligands for both alleles.'''
    _, aA, aB, anchA, anchB = system
    rbmA = load_rbm(aA)
    EA = lambda s, m=rbmA: rbm_energy_batch(np.atleast_2d(s), m)
    offA = ligand_offset(aA, df, EA)
    cA = lambda s: EA(s) - offA
    candA = natural_candidates(aA, df, EA, anchA)
    if aB is None:  # both endpoints come from A
        return cA, None, candA, candA
    rbmB = load_rbm(aB)
    EB = lambda s, m=rbmB: rbm_energy_batch(np.atleast_2d(s), m)
    offB = ligand_offset(aB, df, EB)
    cB = lambda s: EB(s) - offB
    return cA, cB, candA, natural_candidates(aB, df, EB, anchB)


# the value given to a flag=value command line argument, or the default if it is absent
def _opt(flag, default=None):
    for a in sys.argv[1:]:
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return default


def main():
    fixed_beta = '--fixed-beta' in sys.argv
    common = FIXED if fixed_beta else COMMON
    out = OUT.replace('.csv', '_fixedbeta.csv') if fixed_beta else OUT

    # overrides from the command line
    lambdas = [float(x) for x in _opt('--lambdas', '').split(',') if x] or LAMBDAS
    only = [s for s in _opt('--systems', '').split(',') if s]
    if _opt('--n-burn'):
        common = dict(common, n_burn=int(_opt('--n-burn')))
    append = '--append' in sys.argv  # merge new rows into the existing CSV
    systems = [s for s in SYSTEMS if not only or s[0] in only]

    df = pd.read_csv(IEDB_MS_CSV)
    rows = []
    for system in systems:
        label, aA, aB, _, _ = system
        cA, cB, candA, candB = energies(system, df)
        pairs = pick_pairs(candA, candB, 1, min_d=6)
        if not pairs:
            print(f"{label}: no endpoint pair", flush=True)
            continue
        v0, vT, D = pairs[0]
        score = 'either' if cB is not None else 'both'
        print(f"\n{label}  {aA} -> {aB or aA}  D={D}  T={D + K}", flush=True)
        print(f"{'Lambda':>7} {'%direct':>8} {'<d_DS>':>7} {'peak':>6} "
              f"{'bar_dir':>8} {'bar_glob':>9}", flush=True)
        for lam in lambdas:
            dds_mean, dds_peak, bars = [], [], []
            for seed in SEEDS:
                samples, *_ = run_sampler(v0, vT, cA, cB, seed=seed, T=D + K,
                                          Lambda=lam, score=score, **common)
                EA, EB = _node_energies(samples, cA, cB if cB is not None else cA)
                worst = (np.minimum(EA, EB) if cB is not None else EA).max(axis=1)
                for p, w in zip(samples, worst):
                    d = d_direct_space(np.stack(p), v0, vT)
                    dds_mean.append(float(d[1:-1].mean()))
                    dds_peak.append(int(d.max()))
                    bars.append(float(w))
            dds_mean = np.array(dds_mean)
            dds_peak = np.array(dds_peak)
            bars = np.array(bars)
            direct = dds_peak == 0
            f_dir = 100.0 * direct.mean()
            b_dir = float(bars[direct].mean()) if direct.any() else np.nan
            b_glob = float(bars[~direct].mean()) if (~direct).any() else np.nan
            rows.append(dict(system=label, allele_A=aA, allele_B=aB or aA,
                             D=D, T=D + K, Lambda=lam, n_paths=len(bars),
                             frac_direct=f_dir / 100.0,
                             dds_mean=float(dds_mean.mean()),
                             dds_peak=float(dds_peak.mean()),
                             barrier_direct=b_dir, barrier_global=b_glob))
            print(f"{lam:>7.2f} {f_dir:>7.0f}% {dds_mean.mean():>7.2f} "
                  f"{dds_peak.mean():>6.2f} {b_dir:>8.2f} {b_glob:>9.2f}", flush=True)

    new = pd.DataFrame(rows)
    if append and os.path.exists(out):
        old = pd.read_csv(out)
        new = (pd.concat([old, new], ignore_index=True)
                 .drop_duplicates(subset=['system', 'Lambda'], keep='last')
                 .sort_values(['system', 'Lambda']))
    new.to_csv(out, index=False)
    print(f"\nSaved {out}", flush=True)


if __name__ == '__main__':
    main()
