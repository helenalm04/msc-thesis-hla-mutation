# Chapter 5, where along a path one allele gives way to the other. Two systems
# for contrast: a pair whose anchors conflict, where the handover has to happen
# somewhere definite, and a same-supertype pair, where the two energies stay
# close the whole way and there is no real crossover to find.

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

from config import IEDB_MS_CSV, RESULTS_DIR
from b51_either_sweep import (natural_candidates, pick_pairs,
                              ligand_offset, load_rbm, rbm_energy_batch, B51)
from sampler import run_sampler
from path_stats import _node_energies

OUT = os.path.join(RESULTS_DIR, 'energy_crossover.csv')

LAM = 0.35
K = 6
SEEDS = [1, 2, 3]
COMMON = dict(beta=1.0, beta_init=0.1, anneal='linear',
              n_burn=3000, n_iter=12000, n_thin=120)

SYSTEMS = [
    ('anchor-incompatible', B51, 'HLA-B*27:05', {'P', 'A'}, {'R'}),
    ('same-supertype', 'HLA-A*02:01', 'HLA-A*02:04', {'L', 'M'}, {'L', 'M'}),
]


def energies(system, df):
    '''Return the centred energy functions and the candidates for both alleles.'''
    _, aA, aB, anchA, anchB = system
    rbmA, rbmB = load_rbm(aA), load_rbm(aB)
    EA = lambda s, m=rbmA: rbm_energy_batch(np.atleast_2d(s), m)
    EB = lambda s, m=rbmB: rbm_energy_batch(np.atleast_2d(s), m)
    offA, offB = ligand_offset(aA, df, EA), ligand_offset(aB, df, EB)
    cA = lambda s: EA(s) - offA
    cB = lambda s: EB(s) - offB
    return (cA, cB,
            natural_candidates(aA, df, EA, anchA),
            natural_candidates(aB, df, EB, anchB))


def main():
    df = pd.read_csv(IEDB_MS_CSV)
    rows = []
    for system in SYSTEMS:
        label, aA, aB, _, _ = system
        cA, cB, candA, candB = energies(system, df)
        pairs = pick_pairs(candA, candB, 1, min_d=6)
        if not pairs:
            print(f"{label}: no endpoint pair", flush=True)
            continue
        v0, vT, D = pairs[0]
        T = D + K
        print(f"\n{label}  {aA} -> {aB}  D={D}  T={T}", flush=True)

        EA_all, EB_all = [], []
        for seed in SEEDS:
            samples, *_ = run_sampler(v0, vT, cA, cB, seed=seed, T=T,
                                      Lambda=LAM, score='either', **COMMON)
            ea, eb = _node_energies(samples, cA, cB)
            EA_all.append(ea)
            EB_all.append(eb)
        EA_all = np.concatenate(EA_all)  # shape (n_paths, T+1)
        EB_all = np.concatenate(EB_all)
        n = EA_all.shape[0]

        mA, mB = EA_all.mean(axis=0), EB_all.mean(axis=0)
        sA = EA_all.std(axis=0, ddof=1) / np.sqrt(n)
        sB = EB_all.std(axis=0, ddof=1) / np.sqrt(n)
        mmin = np.minimum(mA, mB)
        t_bar = int(np.argmax(mmin[1:-1])) + 1  # the ends are left out

        sign = np.sign(mA - mB)
        cross = np.where(np.diff(sign) != 0)[0]
        t_cross = int(cross[0]) + 1 if len(cross) else -1

        for t in range(T + 1):
            rows.append(dict(system=label, allele_A=aA, allele_B=aB, D=D, T=T,
                             Lambda=LAM, n_paths=n, node=t,
                             frac=t / T, E_A=float(mA[t]), E_B=float(mB[t]),
                             se_A=float(sA[t]), se_B=float(sB[t]),
                             E_min=float(mmin[t]),
                             barrier_node=t_bar, cross_node=t_cross))

        print(f"  n_paths={n}  barrier at node {t_bar} of {T}, "
              f"min-energy {mmin[t_bar]:+.2f} nats", flush=True)
        if t_cross > 0:
            print(f"  energies cross at node {t_cross} of {T} "
                  f"(barrier node {t_bar})", flush=True)
        else:
            print("  energies never cross: one allele presents better "
                  "throughout", flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved {OUT}", flush=True)


if __name__ == '__main__':
    main()
