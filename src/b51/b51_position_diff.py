# Chapter 6, which positions the detour actually uses. The same sweep as
# b51_either_sweep.py, but recording where on the peptide the direct and
# indirect routes differ at the barrier node.
#
# --bulk restricts the endpoints to the bulk of the repertoire, as in
# b51_either_sweep.py, and is the run the figure uses.
#
#     python -u src/b51/b51_position_diff.py --bulk

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, RESULTS_DIR, L, AA
from b51_either_sweep import (natural_candidates, pick_pairs, ligand_offset, in_bulk,
                              load_rbm, rbm_energy_batch, B51, TARGETS,
                              ANCH_A, ANCH_B)
from sampler import run_sampler
from encoding import convert_letter
from path_stats import _node_energies

SEEDS = [1, 2]
N_PAIRS = 2
K, LAM = 4, 0.5  # one cell of the b51_either_sweep grid, the corridor setting
COMMON = dict(score='either', beta=1.0, beta_init=0.1, anneal='linear',
              n_burn=3000, n_iter=18000, n_thin=40)


def barrier_nodes(samples, cE_A, cE_B):
    '''Return the residues and either score at each path barrier node.'''
    EA, EB = _node_energies(samples, cE_A, cE_B)
    either = np.minimum(EA, EB)
    k = either.argmax(axis=1)
    res = np.stack([np.asarray(samples[i][k[i]]) for i in range(len(samples))])
    peak = either[np.arange(len(samples)), k]
    return res.astype(int), peak


def run_target(target, df_ms, rbm51, off51, candA):
    rbm_t = load_rbm(target)
    E51 = lambda s, m=rbm51: rbm_energy_batch(np.atleast_2d(s), m)
    Et = lambda s, m=rbm_t: rbm_energy_batch(np.atleast_2d(s), m)
    offt = ligand_offset(target, df_ms, Et)
    cE51 = lambda s, o=off51: E51(s) - o
    cEt = lambda s, o=offt: Et(s) - o
    candB = natural_candidates(target, df_ms, Et, ANCH_B)
    if '--bulk' in sys.argv:
        candB = in_bulk(candB, target, df_ms, rbm_t)
    pairs = pick_pairs(candA, candB, N_PAIRS)

    n_d = np.zeros((L, 2))  # total barrier nodes, direct then indirect
    off = np.zeros((L, 2))
    off_res = [Counter() for _ in range(L)]
    best = {'direct': (np.inf, None, None), 'indirect': (np.inf, None, None)}

    for a, b, D in pairs:
        ds_set = [{int(a[i]), int(b[i])} for i in range(L)]
        for seed in SEEDS:
            d_s, *_ = run_sampler(a, b, cE51, cEt, T=None, seed=seed, **COMMON)
            i_s, *_ = run_sampler(a, b, cE51, cEt, T=D + K, Lambda=LAM, seed=seed, **COMMON)
            for tag, (col, samples) in {'direct': (0, d_s), 'indirect': (1, i_s)}.items():
                res, peak = barrier_nodes(samples, cE51, cEt)
                for i in range(L):
                    n_d[i, col] += len(res)
                    is_off = np.array([r not in ds_set[i] for r in res[:, i]])
                    off[i, col] += is_off.sum()
                    if col == 1:
                        for r in res[is_off, i]:
                            off_res[i][AA[int(r)]] += 1
                j = int(peak.argmin())  # path with the lowest barrier
                if peak[j] < best[tag][0]:
                    best[tag] = (float(peak[j]), res[j].copy(), (a, b))

    frac = off / np.maximum(n_d, 1)
    print(f"\n{target}  (barrier-node position diff)")
    print(f"  {'pos':>4}{'direct off%':>13}{'indirect off%':>15}{'indirect detour residues':>28}")
    for i in range(L):
        detour = ", ".join(f"{aa}:{c}" for aa, c in off_res[i].most_common(3)) or "-"
        print(f"  P{i + 1:<3}{100 * frac[i, 0]:>12.0f}{100 * frac[i, 1]:>15.0f}   {detour}")
    print(f"  mean off-direct positions at barrier: "
          f"direct={frac[:, 0].sum():.2f}  indirect={frac[:, 1].sum():.2f}")

    for tag in ('direct', 'indirect'):
        peak, res, (a, b) = best[tag]
        seq = convert_letter(res.astype(np.int16))[0]
        marks = ''.join('*' if int(res[i]) not in {int(a[i]), int(b[i])} else ''
                        for i in range(L))
        print(f"  best {tag:<8} barrier peptide: {seq}  (either={peak:.2f})")
        print(f"  {'':<33}{marks}   (* = off direct space)")
    return {'target': target,
            'off_indirect_mean': float(frac[:, 1].sum()),
            'off_by_pos': frac[:, 1].tolist()}


def main():
    df_ms = pd.read_csv(IEDB_MS_CSV)
    rbm51 = load_rbm(B51)
    E51 = lambda s, m=rbm51: rbm_energy_batch(np.atleast_2d(s), m)
    off51 = ligand_offset(B51, df_ms, E51)
    candA = natural_candidates(B51, df_ms, E51, ANCH_A)
    if '--bulk' in sys.argv:
        candA = in_bulk(candA, B51, df_ms, rbm51)
    rows = [run_target(t, df_ms, rbm51, off51, candA) for t in TARGETS]
    print("\nSUMMARY")
    for r in rows:
        print(f"{r['target']}: mean off-direct positions at barrier (indirect) = "
              f"{r['off_indirect_mean']:.2f}")

    recs = [{'target': r['target'], 'pos': i + 1, 'off_frac': f}
            for r in rows for i, f in enumerate(r['off_by_pos'])]
    out = os.path.join(RESULTS_DIR, 'b51_position_diff.csv')
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
