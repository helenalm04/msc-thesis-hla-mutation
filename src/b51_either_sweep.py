# Chapter 6, the B*51:01 sweep. Two targets share the same anchor conflict
# with B*51:01 but come out differently, B*27:05 a corridor and B*27:07 a
# valley. Direct is sampled once, indirect over a grid of (slack, Lambda).
#
# --bulk restricts the endpoints to peptides inside the bulk of the allele's
# repertoire, rather than its lowest-energy ligands, which are outliers. That
# is the run behind the corridor figure, and it writes its own _bulk table.

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, RESULTS_DIR, AA, AA_DICT
# entropy imports PGM's rbm under that bare name, which is how the saved
# models unpickle, so this import is needed even where nothing here calls it
import entropy

# PGM ships a module called rbm too and wins on sys.path, so load this one
# from its path rather than by name
_spec = importlib.util.spec_from_file_location("src_rbm", os.path.join(HERE, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

from sampler import run_sampler, hamming
from encoding import convert_number, convert_letter
from path_stats import _node_energies

B51 = 'HLA-B*51:01'
TARGETS = ['HLA-B*27:05', 'HLA-B*27:07']
# the anchor residues each side accepts at position 2, which is where the two
# motifs disagree and so what any path between them has to get across
ANCH_A = {'P', 'A'}
ANCH_B = {'R'}
SEEDS = [1, 2, 3]
N_PAIRS = 2
INDIRECT_CELLS = [(4, 0.25), (4, 0.5), (4, 1.0), (2, 0.5), (6, 0.5)]  # each cell is (k, Lambda) with T = D + k
N_ITER = 18000
COMMON = dict(score='either', beta=1.0, beta_init=0.1, anneal='linear',
              n_burn=3000, n_iter=N_ITER, n_thin=40)


# mean model energy over an allele's own measured ligands
def ligand_offset(allele, df_ms, E_fn):
    seqs = df_ms[df_ms['MHC allele'] == allele]['Peptide'].tolist()
    return float(np.mean(E_fn(convert_number(seqs))))


# true when a peptide carries at least min_distinct different amino acids
def _complex(enc, min_distinct=6):
    return len(set(int(x) for x in enc)) >= min_distinct


def natural_candidates(allele, df_ms, E_fn, anchors, top_frac=0.25):
    '''Return an allele's lowest-energy ligands with one of the given letters at position 2.'''
    seqs = df_ms[df_ms['MHC allele'] == allele]['Peptide'].tolist()
    enc = convert_number(seqs)
    order = np.argsort(E_fn(enc))
    cut = max(int(len(order) * top_frac), 40)
    codes = {AA_DICT[a] for a in anchors}
    return [enc[i] for i in order[:cut] if int(enc[i][1]) in codes and _complex(enc[i])]


# Drop endpoints that are atypical for the allele. Distance is measured in the
# RBM's hidden space, standardised per unit, so the cut is on how far a peptide
# sits from the centre of the real repertoire rather than on its energy.
def in_bulk(cands, allele, df_ms, rbm, pct=90):
    '''Keep the candidates inside the given percentile of the repertoire.'''
    seqs = convert_number([x for x in df_ms[df_ms['MHC allele'] == allele]['Peptide']
                           if len(x) == 9])
    R = rbm.input_hiddens(np.asarray(seqs).astype(np.int16))
    mu, sd = R.mean(0), R.std(0) + 1e-9
    cut = np.percentile(np.linalg.norm((R - mu) / sd, axis=1), pct)
    C = rbm.input_hiddens(np.stack(cands).astype(np.int16))
    d = np.linalg.norm((C - mu) / sd, axis=1)
    kept = [c for c, ok in zip(cands, d <= cut) if ok]
    print(f"  bulk filter {allele}: kept {len(kept)}/{len(cands)} (cut {cut:.2f})", flush=True)
    return kept


def pick_pairs(candA, candB, n_pairs, min_d=5):
    '''Endpoint pairs with good energy, Hamming at least min_d, and distinct ends.'''
    scored = []
    for i, a in enumerate(candA[:12]):
        for j, b in enumerate(candB[:12]):
            d = hamming(a, b)
            if d >= min_d:
                scored.append((i + j, d, a, b))
    scored.sort(key=lambda x: x[0])  # lowest rank sum first
    pairs, usedA, usedB = [], set(), set()
    for _, d, a, b in scored:
        ka, kb = a.tobytes(), b.tobytes()
        if ka in usedA or kb in usedB:
            continue
        pairs.append((a, b, d)); usedA.add(ka); usedB.add(kb)
        if len(pairs) >= n_pairs:
            break
    return pairs


def either_stats(samples, cE_A, cE_B):
    '''Worst either energy, its baseline, and the P2 proline fraction at the crossover.'''
    EA, EB = _node_energies(samples, cE_A, cE_B)
    either = np.minimum(EA, EB)
    base = max(either[:, 0].mean(), either[:, -1].mean())
    worst = either.max(axis=1)
    p2 = []
    for i, p in enumerate(samples):
        idx = np.where(EB[i] < EA[i])[0]
        if len(idx):
            p2.append(AA[int(np.asarray(p[idx[0]])[1])])
    pro = (sum(1 for x in p2 if x == 'P') / len(p2)) if p2 else float('nan')
    return {'worst_mean': float(worst.mean()),
            'base': float(base),
            'pro_frac': float(pro)}


def main():
    bulk = '--bulk' in sys.argv
    df_ms = pd.read_csv(IEDB_MS_CSV)
    rbm51 = load_rbm(B51)
    E51 = lambda s, m=rbm51: rbm_energy_batch(np.atleast_2d(s), m)
    off51 = ligand_offset(B51, df_ms, E51)
    candA = natural_candidates(B51, df_ms, E51, ANCH_A)
    if bulk:
        candA = in_bulk(candA, B51, df_ms, rbm51)

    all_rows = []
    for target in TARGETS:
        rbm_t = load_rbm(target)
        Et = lambda s, m=rbm_t: rbm_energy_batch(np.atleast_2d(s), m)
        offt = ligand_offset(target, df_ms, Et)
        cE51 = lambda s, o=off51: E51(s) - o
        cEt = lambda s, o=offt: Et(s) - o
        candB = natural_candidates(target, df_ms, Et, ANCH_B)
        if bulk:
            candB = in_bulk(candB, target, df_ms, rbm_t)
        pairs = pick_pairs(candA, candB, N_PAIRS)
        print(f"\n##### {B51} -> {target}: {len(pairs)} endpoint pairs #####", flush=True)
        for a, b, D in pairs:
            print(f"  v0={convert_letter(a)[0]} vT={convert_letter(b)[0]} D={D}", flush=True)

        direct, indirect = {}, {}
        for pi, (a, b, D) in enumerate(pairs):
            for seed in SEEDS:
                d_s, *_ = run_sampler(a, b, cE51, cEt, T=None, seed=seed, **COMMON)
                direct[(pi, seed)] = either_stats(d_s, cE51, cEt)
                for (k, lam) in INDIRECT_CELLS:
                    i_s, *_ = run_sampler(a, b, cE51, cEt, T=D + k, Lambda=lam,
                                          seed=seed, **COMMON)
                    indirect[(pi, seed, k, lam)] = either_stats(i_s, cE51, cEt)
                print(f"  [{target[-5:]} pair{pi} seed{seed}] done", flush=True)

        reps = [(pi, s) for pi in range(len(pairs)) for s in SEEDS]
        d_worst = np.array([direct[r]['worst_mean'] for r in reps])
        d_pro = np.nanmean([direct[r]['pro_frac'] for r in reps])
        print(f"\n  target={target}  base~{np.mean([direct[r]['base'] for r in reps]):.2f}"
              f"  DIRECT worst-either={d_worst.mean():.2f}+/-{d_worst.std():.2f}"
              f"  P2=Pro@crossover={d_pro:.2f}", flush=True)
        print(f"  {'T':>4}{'Lambda':>8}{'indirect worst':>18}{'GAP (dir-ind)':>16}"
              f"{'Pro@cx ind':>12}", flush=True)
        for (k, lam) in INDIRECT_CELLS:
            i_worst = np.array([indirect[(pi, s, k, lam)]['worst_mean'] for pi, s in reps])
            gaps = d_worst - i_worst
            i_pro = np.nanmean([indirect[(pi, s, k, lam)]['pro_frac'] for pi, s in reps])
            print(f"  {'D+' + str(k):>4}{lam:>8}{i_worst.mean():>10.2f}+/-{i_worst.std():<5.2f}"
                  f"{gaps.mean():>+9.2f}+/-{gaps.std():<5.2f}{i_pro:>12.2f}", flush=True)
            all_rows.append({'target': target, 'k': k, 'lambda': lam,
                             'direct_worst': d_worst.mean(),
                             'indirect_worst': i_worst.mean(), 'indirect_worst_std': i_worst.std(),
                             'gap': gaps.mean(), 'gap_std': gaps.std(),
                             'direct_pro': d_pro, 'indirect_pro': i_pro})

    out = os.path.join(RESULTS_DIR,
                       'b51_either_sweep_bulk.csv' if bulk else 'b51_either_sweep.csv')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSaved {out}", flush=True)


if __name__ == '__main__':
    main()
