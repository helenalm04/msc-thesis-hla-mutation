# Chapter 4, how the two joint scores behave along a path. The same endpoint
# pairs are sampled under 'both', which asks every node to bind each allele at
# once, and under 'either', which asks only for the better of the two. The
# profiles side by side show 'both' paying a barrier that 'either' avoids.

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
from encoding import convert_letter
from sampler import run_sampler, hamming
from path_stats import switch_stats, barrier_stats, _node_energies

_r = importlib.util.spec_from_file_location("src_rbm", os.path.join(SRC, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_r); _r.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

B51 = 'HLA-B*51:01'
TARGET = 'HLA-B*27:05'
ANCH_A = {'P', 'A'}
ANCH_B = {'R'}
N_PAIRS = 15
SEEDS = [1, 2, 3]
GRID = np.linspace(0.0, 1.0, 21)
COMMON = dict(beta=1.0, beta_init=0.1, anneal='linear', n_burn=3000, n_iter=15000, n_thin=40)
CACHE = os.path.join(RESULTS_DIR, 'score_behaviour.npz')
C_EITHER, C_BOTH = '#5C82D6', '#EA9542'


# mean energy of an allele's own measured ligands, the zero point of its centred energy scale
def ligand_offset(allele, df_ms, E_fn):
    seqs = df_ms[df_ms['MHC allele'] == allele]['Peptide'].tolist()
    return float(np.mean(E_fn(convert_number(seqs))))


# whether an encoded peptide holds at least min_distinct different residues
def _complex(enc, min_distinct=6):
    return len(set(int(x) for x in enc)) >= min_distinct


def natural_candidates(allele, df_ms, E_fn, anchors, top_frac=0.3):
    '''Real ligands with good energy, an anchor and varied residues, best energy first.'''
    seqs = df_ms[df_ms['MHC allele'] == allele]['Peptide'].tolist()
    enc = convert_number(seqs)
    order = np.argsort(E_fn(enc))
    cut = max(int(len(order) * top_frac), 60)
    codes = {AA_DICT[a] for a in anchors}
    return [enc[i] for i in order[:cut] if int(enc[i][1]) in codes and _complex(enc[i])]


def pick_pairs(candA, candB, n_pairs, min_d=5):
    '''Endpoint pairs far apart in Hamming distance, with no sequence used twice.'''
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


def _pool(v0, vT, cA, cB, score):
    '''Pool the direct paths from all seeds under one score into a single list.'''
    paths = []
    for seed in SEEDS:
        s, *_ = run_sampler(v0, vT, cA, cB, T=None, score=score, seed=seed, **COMMON)
        paths.extend(np.stack(p) for p in s)
    return paths


# mean energy of the better presented and of the worse presented allele at each step of the path
def _min_max_curves(paths, cA, cB):
    EA, EB = _node_energies(paths, cA, cB)
    return np.minimum(EA, EB).mean(0), np.maximum(EA, EB).mean(0)


# a curve of any length resampled onto the shared path progress grid
def _interp(curve):
    x = np.linspace(0.0, 1.0, len(curve))
    return np.interp(GRID, x, curve)


def compute():
    df = pd.read_csv(IEDB_MS_CSV)
    rbmA, rbmB = load_rbm(B51), load_rbm(TARGET)
    EA = lambda s, m=rbmA: rbm_energy_batch(np.atleast_2d(s), m)
    EB = lambda s, m=rbmB: rbm_energy_batch(np.atleast_2d(s), m)
    offA, offB = ligand_offset(B51, df, EA), ligand_offset(TARGET, df, EB)
    cA = lambda s: EA(s) - offA
    cB = lambda s: EB(s) - offB

    candA = natural_candidates(B51, df, EA, ANCH_A)
    candB = natural_candidates(TARGET, df, EB, ANCH_B)
    pairs = pick_pairs(candA, candB, N_PAIRS)
    print(f"{B51} -> {TARGET}: {len(pairs)} endpoint pairs  "
          f"(offsets B51={offA:.2f}, B27:05={offB:.2f}, diff {offB - offA:+.2f} nats)\n")

    min_e, max_e, min_b, max_b = [], [], [], []
    eA_e, eB_e, eA_b, eB_b = [], [], [], []  # A and B are the start and target alleles, B51 and B27:05
    crossover, cross_b, bfrac_c, bfrac_r, worst_e, worst_b, wmax_e, wmax_b = ([] for _ in range(8))
    for pi, (v0, vT, D) in enumerate(pairs):
        either = _pool(v0, vT, cA, cB, 'either')
        both = _pool(v0, vT, cA, cB, 'both')
        either_raw = _pool(v0, vT, EA, EB, 'either')  # control without centring

        me, xe = _min_max_curves(either, cA, cB)
        mb, xb = _min_max_curves(both, cA, cB)
        min_e.append(_interp(me)); max_e.append(_interp(xe))
        min_b.append(_interp(mb)); max_b.append(_interp(xb))

        EAe, EBe = _node_energies(either, cA, cB)
        EAb, EBb = _node_energies(both, cA, cB)
        eA_e.append(_interp(EAe.mean(0))); eB_e.append(_interp(EBe.mean(0)))
        eA_b.append(_interp(EAb.mean(0))); eB_b.append(_interp(EBb.mean(0)))

        sw = switch_stats(either, cA, cB)
        swb = switch_stats(both, cA, cB)
        sw_r = switch_stats(either_raw, cA, cB)
        crossover.append(sw['crossover_frac']); cross_b.append(swb['crossover_frac'])
        bfrac_c.append(sw['b_fraction']); bfrac_r.append(sw_r['b_fraction'])
        worst_e.append(barrier_stats(either, cA, cB)['worst_either_mean'])
        worst_b.append(barrier_stats(both, cA, cB)['worst_either_mean'])
        wmax_e.append(float(np.mean(xe[1:-1]))); wmax_b.append(float(np.mean(xb[1:-1])))
        print(f"  pair{pi} {convert_letter(v0)[0]}->{convert_letter(vT)[0]} D={D}: "
              f"crossover either/both={sw['crossover_frac']:.2f}/{swb['crossover_frac']:.2f} "
              f"mean-max(either/both)={wmax_e[-1]:.2f}/{wmax_b[-1]:.2f}", flush=True)

    A = lambda L: np.array(L)
    np.savez(CACHE, grid=GRID,
             min_e=A(min_e), max_e=A(max_e), min_b=A(min_b), max_b=A(max_b),
             eA_e=A(eA_e), eB_e=A(eB_e), eA_b=A(eA_b), eB_b=A(eB_b),
             crossover=A(crossover), cross_b=A(cross_b), bfrac_c=A(bfrac_c), bfrac_r=A(bfrac_r),
             worst_e=A(worst_e), worst_b=A(worst_b), wmax_e=A(wmax_e), wmax_b=A(wmax_b),
             offA=offA, offB=offB, n_pairs=len(pairs))
    _report()


def _report():
    z = np.load(CACHE, allow_pickle=True)
    n = int(z['n_pairs'])
    def ms(a):
        a = np.asarray(a); return a.mean(), a.std()
    print(f"\naveraged over {n} endpoint pairs (mean +/- between-pair sd)")
    print(f"SWITCH TIMING  crossover to final allele:  either {ms(z['crossover'])[0]:.2f} +/- {ms(z['crossover'])[1]:.2f}"
          f"   both {ms(z['cross_b'])[0]:.2f} +/- {ms(z['cross_b'])[1]:.2f}"
          f"   (either earlier by {np.mean(z['cross_b'] - z['crossover']):+.2f} +/- {np.std(z['cross_b'] - z['crossover']):.2f})")
    print(f"BEST presenter  worst min(E_A,E_B):  either {ms(z['worst_e'])[0]:+.2f} +/- {ms(z['worst_e'])[1]:.2f}"
          f"   both {ms(z['worst_b'])[0]:+.2f} +/- {ms(z['worst_b'])[1]:.2f}"
          f"   gap {np.mean(z['worst_b'] - z['worst_e']):+.2f} +/- {np.std(z['worst_b'] - z['worst_e']):.2f}")
    print(f"WORST presenter mean max(E_A,E_B):  either {ms(z['wmax_e'])[0]:+.2f} +/- {ms(z['wmax_e'])[1]:.2f}"
          f"   both {ms(z['wmax_b'])[0]:+.2f} +/- {ms(z['wmax_b'])[1]:.2f}"
          f"   gap {np.mean(z['wmax_e'] - z['wmax_b']):+.2f} +/- {np.std(z['wmax_e'] - z['wmax_b']):.2f}")
    print(f"centring effect  either B-fraction:  centred {ms(z['bfrac_c'])[0]:.2f} +/- {ms(z['bfrac_c'])[1]:.2f}"
          f"   uncentred {ms(z['bfrac_r'])[0]:.2f} +/- {ms(z['bfrac_r'])[1]:.2f}")


C_INIT, C_FIN = '#5C82D6', '#4CA64C'


def _regime_panel(ax, llA, llB, cross, title):
    '''Draw the presentation log-likelihood for both alleles and mark the crossover.'''
    def band(curves, color, label):
        m, se = curves.mean(0), curves.std(0) / np.sqrt(len(curves))
        ax.plot(GRID, m, '-', color=color, lw=2.6, label=label)
        ax.fill_between(GRID, m - se, m + se, color=color, alpha=0.16, lw=0)
    band(llA, C_INIT, 'initial allele B51:01')
    band(llB, C_FIN, 'final allele B27:05')
    cm, cse = cross.mean(), cross.std() / np.sqrt(len(cross))
    ax.axvline(cm, color='0.35', lw=1.4, ls='--')
    ax.axvspan(cm - cse, cm + cse, color='0.5', alpha=0.15, lw=0)
    ax.axhline(0, color='0.6', lw=0.9, ls=':')
    ax.set_title(title, fontsize=FS_TITLE)


def plot():
    z = np.load(CACHE, allow_pickle=True)
    n = int(z['n_pairs'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    _regime_panel(axes[0], -z['eA_e'], -z['eB_e'], z['crossover'], 'either (switch regime)')
    _regime_panel(axes[1], -z['eA_b'], -z['eB_b'], z['cross_b'], 'both (promiscuous regime)')
    axes[0].legend(fontsize=FS_LEGEND, loc='lower center', framealpha=0.92)
    axes[0].set_ylabel('presentation log-likelihood\n(nats above a typical ligand)',
                       fontsize=FS_LABEL)
    for ax in axes:
        ax.set_xlabel('path progress', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'score_behaviour.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    if '--plot' in sys.argv:
        _report()
    else:
        compute()
    plot()
