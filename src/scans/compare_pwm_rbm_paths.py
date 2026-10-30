# Chapter 3, what the RBM's correlations do to a path. The same endpoint pairs
# are sampled under the RBM and under a PWM fitted to the same ligands, then
# compared on the order the positions change in and on the gap between the two
# energies. That gap is the part a PWM cannot account for, so following it
# along a path shows where the correlations actually matter.

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

from config import IEDB_MS_CSV, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK
from entropy import convert_number, convert_letter, pwm_train
from sampler import run_sampler, hamming
from path_stats import substitution_order, _node_energies

_r = importlib.util.spec_from_file_location("src_rbm", os.path.join(SRC, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_r); _r.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

_p = importlib.util.spec_from_file_location("src_pwm", os.path.join(SRC, "pwm.py"))
_src_pwm = importlib.util.module_from_spec(_p); _p.loader.exec_module(_src_pwm)
pwm_energy_batch = _src_pwm.pwm_energy_batch

ALLELES = ['HLA-B*51:01']
N_CORRIDORS = 3  # how many pairs to sample
SEEDS = [1, 2]
BETA = 1.5
COMMON = dict(beta_init=0.15, anneal='linear', n_burn=4000, n_iter=40000, n_thin=100)
CACHE = os.path.join(FIGURES_DIR, 'pwm_rbm_paths.npz')

C_PWM = '#5C82D6'
C_RBM = '#EA9542'


# whether a peptide uses at least k distinct amino acids
def _complex(enc, k=6):
    return len(set(int(x) for x in enc)) >= k


def pick_endpoints_k(allele, df, E_fn, k=1, top_frac=0.25, E_fn2=None):
    '''Return up to k low-energy ligand pairs that are far apart, balanced and disjoint.'''
    seqs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == 9]
    enc = convert_number(seqs)
    e = np.asarray(E_fn(enc), dtype=float)
    order = np.argsort(e)
    cut = max(int(len(order) * top_frac), 40)
    good = list(order[:cut])
    if E_fn2 is not None:  # good under the other model too
        e2 = np.asarray(E_fn2(enc), dtype=float)
        keep = set(int(i) for i in np.argsort(e2)[:cut])
        good = [i for i in good if int(i) in keep]
    top = [i for i in good if _complex(enc[i])][:30]
    for min_d in (7, 5):  # relax the distance until something qualifies
        cands = []
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                ia, ib = top[a], top[b]
                d = hamming(enc[ia], enc[ib])
                if d >= min_d:
                    imbal = abs(e[ia] - e[ib])
                    if E_fn2 is not None:
                        imbal += abs(e2[ia] - e2[ib])  # imbalance under both models
                    cands.append((imbal - 0.02 * d, ia, ib, d))
        if cands:
            break
    cands.sort(key=lambda t: t[0])
    picks, used = [], set()
    for _, ia, ib, d in cands:  # pairs that share no endpoint
        if ia in used or ib in used:
            continue
        picks.append((enc[ia], enc[ib], int(d)))
        used.update((ia, ib))
        if len(picks) >= k:
            break
    return picks


def sample_pooled(v0, vT, E_fn):
    '''Sample direct paths under one model and pool them over seeds.'''
    pooled = []
    for seed in SEEDS:
        samples, _, acc, _ = run_sampler(v0, vT, E_fn, None, beta=BETA,
                                         seed=seed, **COMMON)
        pooled.extend([np.stack(p) for p in samples])
        print(f"    seed {seed}: {len(samples)} paths, accept={acc:.2f}")
    return np.stack(pooled)


def mean_flip_step(samples, positions):
    '''Mean step at which each differing position flips.'''
    steps = {p: [] for p in positions}
    for path in samples:
        for step, pos in enumerate(substitution_order(path)):
            steps[pos].append(step)
    return np.array([np.mean(steps[p]) if steps[p] else np.nan for p in positions])


def epistatic_profile(samples, E_rbm, E_pwm):
    '''Mean gap between the RBM and PWM energies along the path, in nats.'''
    zero = lambda s: np.zeros(np.atleast_2d(s).shape[0])
    d_rbm, _ = _node_energies(samples, E_rbm, zero)  # shape (n, T+1)
    d_pwm, _ = _node_energies(samples, E_pwm, zero)
    prof = (d_rbm - d_pwm).mean(axis=0)
    return prof - 0.5 * (prof[0] + prof[-1])  # shift the endpoints to zero


def compute():
    df = pd.read_csv(IEDB_MS_CSV)
    store = {}
    for allele in ALLELES:
        print(f"\n{allele}")
        rbm = load_rbm(allele)
        seqs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == 9]
        pwm = pwm_train(convert_number(seqs))
        E_rbm = lambda s, m=rbm: rbm_energy_batch(np.atleast_2d(s), m)
        E_pwm = lambda s, p=pwm: pwm_energy_batch(np.atleast_2d(s), p)

        pairs = pick_endpoints_k(allele, df, E_rbm, k=N_CORRIDORS, E_fn2=E_pwm)
        scored = []
        for v0, vT, D in pairs:
            pp = sample_pooled(v0, vT, E_pwm)
            barrier = float(epistatic_profile(pp, E_rbm, E_pwm).max())
            print(f"  corridor D={D}: PWM-route barrier={barrier:.2f} nats")
            scored.append((barrier, v0, vT, D, pp))
        scored.sort(key=lambda t: t[0])
        barrier, v0, vT, D, paths_pwm = scored[len(scored) // 2]  # the middle pair by barrier
        s0 = ''.join(convert_letter(np.asarray(v0)))
        sT = ''.join(convert_letter(np.asarray(vT)))
        print(f"  -> median corridor: {s0} -> {sT}  (D={D}, barrier={barrier:.2f}); "
              f"sampling RBM route...")
        paths_rbm = sample_pooled(v0, vT, E_rbm)

        key = allele.replace('*', '').replace(':', '')
        store[f'{key}__rbm'] = paths_rbm
        store[f'{key}__pwm'] = paths_pwm
        store[f'{key}__v0'] = np.asarray(v0)
        store[f'{key}__vT'] = np.asarray(vT)
        store[f'{key}__nh'] = np.array([int(rbm.n_h)])
    np.savez(CACHE, **store)
    print(f"\nCached paths -> {CACHE}")


def plot():
    cache = np.load(CACHE, allow_pickle=True)
    df = pd.read_csv(IEDB_MS_CSV)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    for allele in ALLELES:
        key = allele.replace('*', '').replace(':', '')
        paths_rbm = cache[f'{key}__rbm']
        paths_pwm = cache[f'{key}__pwm']
        v0, vT = cache[f'{key}__v0'], cache[f'{key}__vT']
        positions = sorted(int(p) for p in np.where(v0 != vT)[0])
        T = len(positions)
        x = np.linspace(0, 1, T + 1)

        rbm = load_rbm(allele)
        rbm.random_state = np.random.RandomState(0)
        rbm.AIS(M=100, n_betas=20000)
        logZ = float(rbm.log_Z_AIS)
        seqs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == 9]
        pwm = pwm_train(convert_number(seqs))
        E_rbm = lambda s, m=rbm: rbm_energy_batch(np.atleast_2d(s), m)
        E_pwm = lambda s, p=pwm: pwm_energy_batch(np.atleast_2d(s), p)
        zero = lambda s: np.zeros(np.atleast_2d(s).shape[0])

        Fr_pwmroute, _ = _node_energies(paths_pwm, E_rbm, zero)
        Fr_rbmroute, _ = _node_energies(paths_rbm, E_rbm, zero)
        Ep_pwmroute, _ = _node_energies(paths_pwm, E_pwm, zero)
        llR_pwmroute = -Fr_pwmroute.mean(0) - logZ  # the PWM route scored under the RBM
        llR_rbmroute = -Fr_rbmroute.mean(0) - logZ
        llP_pwmroute = -Ep_pwmroute.mean(0)

        gap = llP_pwmroute - llR_pwmroute
        end_gap = 0.5 * (gap[0] + gap[-1])
        barrier = float((gap - end_gap).max())  # peak above the endpoints
        ax.plot(x, llP_pwmroute, '-o', color=C_PWM, lw=2.2, ms=5,
                label=r'under PWM  $\log P_{\mathrm{PWM}}$')
        ax.plot(x, llR_pwmroute, '-s', color=C_RBM, lw=2.2, ms=5,
                label=r'under RBM  $\log P_{\mathrm{RBM}}$')
        ax.fill_between(x, llR_pwmroute, llP_pwmroute, color='0.5', alpha=0.15, zorder=0)
        ax.set_title(f'{allele}:  PWM route scored under both models', fontsize=FS_TITLE)
        ax.set_xlabel('path progress', fontsize=FS_LABEL)
        ax.set_ylabel(r'log-likelihood  (nats)', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND, loc='lower center', framealpha=0.9)

        route_gap = float(np.mean(llR_rbmroute[1:-1] - llR_pwmroute[1:-1]))
        rho = spearmanr(mean_flip_step(paths_pwm, positions),
                        mean_flip_step(paths_rbm, positions)).correlation
        print(f"{allele}: epistatic barrier(PWM route)={barrier:.2f} nats  "
              f"route gap(RBM-PWM)={route_gap:+.2f}  rho(order)={rho:.2f}")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'pwm_rbm_paths_loglik.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    if '--plot' not in sys.argv:
        compute()
    plot()
