# Chapter 4, the figure version of tests/test_boltzmann_proof.py: the same
# enumerable cases, but tracking the total variation distance between the
# sampled and exact laws as the chain runs. It falls off as one over the root
# of the sample count, which is sampling noise and nothing worse.

import itertools
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import FIGURES_DIR, RESULTS_DIR, FS_LABEL, FS_TITLE, FS_TICK
import sampler

L, Q = sampler.L, sampler.Q_PEP
BLUE = '#5C82D6'  # data graph palette
GREEN = '#4CA64C'
MOVES = (('Gibbs', 'local', BLUE),  # label, proposal name, colour
         ('Mauri et al.', 'mauri', GREEN))
CACHE = os.path.join(RESULTS_DIR, 'boltzmann_proof.npz')


def random_pwm_energy(seed, scale=0.4):
    '''Energy E(s) = scale * sum_i M[i, s_i] for a random normal matrix M.'''
    M = scale * np.random.default_rng(seed).standard_normal((L, Q))

    def E_fn(seqs):
        seqs = np.atleast_2d(seqs).astype(np.int64)
        idx_i = np.broadcast_to(np.arange(L), seqs.shape)
        return M[idx_i, seqs].sum(axis=1)
    return E_fn


def restricted_energy(active, allowed, seed, big=40.0):
    '''Energy that is cheap only on the allowed letters at the active positions.'''
    rng = np.random.default_rng(seed)
    M = np.full((L, Q), big, dtype=float)
    M[:, 0] = 0.0
    for pos in active:
        for z in allowed:
            M[pos, z] = 0.0 if z == 0 else float(rng.uniform(0.2, 1.5))

    def E_fn(seqs):
        seqs = np.atleast_2d(seqs).astype(np.int64)
        idx_i = np.broadcast_to(np.arange(L), seqs.shape)
        return M[idx_i, seqs].sum(axis=1)
    return E_fn


# hashable identifier for a path, the raw bytes of each state in order
def path_key(path):
    return tuple(np.asarray(node).astype(np.int64).tobytes() for node in path)


def empirical(samples):
    '''Frequency of each path in the samples.'''
    counts = {}
    for path in samples:
        k = path_key(path)
        counts[k] = counts.get(k, 0) + 1
    n = len(samples)
    return {k: c / n for k, c in counts.items()}, n


# total variation distance, which is the largest disagreement in probability
# the two laws can have about any set of paths
def tvd(exact, emp):
    '''Total variation distance between the exact and sampled path laws.'''
    on = 0.5 * sum(abs(p - emp.get(k, 0.0)) for k, p in exact.items())
    off = 0.5 * max(0.0, 1.0 - sum(emp.get(k, 0.0) for k in exact))
    return on + off


def aligned(exact, emp, N):
    '''Return the exact probabilities and the floored empirical frequencies on the same keys.'''
    floor = 0.3 / N
    keys = list(exact)
    x = np.array([exact[k] for k in keys])
    y = np.array([max(emp.get(k, 0.0), floor) for k in keys])
    return x, y


def direct_exact(s0, sT, diff, E_A, E_B, beta):
    w = {}
    for perm in itertools.permutations(diff):
        path, cur = [s0.copy()], s0.copy()
        for p in perm:
            cur = cur.copy(); cur[p] = sT[p]; path.append(cur)
        e = sampler.path_energy(path, E_A, E_B, sampler.score_both)
        w[path_key(path)] = np.exp(-beta * e)
    Z = sum(w.values())
    return {k: v / Z for k, v in w.items()}


def _to_full(state, active):
    s = np.zeros(L, dtype=np.int16)
    for pos, val in zip(active, state):
        s[pos] = val
    return s


def enumerate_indirect(s0r, sTr, T, allowed):
    '''Every reduced walk of length T from s0r to sTr.'''
    n_active = len(s0r)

    def neighbours(st):
        yield st  # stay
        for i in range(n_active):
            for z in allowed:
                if z != st[i]:
                    yield st[:i] + (z,) + st[i + 1:]

    out = []

    def dfs(cur, left, acc):
        if left == 0:
            if cur == sTr:
                out.append(acc)
            return
        for nb in neighbours(cur):
            dfs(nb, left - 1, acc + [nb])

    dfs(s0r, T, [s0r])
    return out


def indirect_exact(reduced_paths, active, E_A, E_B, beta, Lambda):
    w = {}
    for rp in reduced_paths:
        path = [_to_full(st, active) for st in rp]
        fit = sampler.path_energy(path, E_A, E_B, sampler.score_both)
        elastic = sum(sampler.hamming(path[t], path[t + 1]) for t in range(len(path) - 1))
        w[path_key(path)] = np.exp(-beta * (fit + Lambda * elastic))
    Z = sum(w.values())
    return {k: v / Z for k, v in w.items()}


def mixing_curve(s0, sT, EA, EB, T, Lambda, exact, proposal, n_iter, thin, seed=0):
    samples, *_ = sampler.run_sampler(
        s0, sT, EA, EB, beta=1.0, score='both', T=T, Lambda=Lambda,
        proposal=proposal, n_burn=0, n_iter=n_iter, n_thin=thin, seed=seed)
    keys = list(exact)
    ex = np.array([exact[k] for k in keys])
    idx = {k: j for j, k in enumerate(keys)}
    counts = np.zeros(len(keys))
    off = 0
    n = len(samples)
    checks = list(np.unique(np.geomspace(5, n, 60).astype(int)))
    ci = 0
    upd, curve = [], []
    for i, path in enumerate(samples, 1):
        j = idx.get(path_key(path))
        if j is None:
            off += 1
        else:
            counts[j] += 1
        while ci < len(checks) and i >= checks[ci]:
            emp = counts / i
            t = 0.5 * np.abs(ex - emp).sum() + 0.5 * (off / i)
            upd.append(i * thin); curve.append(t)
            ci += 1
    return np.array(upd), np.array(curve)


def compute():
    beta = 1.0

    print("(a) direct system (T = D = 6)...", flush=True)
    s0 = np.zeros(L, dtype=np.int16); sT = s0.copy()
    diff = [0, 1, 2, 4, 6, 8]
    for p in diff:
        sT[p] = 5
    EAd, EBd = random_pwm_energy(11), random_pwm_energy(22)
    ex_d = direct_exact(s0, sT, diff, EAd, EBd, beta)
    samp, *_ = sampler.run_sampler(s0, sT, EAd, EBd, beta=beta, score='both',
                                   n_burn=20000, n_iter=3_000_000, n_thin=10, seed=0)
    emp_d, Nd = empirical(samp)
    xd, yd = aligned(ex_d, emp_d, Nd)
    tvd_d = tvd(ex_d, emp_d)
    print(f"    {len(ex_d)} paths, TVD={tvd_d:.4f}", flush=True)

    print("(b) indirect correctness (25 paths, both moves)...", flush=True)
    ac, alc = [3], (0, 1, 2, 3, 4)  # one active site, five letters, 25 paths
    s0c, sTc = _to_full((0,), ac), _to_full((1,), ac)
    Tc = sampler.hamming(s0c, sTc) + 2
    EAc, EBc = restricted_energy(ac, alc, 7), restricted_energy(ac, alc, 8)
    ex_c = indirect_exact(enumerate_indirect((0,), (1,), Tc, alc), ac, EAc, EBc, beta, 0.5)
    out_b = {}
    for label, prop, _ in MOVES:
        n_it, n_th = 1_200_000, 10
        s, *_ = sampler.run_sampler(s0c, sTc, EAc, EBc, beta=beta, score='both', T=Tc,
                                    Lambda=0.5, proposal=prop, n_burn=10000,
                                    n_iter=n_it, n_thin=n_th, seed=0)
        emp, N = empirical(s)
        out_b[prop] = (*aligned(ex_c, emp, N), tvd(ex_c, emp))
        print(f"    {label:<24} TVD={out_b[prop][2]:.4f}", flush=True)

    print("(c) mixing curves (66 paths)...", flush=True)
    am, alm = [2, 5], (0, 1, 2)
    s0m, sTm = _to_full((0, 0), am), _to_full((1, 1), am)
    Tm = sampler.hamming(s0m, sTm) + 2
    EAm, EBm = restricted_energy(am, alm, 7), restricted_energy(am, alm, 8)
    ex_m = indirect_exact(enumerate_indirect((0, 0), (1, 1), Tm, alm), am, EAm, EBm, beta, 0.5)
    out_c = {}
    for label, prop, _ in MOVES:
        n_it = 1_500_000
        out_c[prop] = mixing_curve(s0m, sTm, EAm, EBm, Tm, 0.5, ex_m, prop, n_it, 40)
        print(f"    {label:<24} final TVD={out_c[prop][1][-1]:.4f}", flush=True)

    cache = {'xd': xd, 'yd': yd, 'tvd_d': tvd_d, 'n_d': len(ex_d),
             'n_c': len(ex_c), 'n_mix': len(ex_m)}
    for _, prop, _ in MOVES:
        x, y, t = out_b[prop]
        u, c = out_c[prop]
        cache.update({f'x_{prop}': x, f'y_{prop}': y, f'tvd_{prop}': t,
                      f'u_{prop}': u, f'c_{prop}': c})
    np.savez(CACHE, **cache)
    print(f"cached {CACHE}", flush=True)


def plot():
    z = np.load(CACHE)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.0))
    lims = [3e-5, 1.0]

    ax[0].plot(lims, lims, color='0.35', lw=1.0, ls='--', zorder=0)
    ax[0].scatter(z['xd'], z['yd'], s=16, c=BLUE, alpha=0.65, edgecolors='none')
    ax[0].set_title('(a) direct paths, $T=D$', fontsize=FS_TITLE)

    ax[1].plot(lims, lims, color='0.35', lw=1.0, ls='--', zorder=0)
    for label, prop, colour in MOVES:
        ax[1].scatter(z[f'x_{prop}'], z[f'y_{prop}'], s=16, c=colour, alpha=0.65,
                      edgecolors='none', label=label)
    ax[1].set_title('(b) indirect paths, $T>D$', fontsize=FS_TITLE)
    ax[1].legend(fontsize=FS_TICK, loc='lower right', frameon=False)

    for a in ax[:2]:
        a.set_xscale('log'); a.set_yscale('log')
        a.set_xlim(lims); a.set_ylim(lims); a.set_aspect('equal', 'box')
        a.set_xlabel('exact Boltzmann probability', fontsize=FS_LABEL)
        a.set_ylabel('sampled frequency', fontsize=FS_LABEL)
        a.tick_params(labelsize=FS_TICK)

    # the curves overlap, so the second is dashed
    for label, prop, colour in MOVES:
        ax[2].plot(z[f'u_{prop}'], z[f'c_{prop}'], ls='--' if prop == 'mauri' else '-',
                   color=colour, lw=2.4, label=label)
    ax[2].set_xscale('log')
    ax[2].set_ylim(0, 0.60)
    ax[2].set_xlabel('single-node updates', fontsize=FS_LABEL)
    ax[2].set_ylabel('total-variation distance to exact', fontsize=FS_LABEL)
    ax[2].set_title(f"(c) mixing speed, {int(z['n_mix'])}-path system", fontsize=FS_TITLE)
    ax[2].legend(fontsize=FS_TICK, loc='center right', frameon=False)
    ax[2].tick_params(labelsize=FS_TICK)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, 'boltzmann_proof.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == '__main__':
    if '--plot' not in sys.argv:
        compute()
    plot()
