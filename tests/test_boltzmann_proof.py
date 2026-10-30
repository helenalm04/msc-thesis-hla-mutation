# The chain samples the Boltzmann distribution over paths. Each test builds a
# case small enough to enumerate every path, works out the exact
# probabilities, then runs the sampler and compares them one by one.

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sampler

L = sampler.L


def _make_random_pwm_energy(seed):
    '''Additive energy E(s) = sum_i M[i, s_i] with M fixed by the seed.'''
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((L, sampler.Q_PEP))

    def E_fn(seqs):
        seqs = np.atleast_2d(seqs).astype(np.int64)
        idx_i = np.broadcast_to(np.arange(L), seqs.shape)
        return M[idx_i, seqs].sum(axis=1)

    return E_fn


def _path_key(path):
    '''Turn a path into a hashable key.'''
    return tuple(np.asarray(node).astype(np.int64).tobytes() for node in path)


def _setup():
    '''Endpoints three positions apart, with six direct paths.'''
    # three substitutions can be made in 3! = 6 orders, few enough to list
    s0 = np.zeros(L, dtype=np.int16)
    sT = s0.copy()
    diff = [1, 4, 7]
    for p in diff:
        sT[p] = 5
    E_A_fn = _make_random_pwm_energy(seed=11)
    E_B_fn = _make_random_pwm_energy(seed=22)
    return s0, sT, diff, E_A_fn, E_B_fn


# the exact law over direct paths: one path per order of the substitutions
def _exact_distribution(s0, sT, diff, E_A_fn, E_B_fn, beta):
    weights = {}
    for perm in itertools.permutations(diff):
        path = [s0.copy()]
        cur = s0.copy()
        for p in perm:
            cur = cur.copy()
            cur[p] = sT[p]
            path.append(cur)
        e = sampler.path_energy(path, E_A_fn, E_B_fn, sampler.score_both)
        weights[_path_key(path)] = np.exp(-beta * e)
    Z = sum(weights.values())
    return {k: v / Z for k, v in weights.items()}


def _empirical_distribution(s0, sT, E_A_fn, E_B_fn, beta, **cfg):
    '''Run the sampler and return (path frequencies, n_samples).'''
    samples, *_ = sampler.run_sampler(
        s0, sT, E_A_fn, E_B_fn, beta=beta, score='both',
        n_iter=150000, n_thin=10, seed=0, **cfg,
    )
    counts = {}
    for path in samples:
        k = _path_key(path)
        counts[k] = counts.get(k, 0) + 1
    N = len(samples)
    return {k: c / N for k, c in counts.items()}, N


# the default move on direct paths, with and without an anneal, since the
# anneal must leave the distribution it ends at untouched
def test_biased_matches_boltzmann():
    s0, sT, diff, E_A_fn, E_B_fn = _setup()
    beta = 1.0
    exact = _exact_distribution(s0, sT, diff, E_A_fn, E_B_fn, beta)

    for cfg in (dict(n_burn=5000),  # fixed beta
                dict(n_burn=5000, beta_init=0.05, anneal='linear')):
        emp, _ = _empirical_distribution(s0, sT, E_A_fn, E_B_fn, beta, **cfg)
        for k, p in exact.items():
            assert abs(emp.get(k, 0.0) - p) < 0.04, \
                f"path prob {emp.get(k, 0.0):.3f} vs exact {p:.3f}"


def _setup_indirect():
    '''Endpoints one position apart, with one free interior node.'''
    s0 = np.zeros(L, dtype=np.int16)
    sT = s0.copy()
    p = 3
    sT[p] = 5
    E_A_fn = _make_random_pwm_energy(seed=7)
    E_B_fn = _make_random_pwm_energy(seed=8)
    return s0, sT, p, E_A_fn, E_B_fn


# with one free interior node the paths are indexed by its residue, so all
# 20 can be enumerated, elastic cost included
def _exact_indirect(s0, sT, p, E_A_fn, E_B_fn, beta, Lambda):
    weights = {}
    for z in range(sampler.Q_PEP):
        v1 = s0.copy(); v1[p] = z
        path = [s0.copy(), v1, sT.copy()]
        fit = sampler.path_energy(path, E_A_fn, E_B_fn, sampler.score_both)
        elastic = sampler.hamming(s0, v1) + sampler.hamming(v1, sT)
        weights[_path_key(path)] = np.exp(-beta * (fit + Lambda * elastic))
    Z = sum(weights.values())
    return {k: v / Z for k, v in weights.items()}


def test_indirect_matches_boltzmann():
    s0, sT, p, E_A_fn, E_B_fn = _setup_indirect()
    beta, Lambda = 1.0, 0.5
    exact = _exact_indirect(s0, sT, p, E_A_fn, E_B_fn, beta, Lambda)

    samples, *_ = sampler.run_sampler(
        s0, sT, E_A_fn, E_B_fn, beta=beta, score='both',
        T=2, Lambda=Lambda, n_burn=3000, n_iter=200000, n_thin=10, seed=0,
    )
    counts = {}
    for path in samples:
        k = _path_key(path)
        counts[k] = counts.get(k, 0) + 1
    N = len(samples)
    emp = {k: c / N for k, c in counts.items()}
    for k, pr in exact.items():
        assert abs(emp.get(k, 0.0) - pr) < 0.03, \
            f"indirect path prob {emp.get(k, 0.0):.3f} vs exact {pr:.3f}"


def _restricted_energy(vals_at_p, p, big=40.0):
    '''Additive energy allowing residues {0,1,2} at position p.'''
    # nothing is forbidden, just priced out: e^-40 is far below sampling noise,
    # so the chain stays ergodic while the test enumerates a handful of paths
    M = np.full((L, sampler.Q_PEP), big, dtype=float)  # 40 is out of reach at beta = 1
    M[:, 0] = 0.0  # residue 0 is free at every position
    for z, e in enumerate(vals_at_p):
        M[p, z] = e

    def E_fn(seqs):
        seqs = np.atleast_2d(seqs).astype(np.int64)
        idx_i = np.broadcast_to(np.arange(L), seqs.shape)
        return M[idx_i, seqs].sum(axis=1)

    return E_fn


def _node(x, p):
    '''All-zero sequence of length L with value x at position p.'''
    s = np.zeros(L, dtype=np.int16)
    s[p] = x
    return s


def test_indirect_ergodic_matches_boltzmann():
    '''Paths with two interior nodes match the exact distribution.'''
    p = 3
    allowed = (0, 1, 2)  # nine paths to enumerate at T = 3
    s0, sT = _node(0, p), _node(1, p)
    E_A_fn = _restricted_energy((0.0, 0.4, 0.9), p)
    E_B_fn = _restricted_energy((0.3, 0.0, 0.6), p)
    beta, Lambda = 1.0, 0.7

    exact = {}
    for x1 in allowed:
        for x2 in allowed:
            path = [_node(0, p), _node(x1, p), _node(x2, p), _node(1, p)]
            fit = sampler.path_energy(path, E_A_fn, E_B_fn, sampler.score_both)
            elastic = sum(sampler.hamming(path[t], path[t + 1]) for t in range(3))
            exact[_path_key(path)] = np.exp(-beta * (fit + Lambda * elastic))
    Z = sum(exact.values())
    exact = {k: v / Z for k, v in exact.items()}

    samples, *_ = sampler.run_sampler(
        s0, sT, E_A_fn, E_B_fn, beta=beta, score='both',
        T=3, Lambda=Lambda, n_burn=5000, n_iter=200000, n_thin=10, seed=0,
    )
    counts = {}
    for path in samples:
        k = _path_key(path)
        counts[k] = counts.get(k, 0) + 1
    N = len(samples)
    emp = {k: c / N for k, c in counts.items()}

    for k, pr in exact.items():
        assert abs(emp.get(k, 0.0) - pr) < 0.03, \
            f"ergodic path prob {emp.get(k, 0.0):.3f} vs exact {pr:.3f}"
    # the priced-out residues are reachable, so check almost nothing went there
    leaked = 1.0 - sum(emp.get(k, 0.0) for k in exact)
    assert leaked < 0.01, f"{leaked:.3f} probability leaked to off-alphabet paths"


def test_mauri_matches_boltzmann():
    s0, sT, p, E_A_fn, E_B_fn = _setup_indirect()
    beta, Lambda = 1.0, 0.5
    exact = _exact_indirect(s0, sT, p, E_A_fn, E_B_fn, beta, Lambda)

    samples, *_ = sampler.run_sampler(
        s0, sT, E_A_fn, E_B_fn, beta=beta, score='both',
        T=2, Lambda=Lambda, proposal='mauri',
        n_burn=5000, n_iter=200000, n_thin=10, seed=0,
    )
    counts = {}
    for path in samples:
        k = _path_key(path)
        counts[k] = counts.get(k, 0) + 1
    N = len(samples)
    emp = {k: c / N for k, c in counts.items()}
    for k, pr in exact.items():
        assert abs(emp.get(k, 0.0) - pr) < 0.03, \
            f"mauri path prob {emp.get(k, 0.0):.3f} vs exact {pr:.3f}"


def test_mauri_direct_path_matches_boltzmann():
    '''Mauri's move on its own reproduces the distribution over substitution orders.'''
    s0, sT, diff, E_A_fn, E_B_fn = _setup()
    beta = 1.0
    exact = _exact_distribution(s0, sT, diff, E_A_fn, E_B_fn, beta)

    # run_sampler sends direct paths to biased_step for any proposal
    rng = np.random.default_rng(0)
    path = sampler.init_path_direct(s0, sT, rng=rng)
    counts = {}
    for i in range(300000):
        path, _ = sampler.mauri_step(path, E_A_fn, E_B_fn, sampler.score_both,
                                     beta=beta, Lambda=0.5, rng=rng)
        if i >= 5000 and i % 10 == 0:  # drop the first 5000 and keep one in ten
            k = _path_key(path)
            counts[k] = counts.get(k, 0) + 1
    N = sum(counts.values())
    for k, pr in exact.items():
        assert abs(counts.get(k, 0) / N - pr) < 0.02, \
            f"mauri direct path prob {counts.get(k, 0) / N:.3f} vs exact {pr:.3f}"


def test_mauri_case0_spur_matches_boltzmann():
    '''Mauri's move on paths that include spurs.'''
    p = 3
    allowed = (0, 1, 2)
    s0, sT = _node(0, p), _node(1, p)
    E_A_fn = _restricted_energy((0.0, 0.4, 0.9), p)
    E_B_fn = _restricted_energy((0.3, 0.0, 0.6), p)
    beta, Lambda = 1.0, 0.7

    exact = {}
    for x1 in allowed:
        for x2 in allowed:
            path = [_node(0, p), _node(x1, p), _node(x2, p), _node(1, p)]
            fit = sampler.path_energy(path, E_A_fn, E_B_fn, sampler.score_both)
            elastic = sum(sampler.hamming(path[t], path[t + 1]) for t in range(3))
            exact[_path_key(path)] = np.exp(-beta * (fit + Lambda * elastic))
    Z = sum(exact.values())
    exact = {k: v / Z for k, v in exact.items()}

    samples, *_ = sampler.run_sampler(
        s0, sT, E_A_fn, E_B_fn, beta=beta, score='both',
        T=3, Lambda=Lambda, proposal='mauri',
        n_burn=5000, n_iter=400000, n_thin=10, seed=0,
    )
    counts = {}
    for path in samples:
        k = _path_key(path)
        counts[k] = counts.get(k, 0) + 1
    N = len(samples)
    emp = {k: c / N for k, c in counts.items()}
    for k, pr in exact.items():
        assert abs(emp.get(k, 0.0) - pr) < 0.015, \
            f"mauri spur path prob {emp.get(k, 0.0):.4f} vs exact {pr:.4f}"
    leaked = 1.0 - sum(emp.get(k, 0.0) for k in exact)
    assert leaked < 0.01, f"{leaked:.3f} probability leaked to off-alphabet paths"


# at Lambda = 0 the elastic term vanishes and the acceptance ratio should
# reduce to the proposal, which isolates that branch of the move
def test_mauri_case1_acceptance_is_elastic_only():
    '''With Lambda = 0 the chain reproduces the proposal distribution.'''
    s0, sT, p, E_A_fn, E_B_fn = _setup_indirect()
    beta = 1.0
    exact = _exact_indirect(s0, sT, p, E_A_fn, E_B_fn, beta, 0.0)

    samples, *_ = sampler.run_sampler(
        s0, sT, E_A_fn, E_B_fn, beta=beta, score='both',
        T=2, Lambda=0.0, proposal='mauri',
        n_burn=5000, n_iter=200000, n_thin=10, seed=0,
    )
    counts = {}
    for path in samples:
        k = _path_key(path)
        counts[k] = counts.get(k, 0) + 1
    N = len(samples)
    for k, pr in exact.items():
        assert abs(counts.get(k, 0) / N - pr) < 0.03, \
            f"mauri Lambda=0 path prob {counts.get(k, 0) / N:.3f} vs exact {pr:.3f}"


def test_mauri_accepts_direct_paths():
    '''Direct paths are allowed with proposal='mauri'.'''
    s0, sT, diff, E_A_fn, E_B_fn = _setup()
    samples, *_ = sampler.run_sampler(
        s0, sT, E_A_fn, E_B_fn, beta=1.0, score='both',
        proposal='mauri', n_burn=10, n_iter=100, n_thin=10, seed=0,
    )
    assert len(samples) == 10  # n_iter over n_thin


# run the file directly to print the exact and sampled columns side by side,
# which is the table the Appendix reports
if __name__ == '__main__':
    s0, sT, diff, E_A_fn, E_B_fn = _setup()
    beta = 1.0
    exact = _exact_distribution(s0, sT, diff, E_A_fn, E_B_fn, beta)
    emp, N = _empirical_distribution(s0, sT, E_A_fn, E_B_fn, beta, n_burn=5000)

    print(f"\nD = {len(diff)} differing positions  ->  {len(exact)} possible direct paths")
    print(f"Sampler: {N} paths collected at beta = {beta}\n")
    print(f"{'path (mutation order)':<26}{'exact':>9}{'sampled':>9}{'|diff|':>8}")
    print("-" * 52)
    for perm in itertools.permutations(diff):
        path = [s0.copy()]
        cur = s0.copy()
        for p in perm:
            cur = cur.copy(); cur[p] = sT[p]; path.append(cur)
        k = _path_key(path)
        label = " -> ".join(f"P{p + 1}" for p in perm)
        ex, em = exact[k], emp.get(k, 0.0)
        print(f"{label:<26}{ex:>9.3f}{em:>9.3f}{abs(ex - em):>8.3f}")
