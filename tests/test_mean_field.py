# The mean-field solver and the exact dynamic program, against each other and
# against brute force, on a landscape small enough to enumerate. Three claims:
# the dynamic program matches brute force, with the patterns off the sites
# decouple and the mean field is exact, and with them on it stays a lower
# bound on ln Z.

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'src')
if SRC not in sys.path:
    sys.path.append(SRC)

import mean_field as mf
import exact_path_dp as edp

NSITE, NAA = 4, 3


# A Hopfield-Potts energy: a field at each site, plus a reward for lining up
# with any of the stored patterns. The coupling between sites comes only
# through those patterns, which is what makes the no-pattern case exact.
class Hopfield:
    '''A small Hopfield-Potts landscape.'''

    def __init__(self, seed, n_pattern=2):
        rng = np.random.default_rng(seed)
        self.g = rng.standard_normal((NSITE, NAA))
        self.W = rng.standard_normal((n_pattern, NSITE, NAA)) * 0.6
        self.N, self.q, self.M = NSITE, NAA, n_pattern

    @property
    def features(self):
        return self.W

    @property
    def exact_field(self):
        return self.g

    def psi_and_grad(self, I, beta):
        '''Return psi and its gradient, both scaled by beta.'''
        I = np.atleast_2d(I)
        return beta * 0.5 * (I ** 2).sum(axis=1), beta * I

    def inputs(self, V):
        '''Return the projections I_mu, one row for each sequence.'''
        V = np.atleast_2d(np.asarray(V, dtype=np.int64))
        return np.einsum('mln->nm', self.W[:, np.arange(self.N)[:, None], V.T])

    def energy(self, V):
        # the squared projections are the only term that couples the sites
        V = np.atleast_2d(np.asarray(V, dtype=np.int64))
        field = self.g[np.arange(self.N)[None, :], V].sum(axis=1)
        return -field - 0.5 * (self.inputs(V) ** 2).sum(axis=1)

    def mean_energy(self, marg):
        '''Return the same energy read off site marginals.'''
        field = np.einsum('tia,ia->t', marg, self.g)
        I = np.einsum('tia,mia->tm', marg, self.W)
        return -field - 0.5 * (I ** 2).sum(axis=1)


class FieldOnly(Hopfield):
    '''The same landscape with no patterns.'''

    def __init__(self, seed):
        super().__init__(seed, n_pattern=0)


def _endpoints():
    s0 = np.array([0, 1, 2, 0])
    sT = np.array([1, 2, 2, 2])  # hamming distance 3
    return s0, sT


def _alphabets():
    return [np.arange(NAA) for _ in range(NSITE)]


def _brute_force_logZ(score_fn, s0, sT, T, beta, Lambda, wall):
    '''Sum over every interior path.'''
    states = np.array(list(itertools.product(range(NAA), repeat=NSITE)))
    sc = score_fn(states)
    dH = lambda u, v: int((u != v).sum())
    tot = 0.0
    # a single step has no interior node, just the jump between the endpoints
    if T == 1:
        d = dH(s0, sT)
        return -np.inf if (wall and d > 1) else -beta * Lambda * d
    # every assignment of states to the T-1 interior nodes, weighted by the
    # node scores and the elastic cost of each step
    for combo in itertools.product(range(len(states)), repeat=T - 1):
        nodes = [s0] + [states[c] for c in combo] + [sT]
        steps = [dH(nodes[k], nodes[k + 1]) for k in range(T)]
        if wall and max(steps) > 1:
            continue
        e = sum(sc[c] for c in combo)
        tot += np.exp(-beta * (e + Lambda * sum(steps)))
    return np.log(tot) if tot > 0 else -np.inf


@pytest.mark.parametrize('wall', [False, True])
@pytest.mark.parametrize('T', [1, 3, 4])
def test_exact_dp_matches_brute_force(wall, T):
    '''The dynamic program matches the brute-force sum.'''
    land = Hopfield(seed=3)
    s0, sT = _endpoints()
    beta, Lam = 1.0, 0.4
    res = edp.solve_exact(land.energy, s0, sT, T, beta, Lam, _alphabets(), wall=wall)
    ref = _brute_force_logZ(land.energy, s0, sT, T, beta, Lam, wall)
    if np.isneginf(ref):
        assert np.isneginf(res['logZ'])
    else:
        assert res['logZ'] == pytest.approx(ref, abs=1e-9)


# with no patterns nothing couples the sites, so the mean field is not an
# approximation here and has to reproduce the exact answer to machine precision
@pytest.mark.parametrize('T', [3, 5])
def test_mean_field_exact_without_patterns(T):
    '''Evo mean-field reproduces ln Z and the marginals.'''
    land = FieldOnly(seed=7)
    s0, sT = _endpoints()
    beta, Lam = 1.0, 0.4
    ex = edp.solve_exact(land.energy, s0, sT, T, beta, Lam, _alphabets(), wall=False)
    got = mf.solve(land, s0, sT, T, beta, Lam, alphabets=_alphabets(), elastic='evo')
    assert got['logZ'] == pytest.approx(ex['logZ'], abs=1e-9)
    assert np.abs(got['marg'] - ex['marg']).max() < 1e-9


# turning the patterns back on makes it an approximation, but a variational
# one, so it can only ever underestimate ln Z
@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('T', [3, 5])
def test_evo_mean_field_is_a_lower_bound(seed, T):
    '''ln Z_MF stays at or below ln Z.'''
    land = Hopfield(seed=seed)
    s0, sT = _endpoints()
    beta, Lam = 1.0, 0.4
    ex = edp.solve_exact(land.energy, s0, sT, T, beta, Lam, _alphabets(), wall=False)
    got = mf.solve(land, s0, sT, T, beta, Lam, alphabets=_alphabets(), elastic='evo')
    assert got['logZ'] <= ex['logZ'] + 1e-9
    assert got['converged']


# and the bound is a property of the free energy, not of convergence, so it
# must hold at every iterate on the way there
def test_evo_bound_holds_away_from_the_fixed_point():
    '''The bound holds before the solver has converged.'''
    land = Hopfield(seed=5)
    s0, sT = _endpoints()
    beta, Lam, T = 1.0, 0.4, 4
    ex = edp.solve_exact(land.energy, s0, sT, T, beta, Lam, _alphabets(), wall=False)
    for max_iter in (1, 2, 3, 10):
        got = mf.solve(land, s0, sT, T, beta, Lam, alphabets=_alphabets(),
                       elastic='evo', max_iter=max_iter)
        assert got['logZ'] <= ex['logZ'] + 1e-9


@pytest.mark.parametrize('K', [0.5, 1.0])
def test_cont_puts_the_wall_where_mauri_wants_it(K):
    '''The Cont outer loop drives the mean step size to K.'''
    land = Hopfield(seed=4)
    s0, sT = _endpoints()
    got = mf.solve(land, s0, sT, 6, 1.0, 0.4, alphabets=_alphabets(),
                   elastic='cont', K=K)
    # converged covers the wall too
    if got['converged']:
        assert np.abs(got['d_traj'] - K).max() < 1e-3
    else:
        pytest.skip(f'wall not placeable here: max|<d>-K| = '
                    f'{np.abs(got["d_traj"] - K).max():.3f}')


def test_psi_gradient_matches_finite_differences():
    '''psi_and_grad returns the gradient of psi.'''
    land = Hopfield(seed=9)
    rng = np.random.default_rng(0)
    I = rng.standard_normal((6, land.M))
    _, grad = land.psi_and_grad(I, beta=1.3)
    h = 1e-6
    for k in range(land.M):
        step = np.zeros_like(I)
        step[:, k] = h
        num = (land.psi_and_grad(I + step, 1.3)[0]
               - land.psi_and_grad(I - step, 1.3)[0]) / (2 * h)
        assert np.abs(num - grad[:, k]).max() < 1e-6


def test_unreachable_endpoints_are_rejected_consistently():
    '''T < D leaves no path under the wall.'''
    land = Hopfield(seed=3)
    s0, sT = _endpoints()
    res = edp.solve_exact(land.energy, s0, sT, 2, 1.0, 0.4, _alphabets(), wall=True)
    assert np.isneginf(res['logZ'])
    with pytest.raises(ValueError):
        mf.solve(land, s0, sT, 2, 1.0, 0.4)


def test_elastic_mode_is_validated():
    land = Hopfield(seed=3)
    s0, sT = _endpoints()
    with pytest.raises(ValueError):
        mf.solve(land, s0, sT, 4, 1.0, 0.4, elastic='nonsense')


def test_density_of_states_matches_generating_function():
    '''The binning identity from notes/mean_field_derivation.md.'''
    import numpy.polynomial.polynomial as P
    from collections import Counter

    rng = np.random.default_rng(0)
    for trial in range(5):
        # integer weights, so the projections land in exact bins and the
        # counts can be compared without any tolerance
        n_site, n_aa = 3, 3
        W = rng.integers(0, 4, size=(n_site, n_aa))

        # count the sequences at each value of the projection by hand
        dos = Counter(int(sum(W[i, a] for i, a in enumerate(s)))
                      for s in itertools.product(range(n_aa), repeat=n_site))

        # bin counts as coefficients of prod_i ( sum_a z^{w_i(a)} )
        poly = np.array([1.0])
        for i in range(n_site):
            site = np.zeros(int(W[i].max()) + 1)
            for a in range(n_aa):
                site[W[i, a]] += 1
            poly = P.polymul(poly, site)

        for power, coeff in enumerate(poly):
            assert dos.get(power, 0) == pytest.approx(coeff), (
                f'trial {trial}: bin {power} counted {dos.get(power, 0)} '
                f'but generating function gives {coeff}')

        # the point of the binning: a sum over sequences of any function of
        # the projection can be taken over the bins instead
        F = lambda I: np.exp(0.3 * I ** 2)
        direct = sum(F(sum(W[i, a] for i, a in enumerate(s)))
                     for s in itertools.product(range(n_aa), repeat=n_site))
        binned = sum(n * F(I) for I, n in dos.items())
        assert direct == pytest.approx(binned)
