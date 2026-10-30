# The convergence diagnostics checked against AR(1), whose autocorrelation
# time and effective sample size are known in closed form, so the estimators
# are compared against a number rather than against each other.

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import diagnostics as dg


def _ar1(phi, n, rng, mu=0.0):
    '''Stationary AR(1) series with unit marginal variance.'''
    x = np.empty(n)
    x[0] = rng.standard_normal()
    # this scaling on the noise is what holds the marginal variance at one,
    # so the correlation is the only thing phi changes
    s = np.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + s * rng.standard_normal()
    return x + mu


def _ar1_chains(phi, m, n, seed, mu=0.0):
    '''Stack of m chains from one seeded generator.'''
    rng = np.random.default_rng(seed)
    return np.array([_ar1(phi, n, rng, mu) for _ in range(m)])


def test_iat_iid_is_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    tau = dg.autocorr_time(x)
    assert 0.8 < tau < 1.4, tau


def test_iat_ar1_matches_theory():
    for phi in (0.5, 0.8, 0.9):
        rng = np.random.default_rng(int(100 * phi))
        # average over repeats, since one estimate of tau is noisy
        taus = [dg.autocorr_time(_ar1(phi, 40000, rng)) for _ in range(6)]
        tau = np.mean(taus)
        true = (1 + phi) / (1 - phi)
        assert abs(tau - true) / true < 0.2, (phi, tau, true)


def test_ess_iid_is_near_n():
    chains = _ar1_chains(0.0, m=4, n=8000, seed=1)
    S = dg.ess(chains)
    N = chains.size
    assert 0.8 * N < S < 1.2 * N, (S, N)


def test_ess_ar1_matches_theory():
    phi = 0.8
    chains = _ar1_chains(phi, m=6, n=20000, seed=2)
    S = dg.ess(chains)
    N = chains.size
    true = N * (1 - phi) / (1 + phi)  # N/9 at phi=0.8
    assert abs(S - true) / true < 0.25, (S, true)


def test_ess_never_exceeds_n_by_much():
    chains = _ar1_chains(0.3, m=4, n=5000, seed=3)
    assert dg.ess(chains) <= 1.1 * chains.size  # the 10 percent is estimator noise


def test_rhat_mixed_chains_near_one():
    chains = _ar1_chains(0.7, m=6, n=10000, seed=4)
    rh = dg.split_rhat(chains)
    assert 0.99 < rh < 1.02, rh


# chains that never meet should be caught, however well each one mixes
def test_rhat_flags_offset_chains():
    rng = np.random.default_rng(5)
    offsets = np.array([-3.0, -1.0, 1.0, 3.0])  # far apart at unit variance
    chains = np.array([_ar1(0.7, 6000, rng, mu=o) for o in offsets])
    rh = dg.split_rhat(chains)
    assert rh > 1.2, rh


def test_rhat_flags_within_chain_drift():
    '''A single drifting chain is caught.'''
    n = 8000
    rng = np.random.default_rng(6)
    drift = np.linspace(0, 6, n) + 0.3 * rng.standard_normal(n)
    rh = dg.split_rhat(drift[None, :])
    assert rh > 1.2, rh


def test_mcse_consistent_with_sd_over_sqrt_ess():
    '''The MCSE matches its own sd and ESS.'''
    chains = _ar1_chains(0.8, m=6, n=20000, seed=7)
    d = dg.diagnose(chains, name='ar1')
    approx = d['sd'] / np.sqrt(d['ess'])
    assert abs(d['mcse'] - approx) / approx < 1e-6
    assert d['ok']


# the recommendation has to be in sampler steps, not in kept draws, so a run
# already thinned by 10 should ask for about 10 times its own tau
def test_diagnose_reports_thinning_in_raw_steps():
    chains = _ar1_chains(0.9, m=4, n=20000, seed=8)  # true tau = 19
    d = dg.diagnose(chains, thin=10, name='ar1')
    assert 120 < d['rec_thin_steps'] < 260, d['rec_thin_steps']


def test_batch_means_mcse():
    '''Batch means gives the iid standard error on uncorrelated draws.'''
    rng = np.random.default_rng(9)
    vals = 5.0 + 0.5 * rng.standard_normal(40)
    mean, mcse = dg.batch_means_mcse(vals)
    assert abs(mean - 5.0) < 0.3
    assert abs(mcse - 0.5 / np.sqrt(40)) < 0.03


def test_constant_series_is_safe():
    '''A zero variance series gives finite diagnostics instead of nan.'''
    chains = np.full((4, 1000), 3.0)
    d = dg.diagnose(chains, name='const')
    assert np.isfinite(d['split_rhat'])
    assert d['iat'] == 1.0
