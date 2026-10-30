# Fixtures for the sampler tests. The energies are additive, one term per
# position, which is what lets a test enumerate the exact path distribution.

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sampler

L = sampler.L
Q = sampler.Q_PEP


# E(s) = sum_i M[i, s_i] for a random matrix M, the seed fixing the landscape
def _make_random_pwm_energy(seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((L, Q))

    def E_fn(seqs):
        # read off one entry per position and add them up, over a batch at once
        seqs = np.atleast_2d(seqs).astype(np.int64)
        idx_i = np.broadcast_to(np.arange(L), seqs.shape)
        return M[idx_i, seqs].sum(axis=1)

    return E_fn, M


def _path_key(path):
    '''Hashable, order-sensitive key for a path.'''
    return tuple(np.asarray(node).astype(np.int64).tobytes() for node in path)


@pytest.fixture
def energies():
    '''Two independent energy functions with fixed seeds.'''
    E_A_fn, _ = _make_random_pwm_energy(seed=11)
    E_B_fn, _ = _make_random_pwm_energy(seed=22)
    return E_A_fn, E_B_fn


@pytest.fixture
def endpoints():
    '''Two endpoints exactly 5 substitutions apart.'''
    rng = np.random.default_rng(0)
    s0 = rng.integers(0, Q, size=L, dtype=np.int16)
    sT = s0.copy()
    positions = rng.choice(L, size=5, replace=False)
    for p in positions:
        # redraw until the residue really changes, so the distance is exactly 5
        new = int(rng.integers(0, Q))
        while new == s0[p]:
            new = int(rng.integers(0, Q))
        sT[p] = new
    return s0, sT
