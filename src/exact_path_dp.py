# The path measure solved exactly with no factorisation over sites, as the
# reference mean_field.py is checked against. The state at each step is a
# whole sequence, so this only runs on a restricted alphabet; see
# build_alphabets.

import itertools

import numpy as np


# Keep each site down to the residues that matter: the two endpoint residues
# always, plus any named in extra, plus the n_extra best-supported ones from a
# set of marginals. Everything else is dropped from the state space.
def build_alphabets(s0, sT, extra=None, n_extra=0, marg=None):
    '''Return the residues allowed at each site.'''
    s0 = np.asarray(s0, dtype=np.int64)
    sT = np.asarray(sT, dtype=np.int64)
    L = len(s0)
    alphabets = []
    for i in range(L):
        allowed = {int(s0[i]), int(sT[i])}
        if extra is not None:
            allowed |= {int(a) for a in extra[i]}
        if n_extra > 0 and marg is not None:
            rank = np.argsort(-marg[:, i, :].max(axis=0))
            for a in rank:
                if len(allowed) >= len({int(s0[i]), int(sT[i])}) + n_extra:
                    break
                allowed.add(int(a))
        alphabets.append(np.array(sorted(allowed), dtype=np.int64))
    return alphabets


# every sequence the alphabets allow, and the shape of the array they index
def enumerate_states(alphabets):
    shape = tuple(len(a) for a in alphabets)
    states = np.array(list(itertools.product(*[a.tolist() for a in alphabets])),
                      dtype=np.int64)
    return states, shape


# One step of the elastic transfer operator, where off = exp(-beta Lambda) is
# what a single mutation costs. The operator factorises over sites, so it is
# applied one axis at a time rather than built as a matrix.
def _apply_elastic(x, off, wall=True):
    # without the wall any number of sites may change at once, which is the
    # product of the per-site operators
    if not wall:
        out = x
        for ax in range(x.ndim):
            out = off * out.sum(axis=ax, keepdims=True) + (1.0 - off) * out
        return out
    # with it a step either stays put or changes one site, so the terms are
    # added rather than multiplied and nothing changes two sites at once
    out = x.copy()
    for ax in range(x.ndim):
        out = out + off * (x.sum(axis=ax, keepdims=True) - x)
    return out


def solve_exact(score_fn, s0, sT, T, beta, Lambda, alphabets, wall=True, q=None):
    '''Return ln Z, the site marginals and the statistics for each step.

    Args:
        score_fn (callable): Maps sequences (n, L) to scores (n,).
        alphabets (list[np.ndarray]): Allowed residues at each site, which is
            what keeps the enumeration below finite.
        wall (bool): Hold the path to one mutation per step.
        q (int | None): Width of the returned marginals, taken from the
            alphabets when None.

    Returns:
        res (dict): The exact solution. Keys: logZ on this space, marg
            (T-1, L, q) site marginals padded to the full alphabet, q_traj
            (T,) and d_traj (T,) the overlap and mean mutations at each step,
            E_traj (T-1,) the mean score at each interior step, and n_states.
    '''
    s0 = np.asarray(s0, dtype=np.int64)
    sT = np.asarray(sT, dtype=np.int64)
    L = len(s0)
    off = float(np.exp(-beta * Lambda))
    q_out = int(max(a.max() for a in alphabets)) + 1 if q is None else int(q)

    states, shape = enumerate_states(alphabets)
    n_states = states.shape[0]
    scores = np.asarray(score_fn(states), dtype=np.float64)
    logw = -beta * scores
    eta = np.exp(logw - logw.max()).reshape(shape)
    eta_shift = float(logw.max())

    def indicator(seq):
        m = np.zeros(shape)
        idx = tuple(int(np.searchsorted(alphabets[i], seq[i])) for i in range(L))
        if any(alphabets[i][idx[i]] != seq[i] for i in range(L)):
            raise ValueError('endpoint not contained in the restricted alphabet')
        m[idx] = 1.0
        return m

    alphas = [indicator(s0)]
    logscale = [0.0]
    for t in range(1, T):
        a = _apply_elastic(alphas[-1], off, wall) * eta
        c = a.max()
        alphas.append(a / c)
        logscale.append(logscale[-1] + np.log(c) + eta_shift)
    aT = _apply_elastic(alphas[-1], off, wall) * indicator(sT)
    if aT.sum() <= 0.0:  # too few steps to reach the end sequence
        return dict(logZ=-np.inf, marg=np.zeros((max(T - 1, 0), L, q_out)),
                    q_traj=np.full(T, np.nan), d_traj=np.full(T, np.nan),
                    E_traj=np.full(max(T - 1, 0), np.nan), n_states=n_states)
    logZ = float(np.log(aT.sum()) + logscale[-1])

    betas = [None] * (T + 1)
    betas[T] = indicator(sT)
    for t in range(T - 1, 0, -1):
        b = _apply_elastic(betas[t + 1], off, wall) * eta
        betas[t] = b / b.max()
    alphas.append(aT)

    # the two messages meet at each interior step, giving the full joint law
    # over sequences there; one copy of the node weight is divided back out
    marg = np.zeros((T - 1, L, q_out))
    E_traj = np.zeros(T - 1)
    for t in range(1, T):
        joint = alphas[t] * betas[t] / eta
        joint = joint / joint.sum()
        flat = joint.reshape(-1)
        E_traj[t - 1] = float((flat * scores).sum())
        # a site's marginal is the joint with every other axis summed away
        for i in range(L):
            axes = tuple(ax for ax in range(L) if ax != i)
            p = joint.sum(axis=axes)
            marg[t - 1, i, alphabets[i]] = p / p.sum()

    # mutations at each step from consecutive messages
    d_traj = np.zeros(T)
    for t in range(T):
        left = alphas[t] if t > 0 else indicator(s0)
        right = betas[t + 1]
        den = float((left * _apply_elastic(right, off, wall)).sum())
        if wall:
            p_stay = float((left * right).sum()) / den
            d_traj[t] = 1.0 - p_stay  # d is 0 or 1 here
        else:
            # <delta> at each site by contracting every axis but i
            num = 0.0
            for i in range(L):
                r = right
                for ax in range(L):
                    if ax == i:
                        continue
                    r = off * r.sum(axis=ax, keepdims=True) + (1.0 - off) * r
                num += float((left * r).sum())
            d_traj[t] = L * (1.0 - num / den / L)
    q_traj = 1.0 - d_traj / L
    return dict(
        logZ=logZ,
        marg=marg,
        q_traj=q_traj,
        d_traj=d_traj,
        E_traj=E_traj,
        n_states=n_states,
    )
