# The direct path ensemble solved exactly. A direct path makes each of the D
# endpoint mismatches once, so a node is fixed by which subset it has made,
# and that subset is a bitmask. Summing over the 2^D masks by level gives
# ln Z, the occupancies and the route statistics in one sweep.

import math

import numpy as np

from sampler import score_both, score_either, score_lambda, hamming

MAX_D_DEFAULT = 18  # 2^18 nodes is a big energy call


# the number of substitutions each mask has made, its set bits
def _popcount(masks):
    m = np.asarray(masks, dtype=np.int64).copy()
    out = np.zeros_like(m)
    while np.any(m):
        out += m & 1
        m >>= 1
    return out


# the second energy and the joint score this run should use
def _resolve_score(E_B_fn, score, weights, lam):
    # with one allele there is nothing to combine, so B contributes zero
    if E_B_fn is None:
        return (lambda seqs: np.zeros(np.atleast_2d(seqs).shape[0]),
                lambda a, b: a, 'single')
    if score == 'both':
        if len(weights) != 2:
            raise ValueError(f"weights must be (w_A, w_B), got {weights!r}")
        w_A, w_B = weights
        return E_B_fn, (lambda a, b: score_both(a, b, w_A, w_B)), score
    if score == 'either':
        return E_B_fn, score_either, score
    if score == 'lambda':
        return E_B_fn, (lambda a, b: score_lambda(a, b, lam)), score
    raise ValueError(f"score must be 'both', 'either' or 'lambda', got {score!r}")


def subset_states(s0, sT):
    '''The nodes a direct path from s0 to sT can visit, indexed by bitmask.'''
    s0 = np.asarray(s0)
    sT = np.asarray(sT)
    # only the mismatched positions ever change, so bit i of a mask stands for
    # the substitution at pos[i]
    pos = np.flatnonzero(s0 != sT)
    D = len(pos)
    if D == 0:
        raise ValueError("s0 and sT are identical.")
    # build all 2^D nodes at once, starting from s0 everywhere
    n = 1 << D
    masks = np.arange(n, dtype=np.int64)
    states = np.repeat(s0[None, :], n, axis=0)
    # wherever bit i is set, that position has been switched to its sT residue
    for i in range(D):
        sel = ((masks >> i) & 1).astype(bool)
        states[sel, pos[i]] = sT[pos[i]]
    return pos, states, _popcount(masks)


# the joint score of every node, in batches so the energy call stays in memory
def _node_scores(states, E_A_fn, E_B_fn, score_fn, batch_size=8192):
    out = np.empty(len(states), dtype=float)
    for lo in range(0, len(states), batch_size):
        chunk = states[lo:lo + batch_size]
        out[lo:lo + batch_size] = np.asarray(
            score_fn(E_A_fn(chunk), E_B_fn(chunk)), dtype=float)
    return out


def _forward(e, level, beta, allowed=None):
    '''Log sum over partial routes into each mask, node energies included.'''
    n = len(e)
    D = int(level.max())
    f = np.full(n, -np.inf)
    ok = np.ones(n, dtype=bool) if allowed is None else allowed
    if not ok[0]:
        return f
    f[0] = -beta * e[0]
    # work up one level at a time, so every predecessor is already done
    for k in range(1, D + 1):
        ms = np.flatnonzero(level == k)
        ms = ms[ok[ms]]
        if len(ms) == 0:
            continue
        # a mask is reached from each mask one bit lower, so clear bit i in turn
        acc = np.full(len(ms), -np.inf)
        for i in range(D):
            has = ((ms >> i) & 1).astype(bool)
            if not has.any():
                continue
            acc[has] = np.logaddexp(acc[has], f[ms[has] ^ (1 << i)])
        f[ms] = -beta * e[ms] + acc
    return f


def _backward(e, level, beta):
    '''Log sum over completions from each mask to the full mask, own node excluded.'''
    n = len(e)
    D = int(level.max())
    # the full mask is the endpoint and needs no completion
    g = np.full(n, -np.inf)
    g[n - 1] = 0.0
    # come back down the levels, setting each bit that is still free
    for k in range(D - 1, -1, -1):
        ms = np.flatnonzero(level == k)
        acc = np.full(len(ms), -np.inf)
        for i in range(D):
            free = (((ms >> i) & 1) == 0)
            if not free.any():
                continue
            # the successor's own energy belongs to g, this mask's does not,
            # so that f + g counts every node exactly once
            up = ms[free] | (1 << i)
            acc[free] = np.logaddexp(acc[free], -beta * e[up] + g[up])
        g[ms] = acc
    return g


# the route with the lowest total energy, which is the most likely one
def _min_sum(e, level):
    n = len(e)
    D = int(level.max())
    m = np.full(n, np.inf)
    back = np.full(n, -1, dtype=int)
    m[0] = e[0]
    for k in range(1, D + 1):
        for mask in np.flatnonzero(level == k):
            best, arg = np.inf, -1
            for i in range(D):
                if (mask >> i) & 1:
                    v = m[mask ^ (1 << i)]
                    if v < best:
                        best, arg = v, i
            m[mask] = e[mask] + best
            back[mask] = arg
    return m, back


# the route whose highest node is as low as possible, the best way over
def _minimax(e, level):
    n = len(e)
    D = int(level.max())
    b = np.full(n, np.inf)
    back = np.full(n, -1, dtype=int)
    b[0] = e[0]
    for k in range(1, D + 1):
        for mask in np.flatnonzero(level == k):
            best, arg = np.inf, -1
            for i in range(D):
                if (mask >> i) & 1:
                    v = max(b[mask ^ (1 << i)], e[mask])
                    if v < best:
                        best, arg = v, i
            b[mask] = best
            back[mask] = arg
    return b, back


# follow the back pointers from the full mask to recover the order of moves
def _traceback(back, D, pos):
    mask = (1 << D) - 1
    order = []
    while mask:
        i = int(back[mask])
        order.append(int(pos[i]))
        mask ^= (1 << i)
    return tuple(order[::-1])


def exact_direct_ensemble(s0, sT, E_A_fn, E_B_fn=None, beta=1.0,
                          score='both', weights=(1.0, 1.0), lam=0.0,
                          max_D=MAX_D_DEFAULT, batch_size=8192):
    '''Solve the direct path ensemble exactly by dynamic programming over subsets.

    Args:
        score (str): Joint score, 'both', 'either' or 'lambda'.
        lam (float): Regime knob for the lambda score.
        max_D (int): Largest Hamming distance to enumerate, a guard because
            the state space is 2^D.
        batch_size (int): Nodes in each energy call.

    Returns:
        dict: The exact ensemble. Keys:
            D, beta, positions (mismatched sequence positions, shape (D,)),
            node_energy (2^D,), level (2^D,), logZ, mean_energy, S_path,
            S_path_norm (S_path / log(D!)), n_eff (exp S_path),
            occupancy (2^D, probability the node is visited),
            step_marginals (D, D) with [j, i] = P(substitution j is at positions[i]),
            level_mean_energy (D+1,), map_order, map_energy, map_logprob,
            bottleneck (lowest peak any route reaches), bottleneck_order,
            endpoint_mean, barrier_min (bottleneck minus endpoint_mean).
    '''
    s0 = np.asarray(s0)
    sT = np.asarray(sT)
    D = hamming(s0, sT)
    if D == 0:
        raise ValueError("s0 and sT are identical.")
    if D > max_D:
        raise ValueError(
            f"D = {D} is above max_D = {max_D}."
        )
    E_B_fn, score_fn, _ = _resolve_score(E_B_fn, score, weights, lam)

    pos, states, level = subset_states(s0, sT)
    e = _node_scores(states, E_A_fn, E_B_fn, score_fn, batch_size=batch_size)

    # f reaches each mask and g completes it, so f + g is the weight of all
    # routes through that node and f at the full mask is ln Z
    f = _forward(e, level, beta)
    g = _backward(e, level, beta)
    logZ = float(f[-1])
    occ = np.exp(f + g - logZ)

    # the flux across each level boundary says which substitution is made when
    step = np.zeros((D, D))  # probability substitution j is at mismatch i
    for k in range(D):
        ms = np.flatnonzero(level == k)
        for i in range(D):
            free = (((ms >> i) & 1) == 0)
            if not free.any():
                continue
            up = ms[free] | (1 << i)
            terms = f[ms[free]] - beta * e[up] + g[up] - logZ
            mx = terms.max()
            step[k, i] = float(np.exp(mx) * np.exp(terms - mx).sum()) if np.isfinite(mx) else 0.0

    # S = beta <E> + ln Z, and all D! orders being equally likely would give
    # log(D!), so the ratio says how much of the available spread is used
    mean_energy = float((e * occ).sum())
    S_path = beta * mean_energy + logZ
    log_max = math.log(math.factorial(D)) if D >= 2 else 0.0

    # the two single best routes, by total energy and by highest node
    m, m_back = _min_sum(e, level)
    b, b_back = _minimax(e, level)
    map_order = _traceback(m_back, D, pos)
    map_energy = float(m[-1])
    # the endpoints are both good binders, so they set the barrier baseline
    endpoint_mean = 0.5 * (e[0] + e[-1])

    level_mean = np.array([float((e[level == k] * occ[level == k]).sum())
                           for k in range(D + 1)])

    return {
        'D': D, 'beta': float(beta), 'positions': pos,
        'node_energy': e, 'level': level,
        'logZ': logZ, 'mean_energy': mean_energy,
        'S_path': float(S_path),
        'S_path_norm': float(S_path / log_max) if log_max > 0 else 0.0,
        'n_eff': float(np.exp(S_path)),
        'occupancy': occ, 'step_marginals': step, 'level_mean_energy': level_mean,
        'map_order': map_order, 'map_energy': map_energy,
        'map_logprob': float(-beta * map_energy - logZ),
        'bottleneck': float(b[-1]),
        'bottleneck_order': _traceback(b_back, D, pos),
        'endpoint_mean': float(endpoint_mean),
        'barrier_min': float(b[-1] - endpoint_mean),
    }


# the ensemble printed as a block, for the logs behind my thesis tables
def format_report(res, barrier=None):
    D = res['D']
    lines = [
        f"exact direct ensemble   D={D}  beta={res['beta']:.3g}  "
        f"nodes={1 << D}  routes={math.factorial(D)}",
        f"  logZ                {res['logZ']:+.4f}",
        f"  <E_path>            {res['mean_energy']:+.4f}",
        f"  S_path              {res['S_path']:.4f} nats"
        f"   (max log(D!) = {math.log(math.factorial(D)):.4f})",
        f"  S_path_norm         {res['S_path_norm']:.4f}",
        f"  n_eff routes        {res['n_eff']:.1f} of {math.factorial(D)}",
        f"  MAP route           {res['map_order']}  "
        f"E={res['map_energy']:+.4f}  P={math.exp(res['map_logprob']):.3g}",
        f"  bottleneck peak     {res['bottleneck']:+.4f}  "
        f"(barrier_min {res['barrier_min']:+.4f})",
        f"  bottleneck route    {res['bottleneck_order']}",
    ]
    if barrier is not None:
        which = 'interior' if barrier['interior'] else 'all-node'
        note = '  DEGENERATE (constant over routes)' if barrier['degenerate'] else ''
        lines.append(f"  <barrier> {which:<9} {barrier['mean_barrier']:+.4f}  "
                     f"({len(barrier['peaks'])} achievable peaks){note}")
    return "\n".join(lines)
