# The summary statistics taken off a sampled ensemble: how spread out the
# routes are, where one allele gives way to the other, and how high the paths
# climb. summarise_pair returns all three as one row of the scan tables.

import math
from collections import Counter

import numpy as np


# the sampled paths as a list of arrays, one array per path
def _as_path_list(samples):
    if isinstance(samples, np.ndarray):
        return [samples[i] for i in range(len(samples))]
    return [np.asarray(p) for p in samples]


# the sequence position that changes at each step, in the order they change
def substitution_order(path):
    path = np.asarray(path)
    order = []
    for t in range(len(path) - 1):
        diff = np.where(path[t] != path[t + 1])[0]
        # a direct path changes exactly one position per step, so anything
        # else means the path has a spur and has no substitution order
        if len(diff) != 1:
            raise ValueError(
                f"non-direct step at t={t}: {len(diff)} positions changed "
                f"(expected exactly 1)."
            )
        order.append(int(diff[0]))
    return tuple(order)


# total number of residue substitutions a path makes, summed over its steps
def mutational_load(path):
    p = np.asarray(path)
    return int((p[:-1] != p[1:]).sum())


# how spread out the sampled paths are over the distinct substitution orders
def route_entropy(samples):
    paths = _as_path_list(samples)
    if len(paths) == 0:
        return {'S_path': 0.0, 'n_distinct': 0, 'n_eff': 0.0, 'S_path_norm': 0.0}
    # treat each distinct order as one outcome and take its Shannon entropy
    orders = [substitution_order(p) for p in paths]
    counts = np.array(list(Counter(orders).values()), dtype=float)
    probs = counts / counts.sum()
    S = float(-np.sum(probs * np.log(probs)))
    # T! orders are available, so log(T!) is the entropy of using all of them
    T = len(orders[0])
    log_max = math.log(math.factorial(T)) if T >= 2 else 0.0
    return {
        'S_path': S,
        'n_distinct': int(len(counts)),
        'n_eff': float(np.exp(S)),
        'S_path_norm': S / log_max if log_max > 0 else 0.0,
    }


def _node_energies(samples, E_A_fn, E_B_fn):
    '''Return both energies at every node of every path.'''
    paths = _as_path_list(samples)
    n = len(paths)
    Tp1 = len(paths[0])
    flat = np.concatenate([np.asarray(p) for p in paths], axis=0)  # shape (n*Tp1, L)
    EA = np.asarray(E_A_fn(flat), dtype=float).reshape(n, Tp1)
    EB = np.asarray(E_B_fn(flat), dtype=float).reshape(n, Tp1)
    return EA, EB


# where along the path the second allele takes over as the better binder
def switch_stats(samples, E_A_fn, E_B_fn):
    EA, EB = _node_energies(samples, E_A_fn, E_B_fn)
    n, Tp1 = EA.shape
    T = Tp1 - 1
    b_better = EB < EA
    # the first node where B wins, as a fraction of the path length
    crossings, switched = [], 0
    for i in range(n):
        idx = np.where(b_better[i])[0]
        if len(idx) > 0:
            crossings.append(idx[0] / T)
            switched += 1
    # the endpoints are fixed and B always wins at v_T, so they are dropped here
    interior = b_better[:, 1:-1] if T >= 2 else b_better
    return {
        'crossover_frac': float(np.mean(crossings)) if crossings else float('nan'),
        'crossover_std': float(np.std(crossings)) if crossings else float('nan'),
        'switch_rate': switched / n if n else float('nan'),
        'b_fraction': float(interior.mean()),
    }


# how high the paths have to climb, under each of the two joint scores
def barrier_stats(samples, E_A_fn, E_B_fn, weights=(1.0, 1.0)):
    EA, EB = _node_energies(samples, E_A_fn, E_B_fn)
    wA, wB = weights
    # 'both' asks a node to bind each allele, 'either' only the better of them
    both = wA * EA + wB * EB
    either = np.minimum(EA, EB)
    # measure the peak against the endpoints, which are the two good binders
    endpoint_both = 0.5 * (both[:, 0] + both[:, -1])
    barrier = both.max(axis=1) - endpoint_both
    return {
        'barrier_both_mean': float(barrier.mean()),
        'peak_both_mean': float(both.max(axis=1).mean()),
        'worst_either_mean': float(either.max(axis=1).mean()),
    }


def summarise_pair(samples, E_A_fn, E_B_fn, weights=(1.0, 1.0), extra=None):
    '''Return the route, switch and barrier statistics in one flat record.'''
    paths = _as_path_list(samples)
    row = {
        'n_samples': len(paths),
        'T': (len(paths[0]) - 1) if paths else 0,
    }
    row.update(route_entropy(samples))
    row.update(switch_stats(samples, E_A_fn, E_B_fn))
    row.update(barrier_stats(samples, E_A_fn, E_B_fn, weights=weights))
    if extra:
        row.update(extra)
    return row
