# The Markov chain over mutational paths. A path runs from s0 to sT and the
# chain resamples one interior node at a time, holding the endpoints fixed.
# A direct path takes D steps and only the order of the substitutions varies;
# an indirect path gets more steps, so its nodes can detour at a cost Lambda.
# tests/test_boltzmann_proof.py checks each move against an exact answer.

import json
import os
import sys

import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from encoding import convert_number, convert_letter
from config import (L, Q_PEP, AA_DICT, PATHS_DIR, ANCHOR_POSITIONS,
                    PGM_SOURCE, PGM_UTILS)

# the PGM engine has modules whose names collide with the ones here, so put it
# at the front of the path and move it there even if it is already on it
for _p in (PGM_UTILS, PGM_SOURCE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


# a numpy random generator, built from the argument when that is a seed
def _as_rng(rng_or_seed):
    if isinstance(rng_or_seed, np.random.Generator):
        return rng_or_seed
    return np.random.default_rng(rng_or_seed)


# joint score when both alleles are present, the weighted sum of the two energies
def score_both(E_A, E_B, w_A=1.0, w_B=1.0):
    return w_A * E_A + w_B * E_B


# joint score when either allele suffices, the smaller of the two energies
def score_either(E_A, E_B):
    return np.minimum(E_A, E_B)


# joint score between either and both, the smaller energy plus lam times the larger
def score_lambda(E_A, E_B, lam=0.0):
    return np.minimum(E_A, E_B) + lam * np.maximum(E_A, E_B)


# number of positions at which two peptides differ
def hamming(s1, s2):
    return int(np.sum(s1 != s2))


# whether every step of the path is a single substitution
def is_valid_path(path, allow_stay=False):
    for t in range(len(path) - 1):
        d = hamming(path[t], path[t + 1])
        if d > 1 or (d == 0 and not allow_stay):
            return False
    return True


# the joint score of each peptide along a path
def path_scores(path, E_A_fn, E_B_fn, score_fn=score_both):
    path_arr = np.stack(path)
    return np.asarray(score_fn(E_A_fn(path_arr), E_B_fn(path_arr)), dtype=float)


# the total score of a path, summed over its nodes
def path_energy(path, E_A_fn, E_B_fn, score_fn=score_both):
    return float(path_scores(path, E_A_fn, E_B_fn, score_fn).sum())


def choose_endpoints_transition(allele_A, allele_B, df_ms, E_A_fn, E_B_fn,
                                top_k=10, seed=None):
    '''Take the furthest apart pair among the top binders of each allele.

    Returns (v0, vT, T), the two endpoints and their Hamming distance.
    '''
    seqs_A = df_ms[df_ms['MHC allele'] == allele_A]['Peptide'].tolist()
    seqs_B = df_ms[df_ms['MHC allele'] == allele_B]['Peptide'].tolist()
    if len(seqs_A) == 0:
        raise ValueError(f"No IEDB peptides found for {allele_A}")
    if len(seqs_B) == 0:
        raise ValueError(f"No IEDB peptides found for {allele_B}")

    enc_A = convert_number(seqs_A)
    enc_B = convert_number(seqs_B)

    top_A = enc_A[np.argsort(E_A_fn(enc_A))[:top_k]]
    top_B = enc_B[np.argsort(E_B_fn(enc_B))[:top_k]]

    best_d, best_v0, best_vT = 0, None, None
    for i in range(len(top_A)):
        for j in range(len(top_B)):
            d = hamming(top_A[i], top_B[j])
            if d > best_d:
                best_d, best_v0, best_vT = d, top_A[i], top_B[j]

    if best_v0 is None or best_d == 0:
        raise ValueError(
            f"All top-{top_k} candidates for {allele_A} and {allele_B} are "
            f"identical. Try increasing top_k."
        )

    T = best_d
    e_v0 = float(E_A_fn(best_v0.reshape(1, -1))[0])
    e_vT = float(E_B_fn(best_vT.reshape(1, -1))[0])
    print(f"[transition] v_0 ({allele_A}): {convert_letter(best_v0)[0]}  E_A={e_v0:.2f}")
    print(f"[transition] v_T ({allele_B}): {convert_letter(best_vT)[0]}  E_B={e_vT:.2f}")
    print(f"Hamming distance (path length T): {T}")

    return best_v0, best_vT, T

def choose_endpoints_escape(allele_A, df_ms, E_A_fn, top_k=10,
                            escaped_pool=None, seed=None):
    '''Take endpoints for one allele, from a presented v_0 to an escaped v_T.

    v_T is drawn from escaped_pool when given, otherwise from the allele's own
    worst scoring ligands. Returns (v0, vT, T), the two endpoints and their
    Hamming distance.
    '''
    seqs_A = df_ms[df_ms['MHC allele'] == allele_A]['Peptide'].tolist()
    if len(seqs_A) == 0:
        raise ValueError(f"No IEDB peptides found for {allele_A}")
    enc_A = convert_number(seqs_A)
    E_A_vals = E_A_fn(enc_A)

    cand_0 = enc_A[np.argsort(E_A_vals)[:top_k]]

    if escaped_pool is None:
        cand_T = enc_A[np.argsort(E_A_vals)[-top_k:]]
    else:
        if len(escaped_pool) == 0:
            raise ValueError("escaped_pool is empty")
        enc_esc = convert_number(list(escaped_pool))
        E_esc = E_A_fn(enc_esc)
        cand_T = enc_esc[np.argsort(E_esc)[-top_k:]]

    best_d, best_v0, best_vT = 0, None, None
    for i in range(len(cand_0)):
        for j in range(len(cand_T)):
            d = hamming(cand_0[i], cand_T[j])
            if d > best_d:
                best_d, best_v0, best_vT = d, cand_0[i], cand_T[j]

    if best_v0 is None or best_d < 2:  # the sampler needs an interior node
        raise ValueError(
            f"No candidate pair with Hamming distance >= 2 for {allele_A}."
        )

    T = best_d
    e_v0 = float(E_A_fn(best_v0.reshape(1, -1))[0])
    e_vT = float(E_A_fn(best_vT.reshape(1, -1))[0])
    print(f"[escape:{allele_A}] v_0 (presented): {convert_letter(best_v0)[0]}  E_A={e_v0:.2f}")
    print(f"[escape:{allele_A}] v_T (escaped):   {convert_letter(best_vT)[0]}  E_A={e_vT:.2f}")
    print(f"Hamming distance (path length T): {T}")

    return best_v0, best_vT, T

def choose_endpoints_specific(allele_A, allele_B, df_ms, E_A_fn, E_B_fn,
                              top_k=10, seed=None):
    '''Take endpoints that score far better under their own allele than the other.

    Returns (v0, vT, T), the two endpoints and their Hamming distance.
    '''
    seqs_A = df_ms[df_ms['MHC allele'] == allele_A]['Peptide'].tolist()
    seqs_B = df_ms[df_ms['MHC allele'] == allele_B]['Peptide'].tolist()
    if len(seqs_A) == 0:
        raise ValueError(f"No IEDB peptides found for {allele_A}")
    if len(seqs_B) == 0:
        raise ValueError(f"No IEDB peptides found for {allele_B}")
    enc_A = convert_number(seqs_A)
    enc_B = convert_number(seqs_B)

    spec_A = E_A_fn(enc_A) - E_B_fn(enc_A)
    spec_B = E_B_fn(enc_B) - E_A_fn(enc_B)
    top_A = enc_A[np.argsort(spec_A)[:top_k]]
    top_B = enc_B[np.argsort(spec_B)[:top_k]]

    best_d, best_v0, best_vT = 0, None, None
    for i in range(len(top_A)):
        for j in range(len(top_B)):
            d = hamming(top_A[i], top_B[j])
            if d > best_d:
                best_d, best_v0, best_vT = d, top_A[i], top_B[j]

    if best_v0 is None or best_d < 2:
        raise ValueError(
            f"No A-specific/B-specific pair with Hamming >= 2 for {allele_A} / "
            f"{allele_B}. Try a larger top_k."
        )

    T = best_d
    e0A = float(E_A_fn(best_v0.reshape(1, -1))[0])
    e0B = float(E_B_fn(best_v0.reshape(1, -1))[0])
    eTA = float(E_A_fn(best_vT.reshape(1, -1))[0])
    eTB = float(E_B_fn(best_vT.reshape(1, -1))[0])
    print(f"[specific] v_0 ({allele_A}): {convert_letter(best_v0)[0]}  E_A={e0A:.2f} E_B={e0B:.2f}")
    print(f"[specific] v_T ({allele_B}): {convert_letter(best_vT)[0]}  E_A={eTA:.2f} E_B={eTB:.2f}")
    print(f"Hamming distance (path length T): {T}")

    return best_v0, best_vT, T


def _pwm_from_encoded(enc, pseudo=0.5):
    '''Smoothed frequency matrix over positions and residues.'''
    counts = np.full((L, Q_PEP), pseudo, dtype=float)
    for i in range(L):
        idx, c = np.unique(enc[:, i], return_counts=True)
        counts[i, idx] += c
    return counts / counts.sum(axis=1, keepdims=True)


def choose_endpoints_motif(allele_A, allele_B, df_ms, E_A_fn=None, E_B_fn=None,
                           top_k=10, seed=None):
    '''Take endpoints carrying each allele's own anchors.

    Returns (v0, vT, T), the two endpoints and their Hamming distance.
    '''
    seqs_A = df_ms[df_ms['MHC allele'] == allele_A]['Peptide'].tolist()
    seqs_B = df_ms[df_ms['MHC allele'] == allele_B]['Peptide'].tolist()
    if len(seqs_A) == 0:
        raise ValueError(f"No IEDB peptides found for {allele_A}")
    if len(seqs_B) == 0:
        raise ValueError(f"No IEDB peptides found for {allele_B}")
    enc_A = convert_number(seqs_A)
    enc_B = convert_number(seqs_B)

    log_A = np.log(_pwm_from_encoded(enc_A))
    log_B = np.log(_pwm_from_encoded(enc_B))

    def anchor_typicality(enc, log_own):
        d = np.zeros(len(enc))
        for pos in ANCHOR_POSITIONS:
            d += log_own[pos, enc[:, pos]]
        return d

    motif_A = anchor_typicality(enc_A, log_A)
    motif_B = anchor_typicality(enc_B, log_B)
    top_A = enc_A[np.argsort(motif_A)[::-1][:top_k]]
    top_B = enc_B[np.argsort(motif_B)[::-1][:top_k]]

    best_d, best_v0, best_vT = 0, None, None
    for i in range(len(top_A)):
        for j in range(len(top_B)):
            d = hamming(top_A[i], top_B[j])
            if d > best_d:
                best_d, best_v0, best_vT = d, top_A[i], top_B[j]

    if best_v0 is None or best_d < 2:
        raise ValueError(
            f"No A-motif/B-motif pair with Hamming >= 2 for {allele_A} / "
            f"{allele_B}. Try a larger top_k."
        )

    T = best_d
    seq0, seqT = convert_letter(best_v0)[0], convert_letter(best_vT)[0]
    anch0 = ''.join(seq0[p] for p in ANCHOR_POSITIONS)
    anchT = ''.join(seqT[p] for p in ANCHOR_POSITIONS)
    print(f"[motif] v_0 ({allele_A}): {seq0}  anchors={anch0}")
    print(f"[motif] v_T ({allele_B}): {seqT}  anchors={anchT}")
    if E_A_fn is not None and E_B_fn is not None:
        e0A = float(E_A_fn(best_v0.reshape(1, -1))[0])
        e0B = float(E_B_fn(best_v0.reshape(1, -1))[0])
        eTA = float(E_A_fn(best_vT.reshape(1, -1))[0])
        eTB = float(E_B_fn(best_vT.reshape(1, -1))[0])
        print(f"[motif] E(v_0): E_A={e0A:.2f} E_B={e0B:.2f}   "
              f"E(v_T): E_A={eTA:.2f} E_B={eTB:.2f}")
    print(f"Hamming distance (path length T): {T}")

    return best_v0, best_vT, T


# a starting direct path: make the substitutions in a random order
def init_path_direct(s0, sT, rng=None):
    rng = _as_rng(rng)
    diff_pos = np.where(s0 != sT)[0]
    diff_pos = rng.permutation(diff_pos)
    path = [s0.copy()]
    cur = s0.copy()
    for pos in diff_pos:
        nxt = cur.copy()
        nxt[pos] = sT[pos]
        path.append(nxt)
        cur = nxt
    return path


# a starting path of T steps, padded out by repeating nodes. The duplicates
# are steps that change nothing, which the moves then open out into spurs.
def init_path_indirect(s0, sT, T, rng=None):
    rng = _as_rng(rng)
    D = hamming(s0, sT)
    if T < D:
        raise ValueError(f"T ({T}) must be >= hamming(s0, sT) = {D}.")
    path = init_path_direct(s0, sT, rng=rng)  # length D+1
    for _ in range(T - D):
        i = int(rng.integers(1, len(path)))  # upper bound keeps sT at the end
        path.insert(i, path[i - 1].copy())
    return path


def biased_step(path, scores, E_A_fn, E_B_fn, score_fn, beta, rng=None):
    '''One update of a random interior node on a direct path.

    Returns (path, scores, changed), where scores is the node score cache
    updated in place.
    '''
    rng = _as_rng(rng)
    T = len(path) - 1
    t = int(rng.integers(1, T))
    s_prev, s_cur, s_next = path[t - 1], path[t], path[t + 1]
    diff = np.where(s_prev != s_next)[0]
    if len(diff) != 2:
        raise ValueError(
            f"biased_step expects a direct path, got {len(diff)} differing positions."
        )

    # the neighbours differ at two positions and the node sits one step from
    # each, so it either makes p0 first or p1 first. That is the whole choice,
    # which is why this is a Gibbs update on a single bit.
    p0, p1 = diff[0], diff[1]
    c0 = s_prev.copy(); c0[p0] = s_next[p0]
    c1 = s_prev.copy(); c1[p1] = s_next[p1]

    # one of the two is where the node already is, so its score is cached
    if np.array_equal(c0, s_cur):  # only c1 needs scoring
        sc0 = scores[t]
        sc1 = float(score_fn(E_A_fn(c1[None]), E_B_fn(c1[None]))[0])
    else:
        sc1 = scores[t]
        sc0 = float(score_fn(E_A_fn(c0[None]), E_B_fn(c0[None]))[0])

    x = beta * (sc0 - sc1)
    pick_c1 = np.log(rng.random()) < -np.logaddexp(0.0, -x)  # true with probability sigma(x)
    chosen, chosen_sc = (c1, sc1) if pick_c1 else (c0, sc0)

    if np.array_equal(chosen, s_cur):
        return path, scores, False
    scores[t] = chosen_sc
    new_path = list(path)
    new_path[t] = chosen
    return new_path, scores, True


def _local_candidates(a, b):
    '''Every sequence within one substitution of both neighbours.'''
    # how much freedom the node has depends on how far apart its neighbours
    # are, and on a valid path they are never more than two substitutions apart
    diff = np.where(a != b)[0]
    d = len(diff)
    if d == 0:
        # the neighbours agree, so the node may sit on them or one step off in
        # any direction, which is where a spur comes from
        cands = [a.copy()]
        for i in range(L):
            for z in range(Q_PEP):
                if z != a[i]:
                    c = a.copy(); c[i] = z; cands.append(c)
    elif d == 1:
        # one position is in play and any residue there keeps the path valid
        i = diff[0]
        cands = []
        for z in range(Q_PEP):
            c = a.copy(); c[i] = z; cands.append(c)
    elif d == 2:
        # both positions must change over the two steps, so the node only
        # chooses which of them goes first
        i, j = diff
        c0 = a.copy(); c0[i] = b[i]
        c1 = a.copy(); c1[j] = b[j]
        cands = [c0, c1]
    else:
        raise ValueError(
            f"neighbours differ at {d} positions (>2): not a valid path."
        )
    return np.stack(cands)


def local_step(path, E_A_fn, E_B_fn, score_fn, beta, Lambda, rng=None):
    '''One update of a random interior node, drawn from its local candidates.'''
    rng = _as_rng(rng)
    T = len(path) - 1
    t = int(rng.integers(1, T))
    a, b, cur = path[t - 1], path[t + 1], path[t]

    # score every candidate and draw one in proportion to its weight. The
    # candidate set is the node's full conditional, so this always accepts.
    cands = _local_candidates(a, b)
    sc = np.asarray(score_fn(E_A_fn(cands), E_B_fn(cands)), dtype=float)
    # a candidate off the straight line pays Lambda for each extra mutation
    elastic = (cands != a).sum(axis=1) + (cands != b).sum(axis=1)
    logw = -beta * (sc + Lambda * elastic)
    logw -= logw.max()  # subtract the largest before exponentiating
    p = np.exp(logw); p /= p.sum()

    chosen = cands[int(rng.choice(len(cands), p=p))]
    if np.array_equal(chosen, cur):
        return path, False
    new_path = list(path)
    new_path[t] = chosen
    return new_path, True


def _site_logsumexp(a, i, beta, E_A_fn, E_B_fn, score_fn):
    '''Log-normaliser over the residues at one site.'''
    cands = np.repeat(a[None, :], Q_PEP, axis=0)
    cands[:, i] = np.arange(Q_PEP)
    sc = np.asarray(score_fn(E_A_fn(cands), E_B_fn(cands)), dtype=float)
    logw = -beta * sc
    m = logw.max()
    return m + np.log(np.exp(logw - m).sum()), cands, logw


def mauri_step(path, E_A_fn, E_B_fn, score_fn, beta, Lambda, rng=None):
    '''One update of a random interior node with Mauri's proposal.

    Unlike biased_step this move can lengthen or shorten a spur, so Lambda,
    the elastic cost per mutation, enters the acceptance ratio.
    Returns (path, changed).
    '''
    rng = _as_rng(rng)
    T = len(path) - 1
    t = int(rng.integers(1, T))
    a, b, cur = path[t - 1], path[t + 1], path[t]

    diff = np.where(a != b)[0]
    d = len(diff)
    if d > 2:
        raise ValueError(f"neighbours differ at {d} positions (>2): not a valid path.")

    def _draw(logw, cands):
        w = np.exp(logw - logw.max())
        return cands[int(rng.choice(len(cands), p=w / w.sum()))]

    # with the neighbours two apart both candidates are one step from each, so
    # the elastic term is the same either way and cancels: a plain Gibbs draw
    if d == 2:
        cands = _local_candidates(a, b)
        sc = np.asarray(score_fn(E_A_fn(cands), E_B_fn(cands)), dtype=float)
        trial = _draw(-beta * sc, cands)
        if np.array_equal(trial, cur):
            return path, False
        new_path = list(path)
        new_path[t] = trial
        return new_path, True

    # otherwise the proposal is not the conditional, so the move needs a
    # Metropolis-Hastings ratio. Only the elastic term survives in log_alpha
    # here, since the score already shaped the draw.
    if d == 1:
        i = int(diff[0])
        _, cands, logw = _site_logsumexp(a, i, beta, E_A_fn, E_B_fn, score_fn)
        trial = _draw(logw, cands)
        if np.array_equal(trial, cur):
            return path, False
        log_alpha = -beta * Lambda * (
            int((trial != a).sum()) + int((trial != b).sum())
            - int((cur != a).sum()) - int((cur != b).sum()))
    # the neighbours agree, so first pick a site at random and then a residue
    # there. A proposal can now be reached by more than one route, so the
    # Hastings ratio also carries the normalisers of the two proposals.
    else:
        i = int(rng.integers(L))
        logZ_i, cands, logw = _site_logsumexp(a, i, beta, E_A_fn, E_B_fn, score_fn)
        trial = _draw(logw, cands)
        if np.array_equal(trial, cur):
            return path, False

        # sitting on the neighbour is reachable from every site, so its
        # normaliser is a sum over all L of them and costs more to work out
        cur_at_a, trial_at_a = np.array_equal(cur, a), np.array_equal(trial, a)
        if cur_at_a or trial_at_a:
            all_logZ = np.array([_site_logsumexp(a, k, beta, E_A_fn, E_B_fn, score_fn)[0]
                                 for k in range(L)])
            m = (-all_logZ).max()
            log_c_a = m + np.log(np.exp(-all_logZ - m).sum())
        else:
            log_c_a = None

        def _log_c(v, logZ_known=None):
            if np.array_equal(v, a):
                return log_c_a
            if logZ_known is not None:
                return -logZ_known
            k = int(np.where(v != a)[0][0])
            return -_site_logsumexp(a, k, beta, E_A_fn, E_B_fn, score_fn)[0]

        log_alpha = -beta * Lambda * (
            int((trial != a).sum()) + int((trial != b).sum())
            - int((cur != a).sum()) - int((cur != b).sum()))
        log_alpha += _log_c(cur) - _log_c(trial, logZ_known=logZ_i)

    if np.log(rng.random()) < log_alpha:
        new_path = list(path)
        new_path[t] = trial
        return new_path, True
    return path, False


def run_sampler(s0, sT, E_A_fn, E_B_fn=None,
                beta=1.0, beta_init=None, anneal='linear',
                score='both', weights=(1.0, 1.0), lam=0.0,
                T=None, Lambda=0.0, proposal='local',
                n_burn=10000, n_iter=100000, n_thin=100,
                seed=None):
    '''Run the chain and return the sampled paths, their energies and metadata.

    Args:
        beta_init (float | None): Inverse temperature the burn in anneals
            from, no anneal when None.
        anneal (str): Ramp shape, 'linear' or 'geometric'.
        score (str): Joint score, 'both', 'either' or 'lambda'.
        lam (float): Regime knob for the 'lambda' score, not an energy cost.
        T (int | None): Path length in steps, the Hamming distance when None.
            A larger T leaves interior nodes free, making the path indirect.
        Lambda (float): Elastic cost per mutation, which only bites on an
            indirect path.
        proposal (str): Interior node move, 'local' or 'mauri'.

    Returns:
        (samples, energies, accept_rate, info), where samples holds one list
        of (L,) int arrays per path and info records the hyperparameters.
    '''
    D = hamming(s0, sT)
    if D == 0:
        raise ValueError("s0 and sT are identical.")
    if T is None:
        T = D
    T = int(T)
    if T < D:
        raise ValueError(f"T ({T}) must be >= hamming(s0, sT) = {D}.")
    if T < 2:
        raise ValueError(
            "T < 2 leaves no interior node to resample."
        )
    indirect = T > D

    if proposal not in ('local', 'mauri'):
        raise ValueError(
            f"proposal must be 'local' or 'mauri', got {proposal!r}")

    if E_B_fn is None:
        E_B_fn = lambda seqs: np.zeros(np.atleast_2d(seqs).shape[0])
        score_fn = lambda a, b: a
        score = 'single'
    elif score == 'both':
        if len(weights) != 2:
            raise ValueError(f"weights must be (w_A, w_B), got {weights!r}")
        w_A, w_B = weights
        score_fn = lambda a, b: score_both(a, b, w_A, w_B)
    elif score == 'either':
        score_fn = score_either
    elif score == 'lambda':
        score_fn = lambda a, b: score_lambda(a, b, lam)
    else:
        raise ValueError(f"score must be 'both', 'either' or 'lambda', got {score!r}")

    if beta_init is None:
        burn_betas = np.full(n_burn, beta)
    elif anneal == 'linear':
        burn_betas = np.linspace(beta_init, beta, n_burn)
    elif anneal == 'geometric':
        if beta_init <= 0:
            raise ValueError("geometric annealing requires beta_init > 0")
        burn_betas = np.geomspace(beta_init, beta, n_burn)
    else:
        raise ValueError(f"anneal must be 'linear' or 'geometric', got {anneal!r}")

    rng = _as_rng(seed)

    samples, energies = [], []
    n_accepted = 0  # how many moves changed the path
    if not indirect:
        path = init_path_direct(s0, sT, rng=rng)
        scores = path_scores(path, E_A_fn, E_B_fn, score_fn)
        for b in burn_betas:
            path, scores, _ = biased_step(path, scores, E_A_fn, E_B_fn, score_fn, b, rng=rng)
        for i in range(n_iter):
            path, scores, acc = biased_step(path, scores, E_A_fn, E_B_fn, score_fn, beta, rng=rng)
            n_accepted += acc
            if (i + 1) % n_thin == 0:
                samples.append([s.copy() for s in path])
                energies.append(float(scores.sum()))
    else:
        step_fn = {'mauri': mauri_step}.get(proposal, local_step)
        path = init_path_indirect(s0, sT, T, rng=rng)
        for b in burn_betas:
            path, _ = step_fn(path, E_A_fn, E_B_fn, score_fn, b, Lambda, rng=rng)
        for i in range(n_iter):
            path, acc = step_fn(path, E_A_fn, E_B_fn, score_fn, beta, Lambda, rng=rng)
            n_accepted += acc
            if (i + 1) % n_thin == 0:
                samples.append([s.copy() for s in path])
                energies.append(path_energy(path, E_A_fn, E_B_fn, score_fn))

    accept_rate = n_accepted / n_iter if n_iter > 0 else 0.0

    info = {
        'beta': float(beta),
        'beta_init': None if beta_init is None else float(beta_init),
        'anneal': anneal if beta_init is not None else None,
        'score': score,
        'weights': [float(weights[0]), float(weights[1])],
        'proposal': proposal if indirect else 'biased',
        'path_type': 'indirect' if indirect else 'direct',
        'n_burn': int(n_burn),
        'n_iter': int(n_iter),
        'n_thin': int(n_thin),
        'seed': seed,
        'accept_rate': float(accept_rate),
        'T': int(T),
        'D': int(D),
        'Lambda': float(Lambda) if indirect else None,
        's0': convert_letter(s0)[0],
        'sT': convert_letter(sT)[0],
    }
    return samples, np.array(energies), accept_rate, info


def save_paths(samples, energies, allele_A, allele_B, label,
               info=None, outdir=PATHS_DIR):
    '''Write the paths, energies and metadata to outdir.

    Writes paths_, energies_ and meta_ under the given label, plus a lens_
    file when the paths differ in length and cannot be stacked.
    '''
    os.makedirs(outdir, exist_ok=True)
    lens = [len(p) for p in samples]
    ragged = len(set(lens)) > 1
    lens_file = os.path.join(outdir, f"lens_{label}.npy")
    if not samples:
        paths_arr = np.zeros((0, 0, L), dtype=int)
    elif ragged:
        paths_arr = np.concatenate([np.stack(p) for p in samples], axis=0)
        np.save(lens_file, np.array(lens))
    else:
        paths_arr = np.stack([np.stack(p) for p in samples])
    if not ragged and os.path.exists(lens_file):
        os.remove(lens_file)  # a stale lengths file would mis-split this one
    np.save(os.path.join(outdir, f"paths_{label}.npy"), paths_arr)
    np.save(os.path.join(outdir, f"energies_{label}.npy"), energies)

    meta = {
        'label': label,
        'allele_A': allele_A,
        'allele_B': allele_B,
        'n_samples': len(samples),
        'path_length': (None if ragged else (len(samples[0]) if samples else 0)),
        'ragged': bool(ragged),
    }
    if info is not None:
        meta.update(info)

    with open(os.path.join(outdir, f"meta_{label}.json"), 'w') as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == '__main__':
    from pwm import get_pwm_for_allele, pwm_energy_batch

    allele_A = 'HLA-A*02:01'
    allele_B = 'HLA-B*07:02'

    pwm_A, _ = get_pwm_for_allele(allele_A)
    pwm_B, _ = get_pwm_for_allele(allele_B)

    E_A_fn = lambda seqs: pwm_energy_batch(seqs, pwm_A)
    E_B_fn = lambda seqs: pwm_energy_batch(seqs, pwm_B)

    s0 = np.array([AA_DICT[a] for a in 'GILGFVFTL'])
    sT = np.array([AA_DICT[a] for a in 'RPHERNFTV'])

    print(f"Hamming distance s0->sT: {hamming(s0, sT)}")

    init = init_path_direct(s0, sT, rng=0)
    print(f"Initial path length: {len(init)}, valid: {is_valid_path(init)}")
