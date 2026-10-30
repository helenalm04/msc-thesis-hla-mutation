# Chapter 6, the mean-field solution for the three indirect systems. Solver
# and chain are projected onto the same two hidden units, so both answers can
# be drawn in one plane. allocate_alphabet spends a fixed budget of interior
# sequences on the residues the sampler actually visited, which is what keeps
# the exact program affordable. Reads plot_indirect_pca.py's cache.

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import FIGURES_DIR, L, AA
import mean_field as mf
import exact_path_dp as edp
import RBM_utils

CACHE = os.path.join(FIGURES_DIR, '_indirect_pca_paths.npz')
OUT = os.path.join(FIGURES_DIR, '_mean_field_corridor.npz')

SYSTEMS = [  # same order as the cache keys
    ('HLA-B*51:01', 'HLA-B*27:05', 'corridor'),
    ('HLA-B*51:01', 'HLA-B*27:07', 'valley'),
    ('HLA-A*02:01', 'HLA-A*02:04', 'no barrier'),
]
BETA, LAM = 1.0, 0.35
KAPPA = 8.0
BUDGET = 1_500_000  # cap on interior sequences at each step


def syskey(src, tgt):
    clean = lambda a: a.replace('*', '').replace(':', '')
    return f'{clean(src)}__{clean(tgt)}'


def load_rbm(allele):
    '''Load the saved RBM for one allele.'''
    tag = allele.replace('*', '').replace(':', '')
    hits = glob.glob(os.path.join(ROOT, 'data', 'models', f'model_{tag}_*.data'))
    if not hits:
        raise FileNotFoundError(f'no pretrained RBM for {allele}')
    return RBM_utils.loadRBM(hits[0])


def pick_unit(rbm, own, other):
    '''Pick the hidden unit that best separates the two repertoires, and its sign.'''
    Io = rbm.input_hiddens(np.atleast_2d(own).astype(np.int16))
    Ix = rbm.input_hiddens(np.atleast_2d(other).astype(np.int16))
    d = (Io.mean(0) - Ix.mean(0)) / np.sqrt(0.5 * (Io.var(0) + Ix.var(0)) + 1e-12)
    mu = int(np.argmax(np.abs(d)))
    return mu, float(np.sign(d[mu]))


def mf_hidden_traj(land, res, mu_A, sgn_A, mu_B, sgn_B):
    '''Mean-field path in the hidden-unit plane.'''
    (gA, iA0, iA1), (gB, iB0, iB1) = land.slices
    X = np.einsum('tia,kia->tk', res['marg'], land.features)
    return np.stack([sgn_A * X[:, iA0 + mu_A], sgn_B * X[:, iB0 + mu_B]], axis=1)


def mf_hidden_traj_sd(land, res, mu_A, mu_B):
    '''Spread of the same two coordinates under the mean-field measure.'''
    (gA, iA0, iA1), (gB, iB0, iB1) = land.slices
    marg, feat = res['marg'], land.features
    m1 = np.einsum('tia,kia->tik', marg, feat)
    m2 = np.einsum('tia,kia->tik', marg, feat ** 2)
    var = (m2 - m1 ** 2).sum(axis=1)  # shape (T-1, K)
    return np.sqrt(np.stack([var[:, iA0 + mu_A], var[:, iB0 + mu_B]], axis=1))


def ensemble_hidden_traj(paths, rbm_A, rbm_B, mu_A, sgn_A, mu_B, sgn_B):
    '''Mean and spread of the same coordinates over the sampled paths at each step.'''
    n, steps, _ = paths.shape
    flat = paths.reshape(-1, paths.shape[-1]).astype(np.int16)
    xa = sgn_A * rbm_A.input_hiddens(flat)[:, mu_A].reshape(n, steps)
    xb = sgn_B * rbm_B.input_hiddens(flat)[:, mu_B].reshape(n, steps)
    xy = np.stack([xa, xb], axis=-1)
    return xy.mean(axis=0), xy.std(axis=0)


def sampler_marg(ind, q):
    '''Residue frequencies of the sampled paths at each step and site.'''
    n, Tp1, L = ind.shape
    marg = np.zeros((Tp1 - 2, L, q))
    for t in range(1, Tp1 - 1):
        for i in range(L):
            cnt = np.bincount(ind[:, t, i], minlength=q).astype(float)
            marg[t - 1, i] = cnt / cnt.sum()
    return marg


def allocate_alphabet(v0, vT, ind, marg_comb, q, budget):
    '''Alphabet for each site, ranked by `marg_comb` and sized under a state cap.'''
    L = len(v0)
    interior = ind[:, 1:-1, :]
    rank, cum = [], []
    for i in range(L):
        cnt = np.bincount(interior[:, :, i].ravel(), minlength=q).astype(float)
        f = cnt / cnt.sum()
        s = np.maximum(f, marg_comb[:, i, :].max(axis=0))
        forced = sorted({int(v0[i]), int(vT[i])})
        rest = [int(a) for a in np.argsort(-s) if int(a) not in forced]
        r = forced + rest
        rank.append(r)
        cum.append([f[r[:k]].sum() for k in range(1, q + 1)])
    cum = np.array(cum)
    ks = np.array([len(sorted({int(v0[i]), int(vT[i])})) for i in range(L)])
    seqcov = lambda k: float(np.prod([cum[i][k[i] - 1] for i in range(L)]))
    while True:
        best = None
        for i in range(L):
            if ks[i] >= q:
                continue
            new = ks.copy()
            new[i] += 1
            if np.prod(new) > budget:
                continue
            gain = np.log(max(seqcov(new), 1e-12)) - np.log(max(seqcov(ks), 1e-12))
            if gain <= 0:
                continue
            score = gain / np.log(new[i] / ks[i])
            if best is None or score > best[0]:
                best = (score, i)
        if best is None:
            break
        ks[best[1]] += 1
    alph = [np.array(sorted(rank[i][:ks[i]]), dtype=np.int64) for i in range(L)]
    return alph, seqcov(ks)


# substitutions per step, averaged over the paths
def d_step_ensemble(paths):
    return (paths[:, 1:, :] != paths[:, :-1, :]).sum(axis=2).mean(axis=0)


# sites carrying a residue from neither endpoint, for every node of every path
def d_direct_per_path(paths, v0, vT):
    ends = [{int(v0[i]), int(vT[i])} for i in range(len(v0))]
    return np.array([[sum(int(node[i]) not in ends[i] for i in range(len(v0)))
                      for node in path] for path in paths])


def run_system(src, tgt, label, cache, results):
    '''Solve, check and record one system.'''
    key = syskey(src, tgt)
    v0 = cache[f'{key}__v0'].astype(np.int64)
    vT = cache[f'{key}__vT'].astype(np.int64)
    offA = float(cache[f'{key}__offA'][0])
    offB = float(cache[f'{key}__offB'][0])
    ind = cache[f'{key}__indirect'].astype(np.int64)
    T = ind.shape[1] - 1
    D = int((v0 != vT).sum())

    rbm_A, rbm_B = load_rbm(src), load_rbm(tgt)
    land = mf.EitherLandscape(rbm_A, rbm_B, kappa=KAPPA, offsets=(offA, offB))

    print(f'\n{src} -> {tgt}   ({label})')
    print(f'    D = {D}, T = {T}, beta = {BETA}, Lambda = {LAM}, kappa = {KAPPA}')
    print(f'    v0 = {"".join(AA[a] for a in v0)}   vT = {"".join(AA[a] for a in vT)}')

    res = mf.solve(land, v0, vT, T, BETA, LAM, elastic='cont', K=1.0)
    dDS = mf.d_direct_subspace(res['marg'], v0, vT)
    pA = land.allele_weights(res['marg'])
    print(f'    solve: {res["n_iter"]} iters, converged = {res["converged"]}, '
          f'ln Z_MF = {res["logZ"]:.4f}, f_path = {res["f_path"]:.4f}')
    print(f'    d_DS  = {np.round(dDS, 2)}')
    print(f'    p_A   = {np.round(pA, 2)}')

    print('\n    Evo (bound holds).  alphabet  n_states  exact(hard)  exact(soft)'
          '  softening   MF gap')
    marg_comb = np.maximum(res['marg'], sampler_marg(ind, land.q))
    exact_rows = []
    for n_extra in (0, 1, 2):
        alph = edp.build_alphabets(v0, vT, n_extra=n_extra, marg=marg_comb)
        n_states = int(np.prod([len(a) for a in alph]))
        if n_states > 400000:
            print(f'      +{n_extra}  skipped ({n_states} states)')
            continue
        ex_h = edp.solve_exact(lambda V: land.energy(V, hard=True),
                               v0, vT, T, BETA, LAM, alph, wall=False, q=land.q)
        ex_s = edp.solve_exact(lambda V: land.energy(V, hard=False),
                               v0, vT, T, BETA, LAM, alph, wall=False, q=land.q)
        r_evo = mf.solve(land, v0, vT, T, BETA, LAM, alphabets=alph, elastic='evo')
        soften = ex_s['logZ'] - ex_h['logZ']
        gap = ex_s['logZ'] - r_evo['logZ']
        flag = '' if gap >= -1e-6 else '  BOUND VIOLATED'
        print(f'      +{n_extra}   {n_states:9d}  {ex_h["logZ"]:11.4f}  '
              f'{ex_s["logZ"]:11.4f}  {soften:+9.4f}  {gap:+7.4f}{flag}')
        exact_rows.append((n_extra, n_states, ex_h['logZ'], ex_s['logZ'],
                           r_evo['logZ'], gap))

    alph, seqcov = allocate_alphabet(v0, vT, ind, marg_comb, land.q, BUDGET)
    print(f'\n    alphabet: sizes {[len(a) for a in alph]}, '
          f'{int(np.prod([len(a) for a in alph]))} states, '
          f'whole-sequence coverage of the sampled nodes {seqcov:.4f}')
    ex = edp.solve_exact(lambda V: land.energy(V, hard=True),
                         v0, vT, T, BETA, LAM, alph, wall=True, q=land.q)
    r_res = mf.solve(land, v0, vT, T, BETA, LAM, alphabets=alph,
                     elastic='cont', K=1.0)
    dDS_exact = mf.d_direct_subspace(ex['marg'], v0, vT)
    dDS_mf = mf.d_direct_subspace(r_res['marg'], v0, vT)
    _per_path = d_direct_per_path(ind, v0, vT)[:, 1:-1]
    dDS_mcmc = _per_path.mean(axis=0)
    dDS_mcmc_sd = _per_path.std(axis=0, ddof=1)
    dDS_mcmc_se = dDS_mcmc_sd / np.sqrt(len(_per_path))  # takes the paths as independent
    d_mcmc = d_step_ensemble(ind)
    print(f'\n    Cont (the sampler\'s kernel), on the +2 alphabet '
          f'({int(np.prod([len(a) for a in alph]))} states):')
    print(f'      ln Z:  MF {r_res["logZ"]:.4f}   exact {ex["logZ"]:.4f}   '
          f'gap {ex["logZ"] - r_res["logZ"]:+.4f}')
    print(f'      <d_t>  MF      = {np.round(r_res["d_traj"], 3)}')
    print(f'      <d_t>  exact   = {np.round(ex["d_traj"], 3)}')
    print(f'      <d_t>  sampler = {np.round(d_mcmc, 3)}')
    print(f'      d_DS   MF      = {np.round(dDS_mf, 2)}')
    print(f'      d_DS   exact   = {np.round(dDS_exact, 2)}')
    print(f'      d_DS   sampler = {np.round(dDS_mcmc, 2)}')
    print(f'      total variation of marginals, MF vs exact: '
          f'{0.5 * np.abs(r_res["marg"] - ex["marg"]).sum(-1).mean():.4f}')

    rep_A = _repertoire(src)
    rep_B = _repertoire(tgt)
    mu_A, sgn_A = pick_unit(rbm_A, rep_A, rep_B)
    mu_B, sgn_B = pick_unit(rbm_B, rep_B, rep_A)
    xy_mf = mf_hidden_traj(land, res, mu_A, sgn_A, mu_B, sgn_B)
    xy_mc, xy_sd = ensemble_hidden_traj(ind, rbm_A, rbm_B, mu_A, sgn_A, mu_B, sgn_B)
    print(f'    hidden units: A unit {mu_A} (sign {sgn_A:+.0f}), '
          f'B unit {mu_B} (sign {sgn_B:+.0f})')

    results[f'{key}__xy_mf'] = xy_mf
    results[f'{key}__xy_mf_sd'] = mf_hidden_traj_sd(land, res, mu_A, mu_B)
    results[f'{key}__xy_mcmc'] = xy_mc
    results[f'{key}__xy_mcmc_sd'] = xy_sd
    results[f'{key}__n_paths'] = np.array([len(ind)])  # for standard errors in the figure
    results[f'{key}__dDS_mf'] = dDS
    results[f'{key}__dDS_mf_restricted'] = dDS_mf
    results[f'{key}__dDS_exact'] = dDS_exact
    results[f'{key}__dDS_mcmc'] = dDS_mcmc
    results[f'{key}__dDS_mcmc_sd'] = dDS_mcmc_sd
    results[f'{key}__dDS_mcmc_se'] = dDS_mcmc_se
    results[f'{key}__dDS_mf_sd'] = mf.d_direct_subspace_sd(res['marg'], v0, vT)
    results[f'{key}__d_mf'] = r_res['d_traj']
    results[f'{key}__d_exact'] = ex['d_traj']
    results[f'{key}__d_mcmc'] = d_mcmc
    results[f'{key}__pA'] = pA
    results[f'{key}__E_mf'] = res['E_traj']
    results[f'{key}__exact_rows'] = np.array(exact_rows, dtype=float)
    results[f'{key}__meta'] = np.array([D, T, BETA, LAM, KAPPA, res['logZ']])


def _repertoire(allele):
    '''IEDB 9-mer ligands for one allele, integer encoded.'''
    import pandas as pd
    from config import IEDB_MS_CSV
    from encoding import convert_number
    df = pd.read_csv(IEDB_MS_CSV)
    seqs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == L]
    return convert_number(seqs)


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(f'missing {CACHE}; run src/figures/plot_indirect_pca.py first')
    cache = np.load(CACHE, allow_pickle=True)
    results = {}
    for src, tgt, label in SYSTEMS:
        run_system(src, tgt, label, cache, results)
    np.savez_compressed(OUT, **results)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
