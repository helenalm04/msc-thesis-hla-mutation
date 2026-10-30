# Appendix check that the mean-field fixed point is the one the solver finds
# wherever it starts. Each system is solved from the default straight-line
# start, from the sampler's own marginals, and from two random perturbations.

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import FIGURES_DIR
import mean_field as mf
import mean_field_corridor as mfc

CACHE = os.path.join(FIGURES_DIR, '_indirect_pca_paths.npz')
VALLEY_PATHS = os.path.join(FIGURES_DIR, '_valley_resampled_paths.npz')
OUT = os.path.join(FIGURES_DIR, '_mf_fixed_point.npz')

SYSTEMS = [('HLA-B*51:01', 'HLA-B*27:07', 'valley'),
           ('HLA-B*51:01', 'HLA-B*27:05', 'corridor')]
ELASTICS = ['cont', 'evo']


def empirical_marg(paths, q):
    '''Residue frequencies at each interior step and position.'''
    n, Tp1, L = paths.shape
    marg = np.zeros((Tp1 - 2, L, q))
    for t in range(1, Tp1 - 1):
        for i in range(L):
            cnt = np.bincount(paths[:, t, i], minlength=q).astype(float)
            marg[t - 1, i] = cnt / cnt.sum()
    return marg


# one line per start: how far the path leaves the direct subspace, the free
# energy it settles at, and how far its projections moved from the default
def report(tag, r, v0, vT, I_default):
    d = mf.d_direct_subspace(r['marg'], v0, vT)
    dt = r['d_traj']
    print(f'  {tag:26s} d_DS {d.max():5.3f} @ {int(d.argmax()) + 1:2d}   '
          f'lnZ_MF {r["logZ"]:8.4f}   <d_t> {dt.min():.3f}-{dt.max():.3f}   '
          f'conv {str(r["converged"]):5s}   '
          f'|dI| {np.abs(r["I_traj"] - I_default).max():7.4f}', flush=True)
    return d


def main():
    cache = np.load(CACHE, allow_pickle=True)
    store = {}
    for src, tgt, label in SYSTEMS:
        key = mfc.syskey(src, tgt)
        v0 = cache[f'{key}__v0'].astype(np.int64)
        vT = cache[f'{key}__vT'].astype(np.int64)
        offA = float(cache[f'{key}__offA'][0])
        offB = float(cache[f'{key}__offB'][0])

        # the shared cache holds the old valley ensemble
        if label == 'valley' and os.path.exists(VALLEY_PATHS):
            ind = np.load(VALLEY_PATHS)['indirect'].astype(np.int64)
        else:
            ind = cache[f'{key}__indirect'].astype(np.int64)
        T = ind.shape[1] - 1

        land = mf.EitherLandscape(mfc.load_rbm(src), mfc.load_rbm(tgt),
                                  kappa=mfc.KAPPA, offsets=(offA, offB))
        # project the sampled marginals onto the patterns, giving a start that
        # already sits where the sampler says the path goes
        feats = land.features
        I_mc = np.einsum('tia,kia->tk', empirical_marg(ind, land.q), feats)

        for el in ELASTICS:
            kw = dict(elastic=el)
            if el == 'cont':
                kw['K'] = 1.0
            print(f'\n{label}   T = {T}, {ind.shape[0]} paths, Phi_{el}', flush=True)
            base = mf.solve(land, v0, vT, T, mfc.BETA, mfc.LAM, **kw)
            I_default = base['I_traj']
            store[f'{key}__{el}__default'] = report('default (direct)', base, v0, vT,
                                                    I_default)
            r = mf.solve(land, v0, vT, T, mfc.BETA, mfc.LAM, I_init=I_mc, **kw)
            store[f'{key}__{el}__sampler'] = report('from sampler marginals', r, v0, vT,
                                                    I_default)
            # perturb by one and then three standard deviations of the
            # default projections, so the starts are genuinely far apart
            scale = float(np.abs(I_default).std())
            for seed, mult in ((0, 1.0), (1, 3.0)):
                rng = np.random.default_rng(seed)
                I_pert = I_default + mult * scale * rng.normal(size=I_default.shape)
                r = mf.solve(land, v0, vT, T, mfc.BETA, mfc.LAM, I_init=I_pert, **kw)
                store[f'{key}__{el}__rand{seed}'] = report(
                    f'random start, {mult:.0f} sd', r, v0, vT, I_default)

    np.savez_compressed(OUT, **store)
    print(f'\nwrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
