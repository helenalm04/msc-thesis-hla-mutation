# Appendix check that the shape of the d_DS curve belongs to the landscape and
# not to the temperature. Each system is solved at betas two orders of
# magnitude apart and the curves correlated against beta = 1.

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
OUT = os.path.join(FIGURES_DIR, '_mf_landscape_free.npz')
BETAS = [1.0, 0.5, 0.25, 0.05, 0.005]

SYSTEMS = [('HLA-B*51:01', 'HLA-B*27:05', 'corridor'),
           ('HLA-B*51:01', 'HLA-B*27:07', 'valley'),
           ('HLA-A*02:01', 'HLA-A*02:04', 'no barrier')]


def main():
    '''Solve each system at every beta and save the d_DS curves.'''
    cache = np.load(CACHE, allow_pickle=True)
    ref = np.load(os.path.join(FIGURES_DIR, '_mean_field_corridor.npz'),
                  allow_pickle=True)
    store = {}
    for src, tgt, label in SYSTEMS:
        key = mfc.syskey(src, tgt)
        v0 = cache[f'{key}__v0'].astype(np.int64)
        vT = cache[f'{key}__vT'].astype(np.int64)
        offA = float(cache[f'{key}__offA'][0])
        offB = float(cache[f'{key}__offB'][0])
        T = cache[f'{key}__indirect'].shape[1] - 1
        land = mf.EitherLandscape(mfc.load_rbm(src), mfc.load_rbm(tgt),
                                  kappa=mfc.KAPPA, offsets=(offA, offB))

        truth = ref[f'{key}__dDS_exact']
        print(f'\n{label}   T = {T}', flush=True)
        print(f'  exact (beta = 1)        peak {truth.max():.3f} at step '
              f'{int(truth.argmax()) + 1}', flush=True)

        base = None
        for b in BETAS:
            # 'cont' retunes Lambda at each beta, so the step size stays at one
            # mutation and only the temperature changes between rows
            res = mf.solve(land, v0, vT, T, b, mfc.LAM, elastic='cont', K=1.0)
            dds = mf.d_direct_subspace(res['marg'], v0, vT)
            if base is None:
                base = dds  # the beta = 1 baseline for the diff column
            corr = float(np.corrcoef(base, dds)[0, 1])
            print(f'  mean field, beta = {b:<6}  peak {dds.max():.3f} at step '
                  f'{int(dds.argmax()) + 1}   max|diff from beta=1| '
                  f'{np.abs(dds - base).max():.3f}   shape corr {corr:.4f}',
                  flush=True)
            store[f'{key}__b{b}'] = dds
        store[f'{key}__exact'] = truth
    np.savez_compressed(OUT, **store)
    print(f'\nwrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
