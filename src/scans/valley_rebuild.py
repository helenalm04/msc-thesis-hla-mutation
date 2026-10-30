# Redraw the valley ensemble with far longer chains, to check the valley in
# Chapter 6 is not a burn-in artefact, then recompute that one column of the
# corridor cache. The previous version is kept alongside as _prevalley.npz.

import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import FIGURES_DIR
import mean_field_corridor as mfc
import sampler as smp

CACHE = os.path.join(FIGURES_DIR, '_indirect_pca_paths.npz')
OUT = os.path.join(FIGURES_DIR, '_mean_field_corridor.npz')
PATHS_OUT = os.path.join(FIGURES_DIR, '_valley_resampled_paths.npz')

SRC_A, TGT_A, LABEL = 'HLA-B*51:01', 'HLA-B*27:07', 'valley'
N_CHAINS, N_BURN, N_ITER, N_THIN = 12, 40_000, 200_000, 1_000


def resample(v0, vT, T, land):
    '''Draw a fresh valley ensemble.'''
    EA = lambda V: land.parts[0].energy(V) - land.offsets[0]
    EB = lambda V: land.parts[1].energy(V) - land.offsets[1]
    # independent chains, pooled, so the ensemble does not rest on one seed
    out = []
    for c in range(N_CHAINS):
        samples, _, acc, _ = smp.run_sampler(
            v0.astype(np.int16), vT.astype(np.int16), EA, EB,
            beta=mfc.BETA, beta_init=0.1, anneal='linear', score='either', T=T,
            Lambda=mfc.LAM, proposal='local', n_burn=N_BURN, n_iter=N_ITER,
            n_thin=N_THIN, seed=2000 + c)
        out.append(np.array([np.stack(p) for p in samples], dtype=np.int64))
        print(f'    chain {c:2d}: {len(samples)} paths, accept {acc:.3f}', flush=True)
    return np.concatenate(out, axis=0)


def main():
    cache = np.load(CACHE, allow_pickle=True)
    key = mfc.syskey(SRC_A, TGT_A)
    v0 = cache[f'{key}__v0'].astype(np.int64)
    vT = cache[f'{key}__vT'].astype(np.int64)
    offA = float(cache[f'{key}__offA'][0])
    offB = float(cache[f'{key}__offB'][0])
    old = cache[f'{key}__indirect'].astype(np.int64)
    T = old.shape[1] - 1

    land = mfc.mf.EitherLandscape(mfc.load_rbm(SRC_A), mfc.load_rbm(TGT_A),
                                  kappa=mfc.KAPPA, offsets=(offA, offB))

    print(f'resampling {LABEL}: {N_CHAINS} chains, {N_BURN} burn, {N_ITER} iter',
          flush=True)
    paths = resample(v0, vT, T, land)
    print(f'  {paths.shape[0]} paths against {old.shape[0]} before', flush=True)
    np.savez_compressed(PATHS_OUT, indirect=paths, v0=v0, vT=vT,
                        settings=np.array([N_CHAINS, N_BURN, N_ITER, N_THIN]))

    patched = {f'{key}__v0': v0, f'{key}__vT': vT,
               f'{key}__offA': np.array([offA]), f'{key}__offB': np.array([offB]),
               f'{key}__indirect': paths}
    results = {}
    mfc.run_system(SRC_A, TGT_A, LABEL, patched, results)

    # keep the old cache before writing over the valley entries
    shutil.copy(OUT, OUT.replace('.npz', '_prevalley.npz'))
    store = dict(np.load(OUT, allow_pickle=True))
    changed = [k for k in results if k in store]
    store.update(results)  # the other two columns stay as they are
    np.savez_compressed(OUT, **store)
    print(f'\n  replaced {len(changed)} valley entries in {os.path.basename(OUT)}',
          flush=True)
    print(f'  previous version kept as {os.path.basename(OUT).replace(".npz", "_prevalley.npz")}',
          flush=True)
    print('\nnow run:  python -u src/figures/plot_mean_field.py', flush=True)


if __name__ == '__main__':
    main()
