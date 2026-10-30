# Appendix check that the mean-field path is not an artefact of averaging. A
# mean over paths can sit somewhere no single path goes, so the solver's most
# likely path is compared against the most common sampled one.

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
from mean_field_corridor import (syskey, load_rbm, BETA, LAM, KAPPA)

CACHE = os.path.join(FIGURES_DIR, '_indirect_pca_paths.npz')
SYSTEMS = [('HLA-B*51:01', 'HLA-B*27:05', 'corridor'),
           ('HLA-A*02:01', 'HLA-A*02:04', 'no barrier')]


# per step substitution counts along the argmax path, from the mean field marginals and from the modal sampled path
def main():
    cache = np.load(CACHE, allow_pickle=True)
    for src, tgt, label in SYSTEMS:
        key = syskey(src, tgt)
        v0 = cache[f'{key}__v0'].astype(np.int64)
        vT = cache[f'{key}__vT'].astype(np.int64)
        offA = float(cache[f'{key}__offA'][0])
        offB = float(cache[f'{key}__offB'][0])
        ind = cache[f'{key}__indirect'].astype(np.int64)
        T = ind.shape[1] - 1

        land = mf.EitherLandscape(load_rbm(src), load_rbm(tgt), kappa=KAPPA,
                                  offsets=(offA, offB))
        res = mf.solve(land, v0, vT, T, BETA, LAM, elastic='cont', K=1.0)

        nodes = np.concatenate([v0[None, :],
                                res['marg'].argmax(axis=2),
                                vT[None, :]], axis=0)
        d = (nodes[1:] != nodes[:-1]).sum(axis=1)

        # the same on the sampled paths
        counts = np.zeros((T + 1, len(v0), land.q))
        for t in range(T + 1):
            for i in range(len(v0)):
                counts[t, i] = np.bincount(ind[:, t, i], minlength=land.q)
        mc_nodes = counts.argmax(axis=2)
        d_mc = (mc_nodes[1:] != mc_nodes[:-1]).sum(axis=1)

        print(f'\n{label}   T = {T}, one substitution per step allowed')
        print(f'  mean field, per-step Hamming of the argmax reconstruction: {list(d)}')
        print(f'     illegal steps (>1): {int((d > 1).sum())} of {T}, '
              f'largest {int(d.max())}, total substitutions {int(d.sum())} '
              f'against a path length of {T}')
        print(f'  sampler modal path, same reconstruction:                   {list(d_mc)}')
        print(f'     illegal steps (>1): {int((d_mc > 1).sum())} of {T}, '
              f'largest {int(d_mc.max())}')
        print(f'  mean-field <d_t> from the solve itself: '
              f'{np.round(res["d_traj"], 3).tolist()}')


if __name__ == '__main__':
    main()
