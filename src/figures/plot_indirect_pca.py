# Chapter 6, the direct and indirect paths in the plane of the two
# repertoires. Three systems, chosen for their outcomes: a corridor, a valley,
# and a same-supertype pair with no barrier to detour around. Each is sampled
# twice and projected onto a PCA of the two alleles' real ligands.
# Writes the cache the other two indirect figures read, so it runs first.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, FIGURES_DIR, L, Q_PEP
from b51_either_sweep import (natural_candidates, ligand_offset,
                              load_rbm, rbm_energy_batch)
from sampler import run_sampler
from encoding import convert_number

SYSTEMS = [  # source, target, source anchors, target anchors, outcome
    ('HLA-B*51:01', 'HLA-B*27:05', {'P', 'A'}, {'R'}, 'corridor'),
    ('HLA-B*51:01', 'HLA-B*27:07', {'P', 'A'}, {'R'}, 'valley'),
    ('HLA-A*02:01', 'HLA-A*02:04', {'L', 'M'}, {'L', 'M'}, 'no barrier'),
]
TARGETS = [t for _, t, _, _, _ in SYSTEMS]


# identifier for one source and target allele pair
def syskey(src, tgt):
    clean = lambda a: a.replace('*', '').replace(':', '')
    return f'{clean(src)}__{clean(tgt)}'
K, LAM = 6, 0.35
SEEDS = [1, 2]
N_SHOW = 30  # paths drawn for each class in a panel
N_CLOUD = 450  # ligands drawn for each allele
COMMON = dict(score='either', beta=1.0, beta_init=0.1, anneal='linear',
              n_burn=3000, n_iter=12000, n_thin=300)

C_DIRECT = '#5C82D6'
C_INDIRECT = '#EA9542'
C_SRC = '#7FAF6B'
C_TGT = '#B07AA1'
CACHE = os.path.join(FIGURES_DIR, '_indirect_pca_paths.npz')


def onehot(enc):
    enc = np.atleast_2d(np.asarray(enc, dtype=int))
    oh = np.zeros((len(enc), L, Q_PEP))
    oh[np.arange(len(enc))[:, None], np.arange(L)[None, :], enc] = 1.0
    return oh.reshape(len(enc), L * Q_PEP)


# for each node of a path, the number of positions holding a residue from neither endpoint
def d_direct_space(path, v0, vT):
    endset = [(int(a), int(b)) for a, b in zip(v0, vT)]
    return np.array([sum(int(node[i]) not in endset[i] for i in range(L))
                     for node in path])


# integer encoded ligands of length nine measured for one allele
def repertoire(df, allele):
    return convert_number([s for s in df[df['MHC allele'] == allele]['Peptide']
                           if len(s) == 9])


# the input each sequence sends to every hidden unit of an rbm
def hid(rbm, seqs):
    return rbm.input_hiddens(np.atleast_2d(np.asarray(seqs)).astype(np.int16))


# distance from mu to each sequence in hidden input space, in standard deviations per unit
def centroid_distance(seqs, rbm, mu, sd):
    return np.linalg.norm((hid(rbm, np.stack(seqs)) - mu) / sd, axis=1)


# Endpoints have to be real ligands rather than the single best binder, which
# tends to sit out on its own. This takes the best candidate that is still
# inside the bulk of the repertoire and far enough from the other endpoint.
def pick_endpoint(cands, rbm, repertoire, exclude=None, min_d=5, bulk_pct=90):
    '''Pick the best-scoring candidate that sits inside the repertoire bulk.'''
    from sampler import hamming
    R = hid(rbm, repertoire)
    mu, sd = R.mean(0), R.std(0) + 1e-9
    cut = np.percentile(np.linalg.norm((R - mu) / sd, axis=1), bulk_pct)
    dc = centroid_distance(cands, rbm, mu, sd)
    for i in np.argsort(dc > cut, kind='stable'):  # in-bulk first, in energy order
        if dc[i] > cut:
            break
        c = cands[int(i)]
        if exclude is None:
            return c, None
        d = hamming(exclude, c)
        if d >= min_d:
            return c, int(d)
    raise ValueError('no in-bulk candidate far enough from v0')


def sample(v0, vT, D, cA, cB, indirect):
    '''Pool the sampled paths over seeds as (n_paths, T+1, L).'''
    out = []
    for seed in SEEDS:
        kw = dict(T=D + K, Lambda=LAM) if indirect else dict(T=None)
        samples, *_ = run_sampler(v0, vT, cA, cB, seed=seed, **kw, **COMMON)
        out.extend(np.stack(p) for p in samples)
    return np.stack(out)


def build():
    '''Sample or load the direct and indirect paths for every system.'''
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f"loaded {CACHE}", flush=True)
        return {k: z[k] for k in z.files}

    df = pd.read_csv(IEDB_MS_CSV)
    store, v0_by_source = {}, {}
    for src, tgt, anchA, anchB, _ in SYSTEMS:
        rbmA = load_rbm(src)
        EA = lambda s_, m=rbmA: rbm_energy_batch(np.atleast_2d(s_), m)
        offA = ligand_offset(src, df, EA)
        cA = lambda s_, o=offA, e=EA: e(s_) - o
        rbmB = load_rbm(tgt)
        EB = lambda s_, m=rbmB: rbm_energy_batch(np.atleast_2d(s_), m)
        offB = ligand_offset(tgt, df, EB)
        cB = lambda s_, o=offB, e=EB: e(s_) - o

        # systems with the same source reuse v_0
        repA = repertoire(df, src)
        if src not in v0_by_source:
            candA = natural_candidates(src, df, EA, anchA)
            v0_by_source[src] = pick_endpoint(candA, rbmA, repA)[0]
        v0 = v0_by_source[src]
        repB = repertoire(df, tgt)
        candB = natural_candidates(tgt, df, EB, anchB)
        vT, D = pick_endpoint(candB, rbmB, repB, exclude=v0)

        key = syskey(src, tgt)
        print(f"{src} -> {tgt}: D={D}, T={D + K}", flush=True)
        store[f'{key}__direct'] = sample(v0, vT, D, cA, cB, indirect=False)
        store[f'{key}__indirect'] = sample(v0, vT, D, cA, cB, indirect=True)
        store[f'{key}__v0'] = np.asarray(v0)
        store[f'{key}__vT'] = np.asarray(vT)
        store[f'{key}__offA'] = np.array([offA])
        store[f'{key}__offB'] = np.array([offB])
    np.savez(CACHE, **store)
    print(f"sampled + cached {CACHE}", flush=True)
    return store


def main():
    store = build()
    df = pd.read_csv(IEDB_MS_CSV)
    for src, tgt, _, _, outcome in SYSTEMS:
        key = syskey(src, tgt)
        v0, vT = store[f'{key}__v0'], store[f'{key}__vT']
        n = len(store[f'{key}__direct'])
        print(f"  {src} -> {tgt:12s} ({outcome:10s}) D={int((v0 != vT).sum())} "
              f"{n} direct + {len(store[f'{key}__indirect'])} indirect paths")


if __name__ == '__main__':
    main()
