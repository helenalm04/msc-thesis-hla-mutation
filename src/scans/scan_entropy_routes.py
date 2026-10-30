# Chapter 4, how many different routes a transition really has. Route entropy
# counts the distinct orders the substitutions are made in, set against the
# allele's own repertoire entropy. The plug-in entropy is biased low on a
# small sample, so compute_exact solves the same ensembles exactly and
# compares the two directly.

import argparse
import importlib.util
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, ENTROPY_CSV, RESULTS_DIR, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK
from entropy import convert_number
from sampler import run_sampler, hamming
from path_stats import route_entropy
from exact_paths import exact_direct_ensemble

_r = importlib.util.spec_from_file_location("src_rbm", os.path.join(SRC, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_r); _r.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

N_ALLELES = 24
TARGET_D = 6
SEEDS = [1, 2, 3]
BETA = 1.5
COMMON = dict(beta_init=0.15, anneal='linear', n_burn=2500, n_iter=25000, n_thin=50)
CACHE = os.path.join(RESULTS_DIR, 'entropy_routes.csv')
CACHE_MULTI = os.path.join(RESULTS_DIR, 'entropy_routes_multi.csv')
CACHE_EXACT = os.path.join(RESULTS_DIR, 'entropy_routes_exact.csv')
LOCUS_C = {'A': '#5C82D6', 'B': '#EA9542', 'C': '#4CA64C'}


# true when an encoded peptide uses at least k distinct residues
def _complex(enc, k=6):
    return len(set(int(x) for x in enc)) >= k


def select_alleles(n=None):
    '''Alleles spaced evenly across the entropy value range.'''
    n = N_ALLELES if n is None else n
    d = pd.read_csv(ENTROPY_CSV).dropna(subset=['E_corr_rbm'])
    targets = np.linspace(d.E_corr_rbm.min(), d.E_corr_rbm.max(), n)
    used, rows = set(), []
    for t in targets:
        cand = d[~d['HLA allele'].isin(used)]
        i = (cand.E_corr_rbm - t).abs().idxmin()
        used.add(cand.loc[i, 'HLA allele'])
        rows.append(cand.loc[i])
    return pd.DataFrame(rows).reset_index(drop=True)


def pick_endpoints(allele, df, E_fn, target_d=TARGET_D, top_frac=0.25):
    '''The two good binders closest to target_d in Hamming distance.'''
    seqs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == 9]
    enc = convert_number(seqs)
    e = np.asarray(E_fn(enc), dtype=float)
    order = np.argsort(e)
    cut = max(int(len(order) * top_frac), 40)
    top = [i for i in order[:cut] if _complex(enc[i])][:30]
    best = None
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            d = hamming(enc[top[a]], enc[top[b]])
            if d < 3:
                continue
            score = abs(d - target_d)
            if best is None or score < best[0]:
                best = (score, enc[top[a]], enc[top[b]], d)
    return best[1], best[2], best[3]


def pick_endpoints_multi(allele, df, E_fn, n_pairs, target_d=TARGET_D, top_frac=0.25):
    '''Disjoint pairs of good binders, each about target_d apart.'''
    seqs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == 9]
    enc = convert_number(seqs)
    e = np.asarray(E_fn(enc), dtype=float)
    order = np.argsort(e)
    cut = max(int(len(order) * top_frac), 60)
    top = [i for i in order[:cut] if _complex(enc[i])][:60]
    rank = {idx: r for r, idx in enumerate(top)}  # 0 is the best binder, for the tie-break
    cands = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            ia, ib = top[a], top[b]
            d = hamming(enc[ia], enc[ib])
            if d < 3:
                continue
            cands.append((abs(d - target_d), rank[ia] + rank[ib], ia, ib, d))
    cands.sort(key=lambda c: (c[0], c[1]))  # sort on closeness to target_d, then on binding quality
    used, pairs = set(), []
    for _, _, ia, ib, d in cands:
        if ia in used or ib in used:
            continue
        used.add(ia); used.add(ib)
        pairs.append((enc[ia], enc[ib], d))
        if len(pairs) >= n_pairs:
            break
    return pairs


def _sample_pair(v0, vT, E_fn):
    '''Route entropy of one path ensemble pooled over the seeds.'''
    pooled = []
    for seed in SEEDS:
        samples, _, _, _ = run_sampler(v0, vT, E_fn, None, beta=BETA, seed=seed, **COMMON)
        pooled.extend(np.stack(p) for p in samples)
    r = route_entropy(pooled)
    r['n_paths'] = len(pooled)
    return r


def compute(pairs_per_allele):
    '''Sample the route ensembles into the cache.'''
    df = pd.read_csv(IEDB_MS_CSV)
    sel = select_alleles()
    out = CACHE if pairs_per_allele == 1 else CACHE_MULTI
    if os.path.exists(out):
        rows = pd.read_csv(out).to_dict('records')
        done = {r['allele'] for r in rows}
    else:
        rows, done = [], set()

    def _flush():
        pd.DataFrame(rows).to_csv(out, index=False)

    for _, a in sel.iterrows():
        allele = a['HLA allele']
        if allele in done:
            print(f"{allele:14s} (cached, skip)", flush=True)
            continue
        S = float(a['E_corr_rbm'])
        loc = allele.split('-')[1][0]
        rbm = load_rbm(allele)
        E_rbm = lambda s, m=rbm: rbm_energy_batch(np.atleast_2d(s), m)

        if pairs_per_allele == 1:
            v0, vT, D = pick_endpoints(allele, df, E_rbm)
            per_seed_norm = []
            pooled = []
            for seed in SEEDS:
                samples, _, _, _ = run_sampler(v0, vT, E_rbm, None, beta=BETA, seed=seed, **COMMON)
                paths = [np.stack(p) for p in samples]
                pooled.extend(paths)
                per_seed_norm.append(route_entropy(paths)['S_path_norm'])
            r = route_entropy(pooled)
            rows.append(dict(allele=allele, locus=loc, entropy=S, D=D,
                             S_path_norm=r['S_path_norm'], n_eff=r['n_eff'],
                             n_distinct=r['n_distinct'], norm_sd=float(np.std(per_seed_norm)),
                             n_paths=len(pooled)))
            print(f"{allele:14s} S={S:5.2f} D={D} S_norm={r['S_path_norm']:.3f} "
                  f"n_eff={r['n_eff']:5.1f} n_dist={r['n_distinct']:3d}", flush=True)
        else:
            eps = pick_endpoints_multi(allele, df, E_rbm, pairs_per_allele)
            norms = []
            for j, (v0, vT, D) in enumerate(eps):
                r = _sample_pair(v0, vT, E_rbm)
                norms.append(r['S_path_norm'])
                rows.append(dict(allele=allele, locus=loc, entropy=S, pair=j, D=D,
                                 S_path_norm=r['S_path_norm'], n_eff=r['n_eff'],
                                 n_distinct=r['n_distinct'], n_paths=r['n_paths']))
            print(f"{allele:14s} S={S:5.2f} n_pairs={len(eps)} "
                  f"S_norm_mean={np.mean(norms):.3f} between-pair_sd={np.std(norms):.3f}",
                  flush=True)

        _flush()

    print(f"\nCached -> {out}")


def compute_exact(pairs_per_allele):
    '''Solve each allele's route ensemble exactly.'''
    df = pd.read_csv(IEDB_MS_CSV)
    sel = select_alleles()
    rows = []
    for _, a in sel.iterrows():
        allele = a['HLA allele']
        S = float(a['E_corr_rbm'])
        loc = allele.split('-')[1][0]
        rbm = load_rbm(allele)
        E_rbm = lambda s, m=rbm: rbm_energy_batch(np.atleast_2d(s), m)

        if pairs_per_allele == 1:
            v0, vT, D = pick_endpoints(allele, df, E_rbm)
            eps = [(v0, vT, D)]
        else:
            eps = pick_endpoints_multi(allele, df, E_rbm, pairs_per_allele)

        norms = []
        for j, (v0, vT, D) in enumerate(eps):
            r = exact_direct_ensemble(v0, vT, E_rbm, None, beta=BETA)
            norms.append(r['S_path_norm'])
            e = r['node_energy']
            delta = np.array([e[1 << i] - e[0] for i in range(r['D'])])  # one mismatch substituted at a time
            rows.append(dict(allele=allele, locus=loc, entropy=S, pair=j, D=r['D'],
                             S_path=r['S_path'], S_path_norm=r['S_path_norm'],
                             n_eff=r['n_eff'], p_map=float(np.exp(r['map_logprob'])),
                             n_routes=math.factorial(r['D']),
                             delta_sd=float(delta.std()), delta_mean=float(delta.mean()),
                             corridor_epistasis=float(e[-1] - e[0] - delta.sum())))
        print(f"{allele:14s} S={S:5.2f} n_pairs={len(eps):2d} "
              f"S_norm_mean={np.mean(norms):.3f} sd={np.std(norms):.3f}", flush=True)

    pd.DataFrame(rows).to_csv(CACHE_EXACT, index=False)
    print(f"\nCached -> {CACHE_EXACT}")


def _aggregate_exact():
    '''Mean exact diversity and between-pair standard error for each allele.'''
    t = pd.read_csv(CACHE_EXACT)
    g = t.groupby('allele')
    agg = g.agg(entropy=('entropy', 'first'), locus=('locus', 'first'),
                D=('D', 'mean'), n_pairs=('pair', 'nunique'),
                S_path_norm=('S_path_norm', 'mean'),
                norm_sd=('S_path_norm', 'std'),
                n_eff=('n_eff', 'mean'), p_map=('p_map', 'mean')).reset_index()
    agg['norm_sd'] = agg['norm_sd'].fillna(0.0)
    agg['norm_se'] = agg['norm_sd'] / np.sqrt(agg['n_pairs'].clip(lower=1))
    return agg.sort_values('entropy').reset_index(drop=True)


def compare_exact_vs_plugin():
    '''Exact minus plug-in for each allele on the same single-pair ensembles.'''
    if not (os.path.exists(CACHE) and os.path.exists(CACHE_EXACT)):
        return None
    ex = pd.read_csv(CACHE_EXACT)
    if ex['pair'].nunique() > 1:
        return None  # the multi-pair exact runs use a different ensemble
    s = pd.read_csv(CACHE)[['allele', 'D', 'S_path_norm', 'n_eff', 'n_distinct']]
    m = s.merge(ex[['allele', 'D', 'S_path_norm', 'n_eff']], on='allele',
                suffixes=('_plug', '_exact'))
    m = m[m.D_plug == m.D_exact]
    m['d_norm'] = m.S_path_norm_exact - m.S_path_norm_plug
    m['ratio_neff'] = m.n_eff_exact / m.n_eff_plug
    return m


def _partial_spearman(x, y, z):
    '''Partial Spearman of x and y controlling for z.'''
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([rz, np.ones_like(rz)]).T
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return spearmanr(ex, ey).correlation


def _perm_p(x, y, rho, n=10000):
    rng = np.random.default_rng(0)
    null = np.array([spearmanr(rng.permutation(x), y).correlation for _ in range(n)])
    return float((np.abs(null) >= abs(rho)).mean())


def _aggregate_multi():
    '''Collapse the pair rows of CACHE_MULTI to one row for each allele.'''
    t = pd.read_csv(CACHE_MULTI)
    g = t.groupby('allele')
    agg = g.agg(entropy=('entropy', 'first'),
                locus=('locus', 'first'),
                D=('D', 'mean'),
                n_pairs=('pair', 'nunique'),
                S_path_norm=('S_path_norm', 'mean'),
                norm_sd=('S_path_norm', 'std'),
                n_eff=('n_eff', 'mean')).reset_index()
    agg['norm_se'] = agg['norm_sd'] / np.sqrt(agg['n_pairs'].clip(lower=1))
    return agg.sort_values('entropy').reset_index(drop=True)


def plot(pairs_per_allele, exact=False):
    if exact:
        t = _aggregate_exact()
        err_col = 'norm_se'
        n_p = int(t.n_pairs.max())
        subtitle = ('exact enumeration, 1 endpoint pair/allele' if n_p == 1 else
                    f'exact enumeration, {n_p} endpoint pairs/allele (between-pair SE)')
    elif pairs_per_allele == 1:
        t = pd.read_csv(CACHE)
        err_col = 'norm_sd'
        subtitle = '1 endpoint pair/allele (per-seed error bars)'
    else:
        t = _aggregate_multi()
        err_col = 'norm_se'
        subtitle = f'{int(t.n_pairs.max())} endpoint pairs/allele (between-pair SE)'

    rho = spearmanr(t.entropy, t.S_path_norm).correlation
    p = _perm_p(t.entropy.values, t.S_path_norm.values, rho)
    rho_neff = spearmanr(t.entropy, t.n_eff).correlation
    pr_neff = _partial_spearman(t.entropy.values, t.n_eff.values, t.D.values)
    print(f"\nentropy -> routes (n={len(t)} alleles, {subtitle})")
    print(f"S_path_norm (D-normalised):  Spearman rho={rho:+.2f}  perm p={p:.3f}")
    print(f"n_eff:                       Spearman rho={rho_neff:+.2f}  partial|D rho={pr_neff:+.2f}")
    print(f"D range: {t.D.min():.1f}-{t.D.max():.1f}   entropy range: {t.entropy.min():.2f}-{t.entropy.max():.2f}")

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for loc, c in LOCUS_C.items():
        s = t[t.locus == loc]
        ax.errorbar(s.entropy, s.S_path_norm, yerr=s[err_col], fmt='o', ms=7,
                    color=c, capsize=2, lw=0, elinewidth=1, label=f'HLA-{loc}')
    ax.set_xlabel(r'allele repertoire entropy  $S$  (nats)', fontsize=FS_LABEL)
    ax.set_ylabel(r'route diversity  $S_{\mathrm{path}}/\log D!$', fontsize=FS_LABEL)
    ax.set_title(f'Entropy vs escape-route diversity   '
                 f'(Spearman $\\rho={rho:+.2f}$, $p={p:.2f}$)\n{subtitle}', fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, title='locus')
    plt.tight_layout()
    suffix = '_exact' if exact else ('' if pairs_per_allele == 1 else '_multi')
    out = os.path.join(FIGURES_DIR, f'entropy_routes{suffix}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs-per-allele', type=int, default=1)
    ap.add_argument('--plot', action='store_true', help='replot from cache, skip sampling')
    ap.add_argument('--exact', action='store_true',
                    help='solve each ensemble exactly (no MCMC, no plug-in estimator)')
    ap.add_argument('--n-alleles', type=int, default=N_ALLELES,
                    help='alleles to spread across the entropy range; 68 uses every one '
                         'that has both an RBM and an entropy value, which is the whole '
                         'set and the only honest way to raise the power of the '
                         'correlation. Non-default values write a suffixed cache so the '
                         '24-allele result is never overwritten.')
    args = ap.parse_args()
    if args.n_alleles != N_ALLELES:
        N_ALLELES = args.n_alleles
        CACHE_EXACT = CACHE_EXACT.replace('.csv', f'_n{N_ALLELES}.csv')
        CACHE_MULTI = CACHE_MULTI.replace('.csv', f'_n{N_ALLELES}.csv')
    if not args.plot:
        (compute_exact if args.exact else compute)(args.pairs_per_allele)
    plot(args.pairs_per_allele, exact=args.exact)

    if args.exact:
        m = compare_exact_vs_plugin()
        if m is not None and len(m):
            print(f"\nexact minus plug-in, same {len(m)} single-pair ensembles")
            print(f"{'allele':<14}{'D':>3}{'routes seen':>12}{'plug-in':>10}"
                  f"{'exact':>10}{'diff':>9}{'n_eff x':>9}")
            for _, r in m.sort_values('d_norm').iterrows():
                print(f"{r.allele:<14}{int(r.D_plug):>3}{int(r.n_distinct):>12}"
                      f"{r.S_path_norm_plug:>10.3f}{r.S_path_norm_exact:>10.3f}"
                      f"{r.d_norm:>+9.3f}{r.ratio_neff:>9.2f}")
            print(f"\nplug-in understates route diversity by "
                  f"{m.d_norm.mean():+.3f} on average "
                  f"(range {m.d_norm.min():+.3f} to {m.d_norm.max():+.3f}); "
                  f"n_eff by a factor {m.ratio_neff.mean():.2f}")
            print(f"the shortfall grows with the number of routes visited: Spearman "
                  f"rho={spearmanr(m.n_distinct, m.d_norm).correlation:+.2f}")
