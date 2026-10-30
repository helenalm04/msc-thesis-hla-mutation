# Chapter 6, how wide a route is rather than how high its barrier. At each
# edge the number of ways onward is measured as a log sum over the available
# moves, so a wide corridor and a narrow pass can be told apart even at equal
# cost. part1 contrasts crossing between binding modes against staying within
# one; part2 runs the same measure over pairs across the JSD range.

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_pwm_rbm_paths as C
from compare_pwm_rbm_paths import (sample_pooled, epistatic_profile, load_rbm,
                                   rbm_energy_batch, pwm_energy_batch)
from mode_crossing_barrier import topk_pairs, cluster_modes
from path_stats import _node_energies
from sampler import hamming, _local_candidates
from entropy import convert_number, convert_letter, pwm_train
from config import (IEDB_MS_CSV, JSD_CSV, ENTROPY_CSV, ALLELE_SUPERTYPES,
                    RESULTS_DIR, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_TICK)

C.COMMON = dict(beta_init=0.15, anneal='linear', n_burn=3000, n_iter=15000, n_thin=100)
BETA, LAMBDA = 1.0, 0.5
ALLELE = 'HLA-B*51:01'
K_PAIRS = 10  # same value as in mode_crossing_barrier
N_PAIRS_JSD = 8
CACHE = os.path.join(RESULTS_DIR, 'pilot_route_width.npz')
CSV = os.path.join(RESULTS_DIR, 'pilot_route_width.csv')
BLUE, ORANGE = '#5C82D6', '#EA9542'
zero = lambda s: np.zeros(np.atleast_2d(s).shape[0])


def log_kappa_edge(a, b, E_fn, beta=BETA, Lambda=LAMBDA):
    '''log kappa at one edge: log sum_c exp(-beta [E(c) + Lambda * dM(c)]).'''
    cands = _local_candidates(a, b)
    e = np.asarray(E_fn(cands), dtype=float)
    d_ab = hamming(a, b)
    dM = np.array([hamming(a, c) + hamming(c, b) - d_ab for c in cands], dtype=float)
    w = -beta * (e + Lambda * dM)
    m = w.max()
    return float(m + np.log(np.exp(w - m).sum()))


def width_profile(samples, E_fn, n_paths=120):
    '''Mean log-kappa profile across a set of equal-length routes.'''
    s = np.asarray(samples)
    if len(s) > n_paths:
        idx = np.linspace(0, len(s) - 1, n_paths).astype(int)
        s = s[idx]
    prof = np.array([[log_kappa_edge(p[i], p[i + 1], E_fn)
                      for i in range(len(p) - 1)] for p in s]).mean(axis=0)
    return prof - 0.5 * (prof[0] + prof[-1]), prof


def width_profile_node(samples, E_fn, n_paths=120):
    '''Mean log-nu profile at each node across a set of equal-length routes.'''
    s = np.asarray(samples)
    if len(s) > n_paths:
        idx = np.linspace(0, len(s) - 1, n_paths).astype(int)
        s = s[idx]
    cache = {}  # the same node turns up on many routes

    def log_nu(v):
        k = np.asarray(v, dtype=np.int8).tobytes()
        if k not in cache:
            cache[k] = log_kappa_edge(v, v, E_fn, Lambda=0.0)
        return cache[k]

    prof = np.array([[log_nu(v) for v in p] for p in s]).mean(axis=0)
    return prof - 0.5 * (prof[0] + prof[-1]), prof


def part1(k=K_PAIRS):
    '''Cross-mode vs within-mode log-kappa profiles on one allele.'''
    df = pd.read_csv(IEDB_MS_CSV)
    rbm = load_rbm(ALLELE)
    seqs = [s for s in df[df['MHC allele'] == ALLELE]['Peptide'] if len(s) == 9]
    enc = convert_number(seqs)
    pwm = pwm_train(enc)
    E_rbm = lambda s: rbm_energy_batch(np.atleast_2d(s), rbm)
    E_pwm = lambda s, p=pwm: pwm_energy_batch(np.atleast_2d(s), p)
    e_rbm = np.asarray(E_rbm(enc), float)
    e_pwm = np.asarray(E_pwm(enc), float)

    good, km, H = cluster_modes(rbm, enc, e_rbm)
    lab = km.labels_
    m0, m1 = good[lab == 0], good[lab == 1]
    print(f"{ALLELE}: {len(good)} good binders, mode sizes {np.bincount(lab)}")

    cross = topk_pairs([(m0, m1)], enc, e_rbm, e_pwm, k)
    within = topk_pairs([(m0, m0), (m1, m1)], enc, e_rbm, e_pwm, k)

    store, rows = {}, []
    for tag, pairs in (('cross', cross), ('within', within)):
        dips, aligns, profs, barrs = [], [], [], []
        dips_n, aligns_n, profs_n = [], [], []
        for ia, ib, d in pairs:
            v0, vT = enc[ia], enc[ib]
            paths = sample_pooled(v0, vT, E_pwm)
            prof, raw = width_profile(paths, E_rbm)
            epi = epistatic_profile(paths, E_rbm, E_pwm)  # (T+1,), one value for each node
            dip = float(-prof.min())
            # edge midpoints and node positions, both on [0, 1]
            x_edge = (np.arange(len(prof)) + 0.5) / len(prof)
            x_node = np.arange(len(epi)) / (len(epi) - 1)
            epi_at_edge = np.interp(x_edge, x_node, epi)
            align = float(spearmanr(-prof, epi_at_edge).statistic)
            prof_n, _ = width_profile_node(paths, E_rbm)
            dip_n = float(-prof_n.min())
            align_n = float(spearmanr(-prof_n, epi).statistic)  # already on the same grid as epi
            dips.append(dip); aligns.append(align)
            dips_n.append(dip_n); aligns_n.append(align_n)
            profs.append(prof); barrs.append(float(epi.max()))
            profs_n.append(prof_n)
            rows.append({'tag': tag, 'D': d, 'dip': dip, 'align': align,
                         'dip_node': dip_n, 'align_node': align_n,
                         'barrier': float(epi.max()),
                         'v0': ''.join(convert_letter(np.asarray(v0))),
                         'vT': ''.join(convert_letter(np.asarray(vT)))})
            print(f"  {tag}: D={d}  edge dip={dip:+.2f}  node dip={dip_n:+.2f} nats  "
                  f"barrier={float(epi.max()):+.2f}  "
                  f"align(rho)={align:+.2f} / node {align_n:+.2f}", flush=True)
        store[f'{tag}_dips'] = np.array(dips)
        store[f'{tag}_aligns'] = np.array(aligns)
        store[f'{tag}_barriers'] = np.array(barrs)
        store[f'{tag}_dips_node'] = np.array(dips_n)
        store[f'{tag}_aligns_node'] = np.array(aligns_n)
        med = int(np.argsort(dips)[len(dips) // 2])  # the profile drawn in the figure
        store[f'{tag}_prof'] = profs[med]
        store[f'{tag}_prof_x'] = (np.arange(len(profs[med])) + 0.5) / len(profs[med])
        med_n = int(np.argsort(dips_n)[len(dips_n) // 2])
        store[f'{tag}_prof_node'] = profs_n[med_n]
        store[f'{tag}_prof_node_x'] = np.arange(len(profs_n[med_n])) / (len(profs_n[med_n]) - 1)

    for probe, key in (('edge', 'dips'), ('node', 'dips_node')):
        c, w = store[f'cross_{key}'], store[f'within_{key}']
        U, p = mannwhitneyu(c, w, alternative='greater')
        print(f"\n  [{probe}-centred, n={len(c)}+{len(w)}]")
        print(f"  cross-mode dip  {c.mean():+.2f} +/- {c.std(ddof=1):.2f} nats")
        print(f"  within-mode dip {w.mean():+.2f} +/- {w.std(ddof=1):.2f} nats")
        print(f"  Mann-Whitney (cross > within): U={U:.0f}, p={p:.2g}")
        akey = 'aligns' if probe == 'edge' else 'aligns_node'
        for t in ('cross', 'within'):
            a = store[f'{t}_{akey}']
            print(f"  {t} align(rho) mean {a.mean():+.3f} +/- {a.std(ddof=1):.3f} "
                  f"(SE {a.std(ddof=1) / np.sqrt(a.size):.3f})")
        store[f'mw_p_{probe}'] = np.array([p])

    cd, wd = store['cross_dips'], store['within_dips']
    U, p = mannwhitneyu(cd, wd, alternative='greater')
    rho_all = spearmanr(np.concatenate([cd, wd]),
                        np.concatenate([store['cross_barriers'],
                                        store['within_barriers']]))
    print(f"  dip vs energetic barrier across all {len(cd) + len(wd)} corridors: "
          f"rho={rho_all.statistic:+.2f}, p={rho_all.pvalue:.2g}")
    store['mw_p'] = np.array([p])
    store['dip_vs_barrier_rho'] = np.array([rho_all.statistic])
    store['dip_vs_barrier_p'] = np.array([rho_all.pvalue])
    return store, pd.DataFrame(rows)


def _pair_table():
    jsd = pd.read_csv(JSD_CSV, index_col=0)
    ent = pd.read_csv(ENTROPY_CSV)
    sup = pd.read_csv(ALLELE_SUPERTYPES)
    # 'Unclassified' in Supertype_new falls back to Supertype_old
    smap = {a: (o if n == 'Unclassified' else n) for a, n, o
            in zip(sup['HLA allele'], sup['Supertype_new'].astype(str),
                   sup['Supertype_old'].astype(str))}
    emap = dict(zip(ent['HLA allele'], ent['E_emp_rbm']))
    nmap = dict(zip(ent['HLA allele'], ent['N']))

    rows = []
    for _, r in jsd.iterrows():
        a, b = r['HLA allele1'], r['HLA allele2']
        v = r['JSD_corr_rbm']
        if a not in emap or b not in emap or not np.isfinite(v):
            continue
        rows.append({'A': a, 'B': b, 'jsd': float(v),
                     'same_supertype': smap.get(a) == smap.get(b),
                     'S_min': min(emap[a], emap[b]),
                     'N_min': min(nmap[a], nmap[b])})
    return pd.DataFrame(rows)


def part2():
    '''Mean log kappa on each edge for N_PAIRS_JSD corridors spread across JSD.'''
    from b51_either_sweep import (ligand_offset, natural_candidates, pick_pairs,
                                  ANCH_A, ANCH_B)
    tab = _pair_table()
    cross = tab[~tab['same_supertype']].sort_values('jsd')
    if len(cross) < N_PAIRS_JSD:
        raise SystemExit(f"only {len(cross)} cross-supertype pairs available")
    take = np.linspace(0, len(cross) - 1, N_PAIRS_JSD).astype(int)
    chosen = cross.iloc[take]
    print(f"{len(tab)} pairs available, {len(cross)} cross-supertype; "
          f"JSD range {cross['jsd'].min():.3f}-{cross['jsd'].max():.3f}")

    df_ms = pd.read_csv(IEDB_MS_CSV)
    rows = []
    for _, r in chosen.iterrows():
        A, B = r['A'], r['B']
        try:
            rbm_A, rbm_B = load_rbm(A), load_rbm(B)
        except FileNotFoundError as exc:
            print(f"  skip {A} -> {B}: {exc}")
            continue
        E_A = lambda s, m=rbm_A: rbm_energy_batch(np.atleast_2d(s), m)
        E_B = lambda s, m=rbm_B: rbm_energy_batch(np.atleast_2d(s), m)
        oA = ligand_offset(A, df_ms, E_A)
        oB = ligand_offset(B, df_ms, E_B)
        eA = lambda s, o=oA: E_A(s) - o
        eB = lambda s, o=oB: E_B(s) - o
        cA = natural_candidates(A, df_ms, E_A, ANCH_A)
        cB = natural_candidates(B, df_ms, E_B, ANCH_B)
        pairs = pick_pairs(cA, cB, 1)
        if not pairs:  # the anchor filters only fit the B locus
            cA = natural_candidates(A, df_ms, E_A, set('ACDEFGHIKLMNPQRSTVWY'))
            cB = natural_candidates(B, df_ms, E_B, set('ACDEFGHIKLMNPQRSTVWY'))
            pairs = pick_pairs(cA, cB, 1)
        if not pairs:
            print(f"  skip {A} -> {B}: no endpoint pair")
            continue
        v0, vT, D = pairs[0]
        either = lambda s: np.minimum(eA(s), eB(s))
        paths = sample_pooled(np.asarray(v0), np.asarray(vT), either)
        _, raw = width_profile(paths, either)
        E_int, _ = _node_energies(paths, either, zero)
        rows.append({
            'A': A, 'B': B, 'jsd': r['jsd'], 'D': int(D),
            'S_min': r['S_min'], 'N_min': r['N_min'],
            'kappa_bar': float(raw.mean()),  # the mean over edges, not the sum
            'kappa_sum': float(np.log(np.exp(raw).sum())),
            'kappa_min': float(raw.min()),
            'barrier': float(E_int[:, 1:-1].max(axis=1).mean()
                             - 0.5 * (E_int[:, 0].mean() + E_int[:, -1].mean())),
        })
        print(f"  {A} -> {B}  JSD={r['jsd']:.3f} D={D}  "
              f"mean log kappa={raw.mean():+.2f}  barrier={rows[-1]['barrier']:+.2f}",
              flush=True)
    return pd.DataFrame(rows)


def _partial_spearman(df, x, y, controls):
    '''Spearman correlation of x and y after removing `controls` from both.'''
    r = df[[x, y] + list(controls)].rank()
    Z = np.column_stack([np.ones(len(r))] + [r[c].to_numpy() for c in controls])
    resid = {}
    for v in (x, y):
        beta, *_ = np.linalg.lstsq(Z, r[v].to_numpy(), rcond=None)
        resid[v] = r[v].to_numpy() - Z @ beta
    out = spearmanr(resid[x], resid[y])
    return float(out.statistic), float(out.pvalue)


def report_part2(d2):
    '''Print the D and entropy correlations before JSD, then the partial.'''
    if len(d2) < 5:
        print(f"\n  only {len(d2)} corridors, too few to regress")
        return
    print(f"\n  confounds first ({len(d2)} corridors)")
    for c in ('D', 'S_min', 'N_min'):
        rho, p = spearmanr(d2[c], d2['kappa_bar'])
        print(f"  mean log kappa vs {c:6s}: rho={rho:+.2f}  p={p:.2g}")
    rho_j, p_j = spearmanr(d2['jsd'], d2['kappa_bar'])
    print(f"  mean log kappa vs JSD   : rho={rho_j:+.2f}  p={p_j:.2g}  (raw)")
    rho_p, p_p = _partial_spearman(d2, 'jsd', 'kappa_bar', ['D', 'S_min', 'N_min'])
    print(f"  PARTIAL given D, S, N   : rho={rho_p:+.2f}  p={p_p:.2g}")
    rho_s, p_s = spearmanr(d2['jsd'], d2['kappa_sum'])
    print(f"\n  the trap, for contrast, log SUM kappa vs JSD: rho={rho_s:+.2f} "
          f"p={p_s:.2g}; JSD vs D: rho={spearmanr(d2['jsd'], d2['D']).statistic:+.2f}")
    return rho_j, p_j, rho_p, p_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0, help='1, 2, or 0 for both')
    ap.add_argument('--k', type=int, default=K_PAIRS,
                    help='corridors per family in part 1 (default %(default)s)')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    if args.plot:
        plot()
        return

    store = {}
    if args.part in (0, 1):
        print("PART 1: does the entropic route width dip between RBM modes?")
        s1, d1 = part1(args.k)
        store.update(s1)
        d1.to_csv(CSV.replace('.csv', '_part1.csv'), index=False)
    if args.part in (0, 2):
        print("\nPART 2: does JSD predict the mean width, net of confounds?")
        d2 = part2()
        d2.to_csv(CSV.replace('.csv', '_part2.csv'), index=False)
        report_part2(d2)
        for c in d2.columns:
            if d2[c].dtype.kind in 'ifb':
                store[f'p2__{c}'] = d2[c].to_numpy()
    if store:
        # merge so one part does not wipe the other's cache
        if os.path.exists(CACHE):
            with np.load(CACHE, allow_pickle=True) as old:
                merged = {k: old[k] for k in old.files}
            merged.update(store)
            store = merged
        np.savez(CACHE, **store)
        print(f"\nCached -> {CACHE}")
        plot()


def plot():
    c = np.load(CACHE, allow_pickle=True)
    have1 = 'cross_prof_x' in c.files
    have2 = any(k.startswith('p2__') for k in c.files)
    n_ax = 2 * have1 + have2
    if n_ax == 0:
        print("nothing cached to plot")
        return
    fig, axes = plt.subplots(1, n_ax, figsize=(5.4 * n_ax, 4.6))
    ax = list(np.atleast_1d(axes))

    if have1:
        for tag, col, lab in (('cross', ORANGE, 'cross-mode route'),
                              ('within', BLUE, 'within-mode route')):
            ax[0].plot(c[f'{tag}_prof_x'], c[f'{tag}_prof'], marker='o', lw=2.2,
                       color=col, label=lab)
        ax[0].axhline(0, color='0.6', lw=1.0, zorder=0)
        ax[0].set_xlabel('position along route', fontsize=FS_LABEL)
        ax[0].set_ylabel(r'route width $\log\kappa$ (nats, centred)', fontsize=FS_LABEL)
        ax[0].set_title('(a) representative width profiles', fontsize=FS_TITLE)
        ax[0].legend(fontsize=FS_TICK)

        cd, wd = c['cross_dips'], c['within_dips']
        parts = ax[1].boxplot([wd, cd], tick_labels=['within', 'cross'], widths=0.55,
                              patch_artist=True, medianprops=dict(color='k'))
        for patch, col in zip(parts['boxes'], (BLUE, ORANGE)):
            patch.set_facecolor(col); patch.set_alpha(0.65)
        for i, v in ((1, wd), (2, cd)):
            jitter = np.random.default_rng(0).normal(0, 0.04, len(v))
            ax[1].plot(np.full(len(v), i) + jitter, v, 'o', color='0.25', ms=4, zorder=3)
        ax[1].set_ylabel('width dip (nats)', fontsize=FS_LABEL)
        ax[1].set_title(f"(b) dip: Mann-Whitney $p$ = {float(c['mw_p'][0]):.3g}",
                        fontsize=FS_TITLE)

    if have2:
        k = 2 if have1 else 0
        jsd, kb = c['p2__jsd'], c['p2__kappa_bar']
        sc = ax[k].scatter(jsd, kb, c=c['p2__D'], cmap='viridis', s=70,
                           edgecolor='k', linewidth=0.5)
        plt.colorbar(sc, ax=ax[k], label='$D$')
        rho, p = spearmanr(jsd, kb)
        ax[k].set_xlabel('JSD between the two alleles', fontsize=FS_LABEL)
        ax[k].set_ylabel(r'mean per-edge $\log\kappa$ (nats)', fontsize=FS_LABEL)
        ax[k].set_title(rf'(c) JSD vs width: $\rho={rho:+.2f}$, $p={p:.2g}$',
                        fontsize=FS_TITLE)

    for a in ax:
        a.tick_params(labelsize=FS_TICK)
    fig.suptitle('Pilot: entropic route width from the birth-move partition function',
                 fontsize=FS_TITLE + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIGURES_DIR, 'pilot_route_width.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
