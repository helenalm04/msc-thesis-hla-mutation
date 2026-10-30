# Chapter 4, designing peptides rather than sampling paths between them. The
# score the sampler uses is minimised over sequence space by simulated
# annealing, with lambda setting what is asked for. Two allele pairs, with
# overlapping and with conflicting motifs, since what can be designed depends
# on whether both motifs can be satisfied at once at all.

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, RESULTS_DIR, FIGURES_DIR, L, FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK
import anchors
from entropy import convert_number
from encoding import convert_letter

_r = importlib.util.spec_from_file_location("src_rbm", os.path.join(SRC, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_r); _r.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

from sampler import score_lambda

PAIRS = {
    'overlapping': ('HLA-A*02:01', 'HLA-A*02:03'),
    'orthogonal':  ('HLA-B*51:01', 'HLA-B*27:05'),
}
N_CHAINS, N_STEPS = 300, 6000
BETA0, BETA1 = 0.1, 4.0
N_AA = 20
MIN_DISTINCT = 6


# mean energy of an allele's own ligands, the baseline its centred energy is taken from
def ligand_offset(allele, df, E_fn):
    seqs = df[df['MHC allele'] == allele]['Peptide'].tolist()
    return float(np.mean(E_fn(convert_number(seqs))))


# the residues an allele accepts at its P2 anchor position
def anchor_set(allele, df):
    return anchors.anchor_set(allele, df, anchors.P2)


# number of distinct residues in each peptide of a batch
def _n_distinct(X):
    return np.array([np.unique(r).size for r in X])


def _sample_ligands(allele, df, n=600, seed=0):
    seqs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == L]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    return convert_number([seqs[i] for i in idx])


LAMS = [1.0, 0.5, 0.0, -0.25, -0.5]
LAM_EDGE = -0.9
PRESENTED = 2.0  # centred energy above this counts as not presented
CACHE = os.path.join(RESULTS_DIR, 'design_lambda.npz')


def anneal(cE_A, cE_B, lam, seed=0, beta_scale=1.0):
    '''Minimise E_lambda over 9-mers by simulated annealing.'''
    rng = np.random.default_rng(seed)
    X = rng.integers(0, N_AA, size=(N_CHAINS, L))
    sc = lambda B: score_lambda(np.asarray(cE_A(B), float), np.asarray(cE_B(B), float), lam)
    e = sc(X)
    for beta in np.linspace(BETA0, BETA1, N_STEPS) * beta_scale:
        pos = rng.integers(0, L, size=N_CHAINS)
        Xp = X.copy()
        Xp[np.arange(N_CHAINS), pos] = rng.integers(0, N_AA, size=N_CHAINS)
        ep = sc(Xp)
        acc = (rng.random(N_CHAINS) < np.exp(-beta * (ep - e))) & (_n_distinct(Xp) >= MIN_DISTINCT)
        X[acc], e[acc] = Xp[acc], ep[acc]
    return np.unique(X, axis=0)


def summarise(des, cA, cB, anchA, anchB, thrA, thrB):
    '''Return the energies and the anchor classification of each design.'''
    eA, eB = cA(des), cB(des)
    p2 = np.array([s[1] for s in convert_letter(des)])
    okA = np.array([r in anchA for r in p2])
    okB = np.array([r in anchB for r in p2])
    m, M = np.minimum(eA, eB), np.maximum(eA, eB)
    return dict(
        eA=eA, eB=eB, m=m, M=M, sep=M - m, okA=okA, okB=okB,
        n=len(des),
        f_both=float((okA & okB).mean()),
        f_Aonly=float((okA & ~okB).mean()),
        f_Bonly=float((okB & ~okA).mean()),
        f_none=float((~okA & ~okB).mean()),
        # genuine designs need the right anchor and a good-binder energy
        f_super=float((okA & okB & (eA < thrA) & (eB < thrB)).mean()),
        f_single=float((((okA & (eA < thrA) & ~okB) | (okB & (eB < thrB) & ~okA))).mean()),
        f_orth=float((((eA < thrA) & (eB > PRESENTED)) | ((eB < thrB) & (eA > PRESENTED))).mean()),
        f_towardsA=float((eA < eB).mean()),
        mean_m=float(m.mean()), mean_M=float(M.mean()), mean_sep=float((M - m).mean()),
        mean_eA=float(eA.mean()), mean_eB=float(eB.mean()),
    )


def compute():
    df = pd.read_csv(IEDB_MS_CSV)
    store, t0 = {}, time.time()
    for label, (a, b) in PAIRS.items():
        rbmA, rbmB = load_rbm(a), load_rbm(b)
        EA = lambda s, m=rbmA: rbm_energy_batch(np.atleast_2d(s), m)
        EB = lambda s, m=rbmB: rbm_energy_batch(np.atleast_2d(s), m)
        offA, offB = ligand_offset(a, df, EA), ligand_offset(b, df, EB)
        cA = lambda s, o=offA, f=EA: np.asarray(f(s), float) - o
        cB = lambda s, o=offB, f=EB: np.asarray(f(s), float) - o

        anchA, anchB = anchor_set(a, df), anchor_set(b, df)
        ligA, ligB = _sample_ligands(a, df), _sample_ligands(b, df)
        thrA, thrB = float(np.percentile(cA(ligA), 25)), float(np.percentile(cB(ligB), 25))

        print(f"\n{label}: {a} (P2 {'/'.join(sorted(anchA))}) + {b} (P2 {'/'.join(sorted(anchB))}) "
              f" good-binder bar {thrA:+.1f}/{thrB:+.1f} ===", flush=True)
        print(f"{'lam':>6} {'n':>5} {'E_A':>7} {'E_B':>7} {'sep':>6} "
              f"{'both':>6} {'Aonly':>6} {'Bonly':>6} {'none':>6} "
              f"{'super':>6} {'single':>7} {'orth':>6}   (ctrl: m, sep)", flush=True)

        rows = {}
        for lam in LAMS + [LAM_EDGE]:
            des = anneal(cA, cB, lam)
            s = summarise(des, cA, cB, anchA, anchB, thrA, thrB)
            ctl = summarise(anneal(cA, cB, 0.0, beta_scale=1.0 + lam), cA, cB, anchA, anchB, thrA, thrB)
            s['ctl_mean_m'], s['ctl_mean_sep'] = ctl['mean_m'], ctl['mean_sep']
            rows[lam] = s
            print(f"{lam:>6.2f} {s['n']:>5d} {s['mean_eA']:>+7.2f} {s['mean_eB']:>+7.2f} "
                  f"{s['mean_sep']:>6.2f} {100*s['f_both']:>5.0f}% {100*s['f_Aonly']:>5.0f}% "
                  f"{100*s['f_Bonly']:>5.0f}% {100*s['f_none']:>5.0f}% {100*s['f_super']:>5.0f}% "
                  f"{100*s['f_single']:>6.0f}% {100*s['f_orth']:>5.0f}%   "
                  f"({ctl['mean_m']:+.2f}, {ctl['mean_sep']:.2f})", flush=True)
            best = np.argsort(rows[lam]['m'])[:2]
            for i in best:
                print(f"         {convert_letter(des)[i]}  E_A={s['eA'][i]:+.2f}  E_B={s['eB'][i]:+.2f}", flush=True)

        store[label] = dict(a=a, b=b, thrA=thrA, thrB=thrB,
                            anchA='/'.join(sorted(anchA)), anchB='/'.join(sorted(anchB)),
                            ligA_A=cA(ligA), ligA_B=cB(ligA), ligB_A=cA(ligB), ligB_B=cB(ligB),
                            rows=rows)

    flat = {}
    for k, d in store.items():
        for kk, vv in d.items():
            if kk == 'rows':
                for lam, s in vv.items():
                    for sk, sv in s.items():
                        flat[f'{k}__lam{lam}__{sk}'] = sv
            else:
                flat[f'{k}__{kk}'] = vv
    np.savez(CACHE, labels=np.array(list(store.keys())),
             lams=np.array(LAMS + [LAM_EDGE]), **flat)
    print(f"\nCached -> {CACHE}   ({time.time()-t0:.0f}s)")


PANEL_SEEDS = [0, 1, 2]
PANEL_LAM_GRID = [1.0, 0.5, 0.0, -0.25, -0.5]  # same grid as LAMS in scan_lambda_vs_jsd.py
PANEL_CACHE = os.path.join(RESULTS_DIR, 'design_lambda_panel.npz')


def compute_panel():
    sys.path.insert(0, HERE)
    from scan_lambda_vs_jsd import select_pairs

    df = pd.read_csv(IEDB_MS_CSV)
    pairs = select_pairs()
    print(f"{len(pairs)} stratified pairs, JSD {pairs[0][2]:.2f}-{pairs[-1][2]:.2f}, "
          f"{len(PANEL_LAM_GRID)} lambdas x {len(PANEL_SEEDS)} seeds (+ controls)\n", flush=True)

    rows, t0 = [], time.time()
    for pi, (a, b, jsd) in enumerate(pairs):
        try:
            rbmA, rbmB = load_rbm(a), load_rbm(b)
        except Exception as exc:
            print(f"  [skip] {a} + {b}: {exc}", flush=True)
            continue
        EA = lambda s, m=rbmA: rbm_energy_batch(np.atleast_2d(s), m)
        EB = lambda s, m=rbmB: rbm_energy_batch(np.atleast_2d(s), m)
        offA, offB = ligand_offset(a, df, EA), ligand_offset(b, df, EB)
        cA = lambda s, o=offA, f=EA: np.asarray(f(s), float) - o
        cB = lambda s, o=offB, f=EB: np.asarray(f(s), float) - o
        anchA, anchB = anchor_set(a, df), anchor_set(b, df)
        ligA, ligB = _sample_ligands(a, df), _sample_ligands(b, df)
        thrA, thrB = float(np.percentile(cA(ligA), 25)), float(np.percentile(cB(ligB), 25))
        dnh = abs(int(rbmA.n_h) - int(rbmB.n_h))

        for lam in PANEL_LAM_GRID:
            per_seed = [summarise(anneal(cA, cB, lam, seed=s), cA, cB, anchA, anchB, thrA, thrB)
                        for s in PANEL_SEEDS]
            ctl = [summarise(anneal(cA, cB, 0.0, seed=s, beta_scale=1.0 + lam),
                             cA, cB, anchA, anchB, thrA, thrB) for s in PANEL_SEEDS]
            grab = lambda rs, k: np.array([r[k] for r in rs], float)
            rows.append(dict(
                a=a, b=b, jsd=jsd, lam=lam, nhA=int(rbmA.n_h), nhB=int(rbmB.n_h), dnh=dnh,
                sep=grab(per_seed, 'mean_sep').mean(), sep_sd=grab(per_seed, 'mean_sep').std(ddof=1),
                m=grab(per_seed, 'mean_m').mean(), m_sd=grab(per_seed, 'mean_m').std(ddof=1),
                ctl_sep=grab(ctl, 'mean_sep').mean(), ctl_m=grab(ctl, 'mean_m').mean(),
                valid=1.0 - grab(per_seed, 'f_none').mean(),
                valid_sd=grab(per_seed, 'f_none').std(ddof=1),
                genuine=(grab(per_seed, 'f_super') + grab(per_seed, 'f_single')).mean(),
                genuine_sd=(grab(per_seed, 'f_super') + grab(per_seed, 'f_single')).std(ddof=1),
                # stored separately as well as summed
                super_=grab(per_seed, 'f_super').mean(), single=grab(per_seed, 'f_single').mean(),
                orth=grab(per_seed, 'f_orth').mean()))
            r = rows[-1]
            print(f"  [{pi+1:>2}/{len(pairs)}] {a} + {b}  JSD {jsd:.2f}  dn_h {dnh:>2}  "
                  f"lam {lam:>+5.2f}  sep {r['sep']:>6.2f}+/-{r['sep_sd']:<4.2f} "
                  f"(ctl {r['ctl_sep']:>5.2f})  valid {100*r['valid']:>3.0f}%  "
                  f"genuine {100*r['genuine']:>3.0f}%", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(PANEL_CACHE.replace('.npz', '.csv'), index=False)
    np.savez(PANEL_CACHE, **{c: out[c].values for c in out.columns})
    print(f"\nCached -> {PANEL_CACHE}  ({time.time()-t0:.0f}s, {len(out)} rows)")
    return out


C_A, C_B, C_D = '#5C82D6', '#EA9542', '#222222'
C_SAME, C_DIFF = '#4CA64C', '#B07AA1'  # same and cross supertype
PAIR_COL = {'overlapping': C_SAME, 'orthogonal': C_DIFF}
PANEL_LAMS = [1.0, 0.0, -0.5]


def _row_limits(z, label, pad=2.5):
    v = [z[f'{label}__{k}'] for k in ('ligA_A', 'ligA_B', 'ligB_A', 'ligB_B')]
    for lam in PANEL_LAMS:
        v += [z[f'{label}__lam{lam}__eA'], z[f'{label}__lam{lam}__eB']]
    lo = min(float(np.min(x)) for x in v) - pad
    hi = max(float(np.max(x)) for x in v) + pad
    return lo, hi


def _plane(ax, z, label, lam, lim, show_leg=False):
    g = lambda k: z[f'{label}__{k}']
    p = lambda k: z[f'{label}__lam{lam}__{k}']
    a, b = str(g('a')), str(g('b'))
    ax.scatter(g('ligA_A'), g('ligA_B'), s=7, color=C_A, alpha=0.30, lw=0, label=f'{a} ligands')
    ax.scatter(g('ligB_A'), g('ligB_B'), s=7, color=C_B, alpha=0.30, lw=0, label=f'{b} ligands')
    eA, eB, okA, okB = p('eA'), p('eB'), p('okA').astype(bool), p('okB').astype(bool)
    cls = [(okA & okB,  C_D,  '*', 46, 'both anchors'),
           (okA & ~okB, C_A,  '^', 26, f'{a} anchor only'),
           (okB & ~okA, C_B,  '^', 26, f'{b} anchor only'),
           (~okA & ~okB, 'none', 'o', 24, 'neither anchor')]
    for m, col, mk, s, lab in cls:
        if not m.any():
            continue
        kw = dict(facecolors='none', edgecolors='0.55', lw=0.8) if col == 'none' else dict(color=col)
        ax.scatter(eA[m], eB[m], s=s, marker=mk, label=lab, zorder=3, **kw)
    ax.axvline(float(g('thrA')), color='0.7', lw=0.9, ls=':')
    ax.axhline(float(g('thrB')), color='0.7', lw=0.9, ls=':')
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_title(rf'$\lambda={lam:+.1f}$', fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if show_leg:
        ax.legend(fontsize=FS_LEGEND - 5, loc='upper right', framealpha=0.9,
                  handletextpad=0.3, markerscale=2.2)


def plot():
    z = np.load(CACHE, allow_pickle=True)
    labels, lams = [str(x) for x in z['labels']], list(z['lams'])
    order = sorted(lams)  # lambda in ascending order for the sweep panels
    fig, axes = plt.subplots(2, 4, figsize=(21.5, 10.4))

    for row, label in enumerate(['orthogonal', 'overlapping']):
        g = lambda k: z[f'{label}__{k}']
        lim = _row_limits(z, label)
        for col, lam in enumerate(PANEL_LAMS):
            ax = axes[row, col]
            _plane(ax, z, label, lam, lim, show_leg=(col == 0))
            if col == 0:
                ax.set_ylabel(f'centred energy, {str(g("b"))}  (nats)', fontsize=FS_LABEL - 2)
            if row == 1:
                ax.set_xlabel(f'centred energy, {str(g("a"))}  (nats)', fontsize=FS_LABEL - 2)
        axes[row, 0].text(-0.24, 1.06, f'{str(g("a"))} + {str(g("b"))}', transform=axes[row, 0].transAxes,
                          fontsize=FS_TITLE, fontweight='bold', color=PAIR_COL[label], ha='left')

    ax = axes[0, 3]
    for label in labels:
        c = PAIR_COL[label]
        val = [1 - float(z[f'{label}__lam{l}__f_none']) for l in order]
        gen = [float(z[f'{label}__lam{l}__f_super']) + float(z[f'{label}__lam{l}__f_single']) for l in order]
        ax.plot(order, val, '-o', color=c, lw=2, ms=8, label=f'{label}, anchor-valid')
        ax.plot(order, gen, '--s', color=c, lw=2, ms=7, mfc='white', label=f'{label}, genuine binder')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('fraction of designs', fontsize=FS_LABEL - 1)
    ax.set_title('Does the design survive validation?', fontsize=FS_TITLE)

    ax = axes[1, 3]
    for label in labels:
        c = PAIR_COL[label]
        sep = [float(z[f'{label}__lam{l}__mean_sep']) for l in order]
        ctl = [float(z[f'{label}__lam{l}__ctl_mean_sep']) for l in order]
        ax.plot(order, sep, '-o', color=c, lw=2, ms=8, label=f'{label}, $E_\\lambda$')
        ax.plot(order, ctl, ':', color=c, lw=2, label=f'{label}, temperature control')
    ax.set_ylabel(r'route-mean separation $|E_A - E_B|$  (nats)', fontsize=FS_LABEL - 1)
    ax.set_title(r'Is it the score, or just a greedier anneal?', fontsize=FS_TITLE)

    for ax in (axes[0, 3], axes[1, 3]):
        ax.set_xlabel(r'regime parameter $\lambda$', fontsize=FS_LABEL - 1)
        ax.axvline(0.0, color='0.85', lw=1, zorder=0)
        ax.grid(alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND - 5, framealpha=0.9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'design_lambda.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {out}")


def plot_panel():
    '''Plot design separation against the path separation of Chapter 4.'''
    from scipy.stats import spearmanr

    d = pd.read_csv(PANEL_CACHE.replace('.npz', '.csv'))
    zp = np.load(os.path.join(RESULTS_DIR, 'lambda_vs_jsd.npz'), allow_pickle=True)
    p_lams = [float(x) for x in zp['lams']]
    p_key = {frozenset((str(a), str(b))): i for i, (a, b) in enumerate(zip(zp['alleleA'], zp['alleleB']))}

    def path_sep(a, b, lam):
        i = p_key.get(frozenset((a, b)))
        if i is None or lam not in p_lams:
            return np.nan
        return float(np.nanmean(zp['spread'][p_lams.index(lam), i, :]))

    d['path_sep'] = [path_sep(r.a, r.b, r.lam) for r in d.itertuples()]
    lo = d[d.lam == -0.5].sort_values('jsd').reset_index(drop=True)

    def rho(x, y, lab):
        m = np.isfinite(x) & np.isfinite(y)
        r, p = spearmanr(x[m], y[m])
        print(f"  {lab:44s} rho = {r:+.2f}  p = {p:.3f}  (n = {m.sum()})")
        return r, p

    print(f"\nAt lambda=-0.5, across {len(lo)} pairs:")
    r_dj, p_dj = rho(lo.jsd.values, lo.sep.values, "design separation vs JSD")
    r_pj, p_pj = rho(lo.jsd.values, lo.path_sep.values, "path separation vs JSD (Ch.4 replication)")
    r_dn, p_dn = rho(lo.dnh.values.astype(float), lo.sep.values, "design separation vs |dn_h| (capacity)")
    rho(lo.sep.values, lo.genuine.values, "design separation vs genuine fraction")
    print(f"  design sep range {lo.sep.min():.1f}-{lo.sep.max():.1f} nats, "
          f"control {lo.ctl_sep.min():.1f}-{lo.ctl_sep.max():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.6))
    cmap = plt.get_cmap('Blues')
    jn = (lo.jsd - lo.jsd.min()) / max(lo.jsd.max() - lo.jsd.min(), 1e-9)

    ax = axes[0]  # panel a, design against path
    ax.errorbar(lo.jsd, lo.sep, yerr=lo.sep_sd, fmt='o', color=C_DIFF, ms=9, lw=0,
                elinewidth=1.4, capsize=3, label='design (annealed, unpinned)')
    ax.plot(lo.jsd, lo.path_sep, 's', color=C_SAME, ms=8, mfc='white', mew=2,
            label='path (endpoint-pinned, Ch.~4)')
    ax.plot(lo.jsd, lo.ctl_sep, ':', color='0.55', lw=2, label='design, temperature control')
    ax.set_xlabel('pairwise JSD', fontsize=FS_LABEL - 1)
    ax.set_ylabel(r'separation $|E_A - E_B|$  (nats)', fontsize=FS_LABEL - 1)
    ax.set_title(rf'$\lambda=-0.5$: design $\rho={r_dj:+.2f}$, path $\rho={r_pj:+.2f}$',
                 fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND - 5, framealpha=0.9)

    ax = axes[1]
    for _, g in d.sort_values('lam').groupby(['a', 'b']):
        j = float(g.jsd.iloc[0])
        c = cmap(0.25 + 0.75 * (j - d.jsd.min()) / max(d.jsd.max() - d.jsd.min(), 1e-9))
        ax.plot(g.lam, g.genuine, '-o', color=c, lw=1.8, ms=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(d.jsd.min(), d.jsd.max()))
    cb = fig.colorbar(sm, ax=ax, pad=0.02); cb.set_label('pairwise JSD', fontsize=FS_LABEL - 3)
    cb.ax.tick_params(labelsize=FS_TICK - 2)
    ax.set_xlabel(r'regime parameter $\lambda$', fontsize=FS_LABEL - 1)
    ax.set_ylabel('fraction of designs that are genuine', fontsize=FS_LABEL - 1)
    ax.set_title('Design validity across the family', fontsize=FS_TITLE)

    ax = axes[2]
    ax.errorbar(lo.dnh, lo.sep, yerr=lo.sep_sd, fmt='o', color=C_DIFF, ms=9, lw=0,
                elinewidth=1.4, capsize=3)
    ax.set_xlabel(r'hidden-unit mismatch $|n_h^A - n_h^B|$', fontsize=FS_LABEL - 1)
    ax.set_ylabel(r'design separation at $\lambda=-0.5$  (nats)', fontsize=FS_LABEL - 1)
    ax.set_title(rf'Capacity is not the driver ($\rho={r_dn:+.2f}$, $p={p_dn:.2f}$)', fontsize=FS_TITLE)

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.7); ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS_TICK)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'design_lambda_panel.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    if '--panel' in sys.argv:
        if '--plot' not in sys.argv:
            compute_panel()
        plot_panel()
    else:
        if '--plot' not in sys.argv:
            compute()
        plot()
