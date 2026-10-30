# Chapter 5, the barrier between one allele's own binding modes, with no
# second allele involved. The same paths are scored under the RBM and under a
# PWM fitted to the same ligands: a PWM treats the positions as independent
# and cannot represent two modes at all, so the barrier is there under one
# and absent under the other. The allele is set by MCB_ALLELE, since the
# chapter reports two.

import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import logomaker
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_pwm_rbm_paths as C
from compare_pwm_rbm_paths import (sample_pooled, epistatic_profile, load_rbm,
                                    rbm_energy_batch, pwm_energy_batch, _complex)
from path_stats import _node_energies
from sampler import hamming
from entropy import convert_number, convert_letter, pwm_train
from config import IEDB_MS_CSV, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_LEGEND, FS_TICK

C.COMMON = dict(beta_init=0.15, anneal='linear', n_burn=3000, n_iter=15000, n_thin=100)  # lighter budget
ALLELE = os.environ.get('MCB_ALLELE', 'HLA-B*51:01')
K_PAIRS = 10
SHOW_LOGOS = os.environ.get('MCB_LOGOS', '0' if ALLELE == 'HLA-B*51:01' else '1') == '1'
_SUF = '' if ALLELE == 'HLA-B*51:01' else '_' + ALLELE.replace('HLA-', '').replace('*', '').replace(':', '')
CACHE = os.path.join(FIGURES_DIR, f'mode_crossing_barrier{_SUF}.npz')
FIGOUT = os.path.join(FIGURES_DIR, f'mode_crossing_barrier{_SUF}.pdf')
C_PWM = '#5C82D6'
C_RBM = '#EA9542'
AA20 = list('ACDEFGHIKLMNPQRSTVWY')
zero = lambda s: np.zeros(np.atleast_2d(s).shape[0])


def information_matrix(seqs, L=9):
    '''Per-residue information content in bits for a set of peptides.'''
    idx = {a: i for i, a in enumerate(AA20)}
    counts = np.zeros((L, len(AA20)), dtype=int)
    for s in seqs:
        for pos, a in enumerate(s):
            if a in idx:
                counts[pos, idx[a]] += 1
    freq = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    ic = np.clip(np.log2(len(AA20)) + np.sum(freq * np.log2(freq + 1e-9), axis=1), 0, None)
    df = pd.DataFrame(freq * ic[:, None], columns=AA20)
    df.index = np.arange(1, L + 1)
    return df


def mode_divergence(seqs0, seqs1, L=9):
    '''Per-position Jensen-Shannon divergence in bits between the two mode motifs.'''
    f0, f1 = information_matrix(seqs0, L), information_matrix(seqs1, L)
    # information_matrix returns freq * IC
    norm = lambda d: d.values / np.maximum(d.values.sum(axis=1, keepdims=True), 1e-12)
    p, q = norm(f0), norm(f1)
    m = 0.5 * (p + q)
    safe = lambda x: np.maximum(x, 1e-12)  # keep log2 off zero
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(safe(a) / safe(b)), 0), axis=1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def topk_pairs(mode_pairs, enc, e_rbm, e_pwm, K, min_d=7):
    '''K disjoint distant pairs with the smallest energy gap under both models.'''
    seen, cands = set(), []
    for A, B in mode_pairs:
        for ia in A:
            for ib in B:
                if ia == ib:
                    continue
                key = (min(int(ia), int(ib)), max(int(ia), int(ib)))
                if key in seen:
                    continue
                seen.add(key)
                d = hamming(enc[ia], enc[ib])
                if d < min_d:
                    continue
                score = abs(e_rbm[ia] - e_rbm[ib]) + abs(e_pwm[ia] - e_pwm[ib]) - 0.02 * d
                cands.append((score, int(ia), int(ib), int(d)))
    cands.sort(key=lambda t: t[0])
    picks, used = [], set()
    for _, ia, ib, d in cands:
        if ia in used or ib in used:
            continue
        picks.append((ia, ib, d)); used.update((ia, ib))
        if len(picks) >= K:
            break
    return picks


def cluster_modes(rbm, enc, e_rbm, top_frac=0.25):
    order = np.argsort(e_rbm)
    cut = max(int(len(order) * top_frac), 40)
    good = np.array([i for i in order[:cut] if _complex(enc[i])])
    H = rbm.input_hiddens(enc[good])
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(H)
    return good, km, H


def compute():
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
    d_between = float(np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1]))
    d_within = float(np.mean([np.linalg.norm(H[i] - km.cluster_centers_[lab[i]])
                              for i in range(len(H))]))
    m0, m1 = good[lab == 0], good[lab == 1]
    print(f"{ALLELE}: {len(good)} good binders, mode sizes {np.bincount(lab)}, "
          f"separation between/within = {d_between/d_within:.2f}")

    cross = topk_pairs([(m0, m1)], enc, e_rbm, e_pwm, K_PAIRS)
    within = topk_pairs([(m0, m0), (m1, m1)], enc, e_rbm, e_pwm, K_PAIRS)

    rbm.random_state = np.random.RandomState(0); rbm.AIS(M=100, n_betas=20000)
    logZ = float(rbm.log_Z_AIS)

    store = {'separation': np.array([d_between / d_within]),
             'mode_sizes': np.bincount(lab)}
    for tag, pairs in [('cross', cross), ('within', within)]:
        barriers, belows, profiles = [], [], []
        for ia, ib, d in pairs:
            v0, vT = enc[ia], enc[ib]
            paths = sample_pooled(v0, vT, E_pwm)
            barrier = float(epistatic_profile(paths, E_rbm, E_pwm).max())
            Fr, _ = _node_energies(paths, E_rbm, zero)
            Ep, _ = _node_energies(paths, E_pwm, zero)
            llR = -Fr.mean(0) - logZ
            llP = -Ep.mean(0)
            below = float(np.mean(llR[1:-1] < llP[1:-1]))
            barriers.append(barrier); belows.append(below)
            profiles.append((ia, ib, d, llP, llR))
            print(f"  {tag}: {''.join(convert_letter(np.asarray(v0)))} -> "
                  f"{''.join(convert_letter(np.asarray(vT)))}  D={d}  "
                  f"barrier={barrier:.2f}  below={below:.2f}")
        barriers = np.array(barriers)
        store[f'{tag}_barriers'] = barriers
        store[f'{tag}_below'] = np.array(belows)
        med_idx = int(np.argsort(barriers)[len(barriers) // 2])  # median of the ten pairs
        ia, ib, d, llP, llR = profiles[med_idx]
        store[f'{tag}_x'] = np.linspace(0, 1, len(llP))
        store[f'{tag}_llP'] = llP
        store[f'{tag}_llR'] = llR
        store[f'{tag}_repr_barrier'] = np.array([barriers[med_idx]])
        store[f'{tag}_seq'] = np.array([''.join(convert_letter(np.asarray(enc[ia]))),
                                        ''.join(convert_letter(np.asarray(enc[ib])))])
    np.savez(CACHE, **store)
    print(f"\nCached -> {CACHE}")


def plot():
    c = np.load(CACHE, allow_pickle=True)
    cross_b, within_b = c['cross_barriers'], c['within_barriers']
    U, p = mannwhitneyu(cross_b, within_b, alternative='greater')
    # re-deriving the modes here samples nothing
    df = pd.read_csv(IEDB_MS_CSV)
    rbm = load_rbm(ALLELE)
    seqs = [s for s in df[df['MHC allele'] == ALLELE]['Peptide'] if len(s) == 9]
    enc = convert_number(seqs)
    e_rbm = np.asarray(rbm_energy_batch(np.atleast_2d(enc), rbm), float)
    good, km, H = cluster_modes(rbm, enc, e_rbm)
    pca = PCA(n_components=2, random_state=0).fit(H)
    xy, lab = pca.transform(H), km.labels_
    all_seqs = [''.join(convert_letter(np.asarray(enc[i]))) for i in range(len(enc))]
    pwm = pwm_train(enc)
    E_pwm = lambda s, p=pwm: pwm_energy_batch(np.atleast_2d(s), p)

    E_rbm = lambda s, m=rbm: rbm_energy_batch(np.atleast_2d(s), m)

    def route_paths(seqpair):
        idx = [all_seqs.index(str(s)) for s in seqpair]
        return sample_pooled(enc[idx[0]], enc[idx[1]], E_pwm)

    def route_traj(paths):
        '''The sampled paths projected into the hidden PCA plane, (n, T+1, 2).'''
        n, T1 = paths.shape[0], paths.shape[1]
        Hin = rbm.input_hiddens(paths.reshape(n * T1, -1)).reshape(n, T1, -1)
        return pca.transform(Hin.reshape(n * T1, -1)).reshape(n, T1, 2)

    def route_spread(paths):
        '''Standard deviation of the two log-likelihoods across paths at each node.'''
        d_rbm, _ = _node_energies(paths, E_rbm, zero)  # (n, T+1)
        d_pwm, _ = _node_energies(paths, E_pwm, zero)
        return d_pwm.std(0), d_rbm.std(0), -d_pwm.mean(0)

    # shared by the trajectory panel and the profile bands
    ens = {tag: route_paths(c[f'{tag}_seq']) for tag in ('cross', 'within')}
    spread = {tag: route_spread(p) for tag, p in ens.items()}
    for tag, (_, _, mean_llP) in spread.items():  # ensemble check
        drift = float(np.abs(mean_llP - c[f'{tag}_llP']).max())
        print(f"{tag}: re-sampled ensemble matches cache to {drift:.2e} nats")

    var = pca.explained_variance_ratio_ * 100
    # the bit scale stretches a logo's letters vertically
    fig, axes = plt.subplots(3 if SHOW_LOGOS else 2, 2,
                             figsize=(12.5, 12.5 if SHOW_LOGOS else 10.5),
                             gridspec_kw=dict(height_ratios=[1, 1, 0.5])
                             if SHOW_LOGOS else None)
    C_M0, C_M1 = '#59A14F', '#B07AA1'  # distinct from PWM/RBM

    ax = axes[0, 0]
    ax.scatter(xy[lab == 0, 0], xy[lab == 0, 1], s=16, color=C_M0, alpha=0.45,
               edgecolor='none', label='mode 1')
    ax.scatter(xy[lab == 1, 0], xy[lab == 1, 1], s=16, color=C_M1, alpha=0.45,
               edgecolor='none', label='mode 2')
    tc = route_traj(ens['cross'])
    tw = route_traj(ens['within'])
    for tr in tc:
        ax.plot(tr[:, 0], tr[:, 1], '-', color='black', lw=0.6, alpha=0.05, zorder=3)
    for tr in tw:
        ax.plot(tr[:, 0], tr[:, 1], '-', color='0.35', lw=0.6, alpha=0.05, zorder=3)
    mc, mw = tc.mean(0), tw.mean(0)
    ax.plot(mc[:, 0], mc[:, 1], '-', color='black', lw=2.3, zorder=4, label='cross-mode route')
    ax.scatter(mc[[0, -1], 0], mc[[0, -1], 1], marker='*', s=260, color='black', zorder=5)
    ax.plot(mw[:, 0], mw[:, 1], '--', color='0.35', lw=2.3, zorder=4, label='within-mode route')
    ax.scatter(mw[[0, -1], 0], mw[[0, -1], 1], marker='D', s=80, color='0.35', zorder=5)
    ax.set_title('binding modes  (RBM hidden PCA)', fontsize=FS_TITLE)
    ax.set_xlabel(f'PC1 of hidden inputs  ({var[0]:.0f}%)', fontsize=FS_LABEL)
    ax.set_ylabel(f'PC2 of hidden inputs  ({var[1]:.0f}%)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_TICK, loc='best', framealpha=0.9)

    for ax, tag, name in [(axes[0, 1], 'cross', 'cross-mode'),
                          (axes[1, 0], 'within', 'within-mode')]:
        x, llP, llR = c[f'{tag}_x'], c[f'{tag}_llP'], c[f'{tag}_llR']
        sd_P, sd_R, _ = spread[tag]
        ax.fill_between(x, llR, llP, color='0.5', alpha=0.15, zorder=0)
        ax.fill_between(x, llP - sd_P, llP + sd_P, color=C_PWM, alpha=0.16,
                        lw=0, zorder=1)
        ax.fill_between(x, llR - sd_R, llR + sd_R, color=C_RBM, alpha=0.16,
                        lw=0, zorder=1)
        ax.plot(x, llP, '-o', color=C_PWM, lw=2.3, ms=6, zorder=2,
                label=r'under PWM  $\log p^{\mathrm{PWM}}$')
        ax.plot(x, llR, '-s', color=C_RBM, lw=2.3, ms=6, zorder=2,
                label=r'under RBM  $\log p^{\mathrm{RBM}}$')
        ax.set_title(f'{name} route', fontsize=FS_TITLE)
        ax.set_xlabel('path progress', fontsize=FS_LABEL)
        ax.set_ylabel('log-likelihood  (nats)', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND, loc='lower center', framealpha=0.9)

    ax = axes[1, 1]
    rng = np.random.RandomState(0)
    for i, (b, col) in enumerate([(cross_b, C_RBM), (within_b, C_PWM)]):
        xj = rng.normal(i, 0.06, len(b))
        ax.scatter(xj, b, color=col, s=55, alpha=0.85, edgecolor='white', linewidth=0.6, zorder=2)
        ax.hlines(np.median(b), i - 0.22, i + 0.22, color='black', lw=2.4, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['cross-mode', 'within-mode'], fontsize=FS_LABEL)
    ax.set_ylabel(r'peak $\log p^{\mathrm{PWM}} - \log p^{\mathrm{RBM}}$  (nats)', fontsize=FS_LABEL)
    ax.set_title('additive overestimate', fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(-0.5, 1.5)

    seqs_by_mode = [[all_seqs[i] for i, m in zip(good, lab) if m == k] for k in (0, 1)]
    jsd_pos = mode_divergence(*seqs_by_mode)
    split = np.where(jsd_pos > 0.5)[0] + 1  # positions that separate the modes
    if SHOW_LOGOS:
        for ax, k, col, name in [(axes[2, 0], 0, C_M0, 'mode 1'), (axes[2, 1], 1, C_M1, 'mode 2')]:
            logomaker.Logo(information_matrix(seqs_by_mode[k]), ax=ax, color_scheme='chemistry')
            ax.set_title(f'{name} motif  ({len(seqs_by_mode[k])} binders)',
                         fontsize=FS_TITLE, color=col)
            ax.set_xlabel('position', fontsize=FS_LABEL)
            ax.set_ylabel('bits', fontsize=FS_LABEL)
            ax.set_xticks(np.arange(1, 10))
            ax.set_ylim(0, np.log2(len(AA20)) + 0.1)
            ax.tick_params(labelsize=FS_TICK)
            for pos in split:
                ax.axvspan(pos - 0.5, pos + 0.5, color='0.5', alpha=0.12, zorder=0)
    print(f"mode sizes {[len(s) for s in seqs_by_mode]}; per-position mode JSD (bits): "
          f"{', '.join(f'P{i+1} {v:.2f}' for i, v in enumerate(jsd_pos))}")
    print(f"positions separating the modes (JSD > 0.5): {list(split)}")

    for axp, L in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        axp.annotate(L, xy=(0, 1), xycoords='axes fraction', xytext=(2, 6),
                     textcoords='offset points', fontsize=FS_TITLE, fontweight='bold',
                     va='bottom', ha='left')
    plt.tight_layout()
    out = FIGOUT
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"cross barrier  = {cross_b.mean():.2f} +/- {cross_b.std():.2f} nats "
          f"(median {np.median(cross_b):.2f})")
    print(f"within barrier = {within_b.mean():.2f} +/- {within_b.std():.2f} nats "
          f"(median {np.median(within_b):.2f})")
    print(f"cross interior RBM<PWM = {c['cross_below'].mean():.2f};  "
          f"within = {c['within_below'].mean():.2f}")
    print(f"Mann-Whitney (cross > within) p = {p:.2e}")
    print(f"Saved: {out}")


if __name__ == '__main__':
    if '--plot' not in sys.argv:
        compute()
    plot()
