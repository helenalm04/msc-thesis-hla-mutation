# The drawing behind the Chapter 5 case study; nothing here computes a result.
# The scatters use _specificity_projection, which takes one hidden unit from
# each allele's own RBM, the one best separating its ligands from the other's.
# The plane is then the models' own reading of what makes a peptide specific.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


def _scatter_legend(allele_A, allele_B, nA, nB, v0_label, vT_label,
                    show_paths=False):
    handles = [
        Line2D([0], [0], marker='o', linestyle='none', markersize=9,
               markerfacecolor='#5C82D6', markeredgecolor='none',
               label=f'{allele_A} (n={nA})'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=9,
               markerfacecolor='#EA9542', markeredgecolor='none',
               label=f'{allele_B} (n={nB})'),
        Line2D([0], [0], marker='*', linestyle='none', markersize=14,
               markerfacecolor='#5C82D6', markeredgecolor='k', label=v0_label),
        Line2D([0], [0], marker='*', linestyle='none', markersize=14,
               markerfacecolor='#EA9542', markeredgecolor='k', label=vT_label),
    ]
    if show_paths:
        handles.append(Line2D([0], [0], color='black', lw=2.0,
                              label='sampled paths'))
    return handles

sys.path.insert(0, os.path.dirname(__file__))
from config import (IEDB_MS_CSV, IEDB_BA_CSV, ALLELE_SUPERTYPES,
                    JSD_CSV, ENTROPY_CSV, RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
                    L, Q_PEP, FS_LABEL, FS_TICK, FS_LEGEND, FS_TITLE)

_FS_LABEL = FS_LABEL
_FS_TICK = FS_TICK
_FS_LEG = FS_LEGEND
_FS_TITLE = FS_TITLE


def load_iedb_data():
    df_ms = pd.read_csv(IEDB_MS_CSV)
    df_ba = pd.read_csv(IEDB_BA_CSV)
    supertypes = pd.read_csv(ALLELE_SUPERTYPES)
    print(f"MS dataset: {len(df_ms)} peptides, {df_ms['MHC allele'].nunique()} alleles")
    print(f"BA dataset: {len(df_ba)} peptides, {df_ba['MHC allele'].nunique()} alleles")
    print("Allele supertypes:")
    print(supertypes)
    return df_ms, df_ba, supertypes


def _overlay_paths(ax, paths, proj_path, n_paths=100, alpha=0.2, seed=0):
    if not paths:
        return

    srng = np.random.default_rng(seed)
    sel = (paths if len(paths) <= n_paths
           else [paths[i] for i in srng.choice(len(paths), n_paths, replace=False)])

    for p in sel:
        P = proj_path(p)
        ax.plot(P[:, 0], P[:, 1], '-', color='black', lw=1.2, alpha=alpha,
                zorder=5)


def _specificity_projection(rbm_A, rbm_B, encA, encB):
    '''Return a projection onto one hidden unit from each RBM.'''
    def I(rbm, enc):
        return np.asarray(rbm.input_hiddens(np.asarray(enc).astype(np.int16)))

    # the most discriminating unit, by a standardised difference of means so
    # that units on different scales are compared fairly. The sign is
    # returned with it, so the axis always points away from the other allele.
    def pick(I_self, I_other):
        d = I_self.mean(0) - I_other.mean(0)
        s = np.sqrt(0.5 * (I_self.var(0) + I_other.var(0))) + 1e-9
        score = d / s
        u = int(np.argmax(np.abs(score)))
        return u, (1.0 if score[u] >= 0 else -1.0)

    # each model is run on both repertoires, so each can be asked which of its
    # own units best tells them apart
    IA_A, IA_B = I(rbm_A, encA), I(rbm_A, encB)  # A-RBM inputs on A and B ligands
    IB_A, IB_B = I(rbm_B, encA), I(rbm_B, encB)
    muA, sgnA = pick(IA_A, IA_B)
    nuB, sgnB = pick(IB_B, IB_A)

    def proj(enc):
        enc = np.atleast_2d(np.asarray(enc))  # takes one peptide or a batch
        x = sgnA * I(rbm_A, enc)[:, muA]
        y = sgnB * I(rbm_B, enc)[:, nuB]
        return np.column_stack([x, y])

    XA = np.column_stack([sgnA * IA_A[:, muA], sgnB * IB_A[:, nuB]])
    XB = np.column_stack([sgnA * IA_B[:, muA], sgnB * IB_B[:, nuB]])
    return proj, XA, XB, muA, nuB


def plot_case_study_figure(allele_A, allele_B, rbm_A, rbm_B, df_ms=None,
                           v0=None, vT=None, paths=None, n_paths=20,
                           path_alpha=0.7, n_pc=2, seed=0, save=True):
    '''Draw the PCA and specificity scatters above the loading logos.

    Returns the saved file path when save, otherwise the Figure.
    '''
    import logomaker
    from sklearn.decomposition import PCA
    from encoding import convert_number
    from config import AA_20

    if df_ms is None:
        df_ms = pd.read_csv(IEDB_MS_CSV)

    def onehot(enc):
        N = len(enc)
        oh = np.zeros((N, L, Q_PEP))
        oh[np.arange(N)[:, None], np.arange(L)[None, :], enc] = 1.0
        return oh.reshape(N, L * Q_PEP)

    encA = convert_number(df_ms[df_ms['MHC allele'] == allele_A]['Peptide'].tolist())
    encB = convert_number(df_ms[df_ms['MHC allele'] == allele_B]['Peptide'].tolist())
    ohA, ohB = onehot(encA), onehot(encB)

    pca = PCA(n_components=n_pc, random_state=seed).fit(np.vstack([ohA, ohB]))
    PA, PB = pca.transform(ohA), pca.transform(ohB)
    comp = pca.components_.reshape(n_pc, L, Q_PEP)

    def proj_pca(v):
        return pca.transform(onehot(np.atleast_2d(np.asarray(v))))[:, :2]

    proj_spec, SA, SB, muA, nuB = _specificity_projection(rbm_A, rbm_B, encA, encB)

    if v0 is None or vT is None:
        from sampler import choose_endpoints_cluster
        v0, vT, _ = choose_endpoints_cluster(allele_A, allele_B, df_ms, seed=seed)

    fig = plt.figure(figsize=(14, 6.5 + 1.1 * n_pc))
    gs = fig.add_gridspec(1 + n_pc, 2,
                          height_ratios=[6.5] + [1.0] * n_pc, hspace=0.4,
                          top=0.96, bottom=0.05)

    panels = [
        (fig.add_subplot(gs[0, 0]), PA, PB, proj_pca, 'PC1', 'PC2'),
        (fig.add_subplot(gs[0, 1]), SA, SB, proj_spec,
         rf'$I_{{{muA}}}(v)$   ({allele_A} RBM)',
         rf'$I_{{{nuB}}}(v)$   ({allele_B} RBM)'),
    ]
    for k, (ax, XA, XB, proj, xlab, ylab) in enumerate(panels):
        ax.scatter(XA[:, 0], XA[:, 1], s=8, alpha=0.25, color='#5C82D6')
        ax.scatter(XB[:, 0], XB[:, 1], s=8, alpha=0.25, color='#EA9542')
        _overlay_paths(ax, paths, lambda p, proj=proj: proj(np.stack(p)),
                       n_paths=n_paths, alpha=path_alpha, seed=seed)
        ax.scatter(*proj(v0)[0], marker='*', s=320, color='#5C82D6',
                   edgecolor='k', linewidth=1.2, zorder=6)
        ax.scatter(*proj(vT)[0], marker='*', s=320, color='#EA9542',
                   edgecolor='k', linewidth=1.2, zorder=6)
        ax.set_xlabel(xlab, fontsize=_FS_LABEL)
        ax.set_ylabel(ylab, fontsize=_FS_LABEL)
        ax.tick_params(labelsize=_FS_TICK)
        ax.set_title(f'({"ab"[k]})', loc='left', fontsize=_FS_LABEL)

    fig.legend(handles=_scatter_legend(allele_A, allele_B, len(encA), len(encB),
                                       'v0 (A)', 'vT (B)', show_paths=bool(paths)),
               loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=5,
               fontsize=_FS_LEG, framealpha=0.9)

    ymax = np.abs(comp).max() * 1.05
    for k in range(n_pc):
        sub = gs[1 + k, :].subgridspec(1, 3, width_ratios=[1, 3, 1])  # centred so the logo stays narrow
        axl = fig.add_subplot(sub[0, 1])
        df_load = pd.DataFrame(comp[k], columns=AA_20)
        logomaker.Logo(df_load, ax=axl, color_scheme='chemistry', flip_below=False)
        axl.axhline(0, color='k', lw=0.6)
        axl.set_ylim(-ymax, ymax)
        axl.set_xticks(range(L))
        axl.set_xticklabels(range(1, L + 1), fontsize=_FS_TICK)
        axl.set_ylabel(f'PC{k + 1}', fontsize=_FS_LABEL)
        axl.tick_params(axis='y', labelsize=_FS_TICK)
        axl.set_title(f'({"cdefg"[k]})', loc='left', fontsize=_FS_LABEL)

    if save:
        fa = allele_A.replace('*', '').replace(':', '')
        fb = allele_B.replace('*', '').replace(':', '')
        out = os.path.join(FIGURES_DIR, f'case_study_{fa}_vs_{fb}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")
        return out
    return fig


def plot_energy_profile(samples, E_A_fn, E_B_fn, allele_A, allele_B,
                        df_ms=None, center=True, n_show=60, seed=0,
                        title=None, save=True):
    '''Plot the energy under each allele along the sampled paths.

    With center the mean energy over each allele's real ligands is subtracted,
    which puts the two curves on one scale. Returns the saved file path when
    save, otherwise the Figure.
    '''
    from encoding import convert_number

    if df_ms is None:
        df_ms = pd.read_csv(IEDB_MS_CSV)

    # every path has the same length so this stacks
    EA = np.array([E_A_fn(np.stack(p)) for p in samples])
    EB = np.array([E_B_fn(np.stack(p)) for p in samples])

    bA = bB = 0.0
    if center:
        encA = convert_number(df_ms[df_ms['MHC allele'] == allele_A]['Peptide'].tolist())
        encB = convert_number(df_ms[df_ms['MHC allele'] == allele_B]['Peptide'].tolist())
        bA = float(np.mean(E_A_fn(encA)))
        bB = float(np.mean(E_B_fn(encB)))
    EA, EB = EA - bA, EB - bB

    t = np.arange(EA.shape[1])  # path steps 0 to T
    mEA, mEB = EA.mean(0), EB.mean(0)

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in EA[:n_show]:
        ax.plot(t, row, color='#5C82D6', lw=0.5, alpha=0.12, zorder=2)
    for row in EB[:n_show]:
        ax.plot(t, row, color='#EA9542', lw=0.5, alpha=0.12, zorder=2)
    ax.plot(t, mEA, color='#5C82D6', lw=2.5, zorder=4, label=f'$E_A$  ({allele_A})')
    ax.plot(t, mEB, color='#EA9542', lw=2.5, zorder=4, label=f'$E_B$  ({allele_B})')

    d = mEA - mEB
    sign_change = np.where(np.diff(np.sign(d)))[0]
    if len(sign_change):
        i = sign_change[0]
        x_cross = i + d[i] / (d[i] - d[i + 1])  # interpolate between step i and i+1
        ax.axvline(x_cross, ls='--', color='0.4', lw=1.2, zorder=3,
                   label='specificity switch')

    ax.set_xlabel('path step $t$   ($v_0 \\to v_T$)', fontsize=_FS_LABEL)
    ax.set_ylabel(('centred ' if center else '') + 'energy  $-\\log p$',
                  fontsize=_FS_LABEL)
    ax.tick_params(labelsize=_FS_TICK)
    if title:
        ax.set_title(title, fontsize=_FS_LABEL)
    ax.legend(fontsize=_FS_LEG, framealpha=0.9)
    plt.tight_layout()

    if save:
        fa = allele_A.replace('*', '').replace(':', '')
        fb = allele_B.replace('*', '').replace(':', '')
        out = os.path.join(FIGURES_DIR, f'energy_profile_{fa}_vs_{fb}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")
        return out
    return fig


def plot_substitution_timing(samples, allele_A, allele_B, seed=0,
                             title=None, save=True):
    '''Box plot the step at which each differing position first changes.

    Returns the saved file path when save, otherwise the Figure.
    '''
    aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
          'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

    paths = [np.stack(p) for p in samples]
    if not paths:
        raise ValueError("no sampled paths")
    v0, vT = paths[0][0], paths[0][-1]
    diff_pos = np.where(v0 != vT)[0]
    if len(diff_pos) == 0:
        raise ValueError("v0 == vT: no substitutions to time")
    T = paths[0].shape[0] - 1

    # step at which each differing position first changes
    timing = {int(pos): [] for pos in diff_pos}
    for P in paths:
        for pos in diff_pos:
            col = P[:, pos]
            changes = np.where(col[1:] != col[:-1])[0]
            if len(changes):
                timing[int(pos)].append(int(changes[0]) + 1)

    order = sorted((int(p) for p in diff_pos), key=lambda p: np.median(timing[p]))
    data = [timing[p] for p in order]
    labels = [f'P{p + 1}: {aa[v0[p]]}$\\to${aa[vT[p]]}' for p in order]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(order) + 1.6))
    ypos = np.arange(len(order))[::-1]  # first in order at the top

    bp = ax.boxplot(data, positions=ypos, vert=False, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=1.5))
    for box in bp['boxes']:
        box.set(facecolor='#5C82D6', alpha=0.85, edgecolor='black', lw=0.8)
    for el in ('whiskers', 'caps'):
        for line in bp[el]:
            line.set(color='black', lw=0.8)

    rng = np.random.default_rng(seed)
    for y, d in zip(ypos, data):
        jit = y + (rng.random(len(d)) - 0.5) * 0.35
        ax.scatter(d, jit, s=6, color='black', alpha=0.12, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=_FS_TICK)
    ax.set_xlabel('path step $t$ of substitution   ($v_0 \\to v_T$)',
                  fontsize=_FS_LABEL)
    ax.set_xlim(0.5, T + 0.5)
    ax.set_xticks(np.arange(1, T + 1))
    ax.tick_params(labelsize=_FS_TICK)
    if title:
        ax.set_title(title, fontsize=_FS_LABEL)
    plt.tight_layout()

    if save:
        fa = allele_A.replace('*', '').replace(':', '')
        fb = allele_B.replace('*', '').replace(':', '')
        out = os.path.join(FIGURES_DIR, f'substitution_timing_{fa}_vs_{fb}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")
        return out
    return fig


def load_entropy_jsd():
    df_E = pd.read_csv(ENTROPY_CSV, index_col=0)
    df_jsd = pd.read_csv(JSD_CSV, index_col=0)
    print("Top 5 alleles by entropy:")
    print(df_E.sort_values('E_corr_pwm', ascending=False).head())
    print("Bottom 5 alleles by entropy:")
    print(df_E.sort_values('E_corr_pwm', ascending=True).head())
    print("Top 5 most similar allele pairs by JSD:")
    print(df_jsd.nsmallest(5, 'JSD_corr_pwm')[['HLA allele1', 'HLA allele2', 'JSD_corr_pwm']])
    print("Top 5 most different allele pairs by JSD:")
    print(df_jsd.nlargest(5, 'JSD_corr_pwm')[['HLA allele1', 'HLA allele2', 'JSD_corr_pwm']])
    return df_E, df_jsd


def plot_jsd_heatmap(df_jsd):
    alleles = pd.unique(df_jsd[['HLA allele1', 'HLA allele2']].values.ravel())
    mat = pd.DataFrame(np.nan, index=alleles, columns=alleles)
    for _, row in df_jsd.iterrows():
        mat.loc[row['HLA allele1'], row['HLA allele2']] = row['JSD_corr_pwm']
        mat.loc[row['HLA allele2'], row['HLA allele1']] = row['JSD_corr_pwm']
    np.fill_diagonal(mat.values, 0.0)

    short = {a: a.replace('HLA-', '') for a in alleles}
    mat.index = [short[a] for a in mat.index]
    mat.columns = [short[a] for a in mat.columns]

    g = sns.clustermap(
        mat, cmap='viridis_r', figsize=(14, 12),
        xticklabels=True, yticklabels=True,
        cbar_kws={'label': 'JSD (corrected, PWM)'},
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=5, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=5, rotation=0)
    g.fig.suptitle('Pairwise JSD between HLA alleles (hierarchical clustering)', y=1.01)

    out = os.path.join(FIGURES_DIR, 'jsd_heatmap.pdf')
    g.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_entropy_barplot(df_E, supertypes):
    merged = df_E.sort_values('E_corr_pwm', ascending=False)

    supers = merged['Supertype_new'].fillna('Unknown').unique()
    palette = dict(zip(supers, sns.color_palette('tab20', len(supers))))
    colors = merged['Supertype_new'].fillna('Unknown').map(palette)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(merged)), merged['E_corr_pwm'], color=colors, width=0.8)
    ax.set_xticks(range(len(merged)))
    ax.set_xticklabels(
        merged['HLA allele'].str.replace('HLA-', ''),
        rotation=90, fontsize=6
    )
    ax.set_ylabel('Entropy (nats, corrected PWM)')
    ax.set_title('Per-allele binding motif entropy, coloured by supertype')

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[s]) for s in supers]
    ax.legend(handles, supers, title='Supertype', bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=7)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'entropy_barplot.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def select_allele_pairs(df_jsd, df_E, n_high=5, n_low=5):
    '''Pick the highest and lowest JSD pairs among alleles with a model file.

    Returns the chosen pairs with their JSD and category.
    '''
    model_files = os.listdir(MODELS_DIR)

    def allele_to_model_key(allele):
        return allele.replace('*', '').replace(':', '')

    has_model = {
        a for a in pd.unique(df_jsd[['HLA allele1', 'HLA allele2']].values.ravel())
        if any(allele_to_model_key(a) in f for f in model_files)
    }
    print(f"{len(has_model)} alleles have a pretrained RBM model")

    pairs_df = df_jsd[
        df_jsd['HLA allele1'].isin(has_model) &
        df_jsd['HLA allele2'].isin(has_model) &
        (df_jsd['HLA allele1'] != df_jsd['HLA allele2'])
    ][['HLA allele1', 'HLA allele2', 'JSD_corr_pwm']].copy()
    pairs_df = pairs_df.rename(columns={'HLA allele1': 'source',
                                        'HLA allele2': 'target',
                                        'JSD_corr_pwm': 'jsd'})

    high_pairs = pairs_df.nlargest(n_high, 'jsd').copy()
    high_pairs['category'] = 'high_jsd'
    low_pairs = pairs_df.nsmallest(n_low, 'jsd').copy()
    low_pairs['category'] = 'low_jsd'

    selected = pd.concat([high_pairs, low_pairs], ignore_index=True)
    out = os.path.join(RESULTS_DIR, 'allele_pairs.csv')
    selected.to_csv(out, index=False)
    print(f"Saved: {out}")
    return selected


if __name__ == '__main__':
    df_ms, df_ba, supertypes = load_iedb_data()
    df_E, df_jsd = load_entropy_jsd()
    plot_jsd_heatmap(df_jsd)
    plot_entropy_barplot(df_E, supertypes)
    pairs = select_allele_pairs(df_jsd, df_E)
    print("Selected allele pairs:")
    print(pairs)
