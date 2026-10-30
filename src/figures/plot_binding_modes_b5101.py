# Chapter 2, the two binding modes of B*51:01. The allele takes either
# proline or alanine at position 2 and its ligands split accordingly, so each
# group is drawn as its own logo, showing that the rest of the motif shifts
# with the anchor rather than the two being one motif with a loose anchor.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, gaussian_kde
from scipy.signal import find_peaks
import logomaker
# the saved models unpickle by module name, so PGM's rbm has to be the one
# registered under that name before any of them is loaded
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'PGM', 'source'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'PGM', 'utilities'))
import rbm as _pgm_rbm
sys.modules['rbm'] = _pgm_rbm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IEDB_MS_CSV, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from entropy import convert_number
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("src_rbm", os.path.join(os.path.dirname(os.path.dirname(__file__)), "rbm.py"))
_src_rbm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm


ALLELE = 'HLA-B*51:01'

CHEM_ORDER = ['hydrophilic/charged', 'hydrophobic']
CHEM_COLORS = {'hydrophilic/charged': '#0000CD', 'hydrophobic': '#E8730C'}


# chemical class of each peptide's P2 residue, proline counting as hydrophobic
def mode_at_p2(seqs):
    return np.array(['hydrophobic' if s[1] == 'P' else 'hydrophilic/charged'
                     for s in seqs])


rbm = load_rbm(ALLELE)
df_ms = pd.read_csv(IEDB_MS_CSV)
seqs = df_ms[df_ms['MHC allele'] == ALLELE]['Peptide'].tolist()
print(f"  N = {len(seqs)} peptides, M = {rbm.n_h} hidden units")

enc = convert_number(seqs).astype(np.int16)
I = rbm.input_hiddens(enc)  # (N, M)


# weights are indexed by hidden unit, position, amino acid
w_pos = np.linalg.norm(rbm.weights, axis=2)
dom_pos = np.argmax(w_pos, axis=1)
kurt = kurtosis(I, axis=0)  # negative means bimodal

for mu in range(rbm.n_h):
    print(f"  h_{mu}: dominant position = P{dom_pos[mu]+1}, kurtosis = {kurt[mu]:+.2f}")

best_mu = int(np.argmin(kurt))
best_pos = int(dom_pos[best_mu])
print(f"\nMost bimodal: h_{best_mu} anchored at P{best_pos+1} (kurtosis = {kurt[best_mu]:+.2f})")


M = rbm.n_h
fig, axes = plt.subplots(1, M, figsize=(3.4 * M, 3.6), sharey=False)
if M == 1:
    axes = [axes]

for mu, ax in enumerate(axes):
    chem = mode_at_p2(seqs)
    data_by_chem = [I[chem == c, mu] for c in CHEM_ORDER]
    colors = [CHEM_COLORS[c] for c in CHEM_ORDER]
    labels = [f'{c} (n={len(d)})' for c, d in zip(CHEM_ORDER, data_by_chem)]

    ax.hist(data_by_chem, bins=30, stacked=True, color=colors,
            label=labels, edgecolor='white', linewidth=0.3)

    marker = ' [bimodal]' if kurt[mu] < -0.5 else ''
    ax.set_title(f'$h_{{{mu}}}$  (P{dom_pos[mu]+1}){marker}', fontsize=FS_TITLE)
    ax.set_xlabel(rf'Input to $h_{{{mu}}}$', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    if mu == 0:
        ax.set_ylabel('# peptides', fontsize=FS_LABEL)
    ax.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.85)

plt.tight_layout()
out = os.path.join(FIGURES_DIR, 'binding_modes_b5101_all_units.pdf')
fig.savefig(out, bbox_inches='tight')
print(f"\nSaved grid: {out}")


AA20 = list('ACDEFGHIKLMNPQRSTVWY')


# letter heights for a logo: the information content of each column, shared
# out between its residues in proportion to how often they appear
def stratified_information_matrix(seqs_subset, L=9):
    '''Return the letter heights in bits at each position.'''
    counts = np.zeros((L, len(AA20)), dtype=int)
    aa2idx = {a: i for i, a in enumerate(AA20)}
    for s in seqs_subset:
        for pos, a in enumerate(s):
            if a in aa2idx:
                counts[pos, aa2idx[a]] += 1
    freq = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    eps = 1e-9
    H = -np.sum(freq * np.log2(freq + eps), axis=1)
    ic = np.log2(len(AA20)) - H  # 20 amino acids cap the ic at about 4.32 bits
    ic = np.clip(ic, 0, None)  # drop tiny negative values
    info = freq * ic[:, None]  # height of each letter at each position
    df = pd.DataFrame(info, columns=AA20)
    df.index = np.arange(1, L + 1)
    return df


def draw_logo(seqs_subset, ax, title):
    info_df = stratified_information_matrix(seqs_subset)
    logomaker.Logo(info_df, ax=ax, color_scheme='chemistry')
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_ylabel('bits', fontsize=FS_LABEL)
    ax.set_xlabel('Position', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(0, np.log2(len(AA20)) + 0.1)
    ax.set_xticks(np.arange(1, 10))


I_best = I[:, best_mu]
kde = gaussian_kde(I_best)
x_grid = np.linspace(I_best.min(), I_best.max(), 800)
y_grid = kde(x_grid)
peaks, _ = find_peaks(y_grid)
if len(peaks) >= 2:
    p1, p2 = peaks[np.argsort(y_grid[peaks])[-2:]]
    lo, hi = sorted((p1, p2))
    trough_idx = lo + int(np.argmin(y_grid[lo:hi + 1]))
    trough_x = float(x_grid[trough_idx])
else:
    trough_x = float(np.median(I_best))

mode_low_mask = I_best <  trough_x
mode_high_mask = I_best >= trough_x
seqs_low = [s for s, m in zip(seqs, mode_low_mask)  if m]
seqs_high = [s for s, m in zip(seqs, mode_high_mask) if m]

# share of a set of peptides carrying proline at the unit's dominant position
def pro_frac(ss): return sum(s[best_pos] == 'P' for s in ss) / max(len(ss), 1)
title_pro = '"hydrophobic" pattern'
title_ala = '"hydrophilic/charged" pattern'
if pro_frac(seqs_high) > pro_frac(seqs_low):
    label_low,  label_high = title_ala, title_pro
else:
    label_low,  label_high = title_pro, title_ala

print(f"\nMode split at I_{best_mu}(v) = {trough_x:+.2f}")
print(f"  low  mode (n={len(seqs_low)}):  Pro@P2 = {pro_frac(seqs_low) :.1%}  -> {label_low}")
print(f"  high mode (n={len(seqs_high)}): Pro@P2 = {pro_frac(seqs_high):.1%}  -> {label_high}")

fig2 = plt.figure(figsize=(11, 8))
gs = fig2.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.45, wspace=0.25)
ax_hist = fig2.add_subplot(gs[0, :])
ax_lo = fig2.add_subplot(gs[1, 0])
ax_hi = fig2.add_subplot(gs[1, 1])

# colour the histogram by the split used for the logos
low_is_pro = pro_frac(seqs_low) > pro_frac(seqs_high)
color_low = CHEM_COLORS['hydrophobic' if low_is_pro else 'hydrophilic/charged']
color_high = CHEM_COLORS['hydrophilic/charged' if low_is_pro else 'hydrophobic']

ax_hist.hist([I_best[mode_low_mask], I_best[mode_high_mask]], bins=40,
             stacked=True, color=[color_low, color_high],
             label=[f'{label_low} (n={len(seqs_low)})',
                    f'{label_high} (n={len(seqs_high)})'],
             edgecolor='white', linewidth=0.4)
ax_hist.axvline(trough_x, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
ax_hist.set_xlabel(rf'Input to $h_{{{best_mu}}}$', fontsize=FS_LABEL)
ax_hist.set_ylabel('# peptides', fontsize=FS_LABEL)
ax_hist.tick_params(labelsize=FS_TICK)
ax_hist.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.9)

draw_logo(seqs_low,  ax_lo, f'{label_low}  (n={len(seqs_low)})')
draw_logo(seqs_high, ax_hi, f'{label_high} (n={len(seqs_high)})')

out2 = os.path.join(FIGURES_DIR, f'binding_modes_b5101_h{best_mu}.pdf')
fig2.savefig(out2, bbox_inches='tight')
print(f"Saved zoom:  {out2}")
