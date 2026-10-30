# The position weight matrix baseline. A PWM takes each position of the
# peptide independently, so it has an allele's motif but none of the
# correlations between positions, which is what Chapter 3 compares the RBM
# against. entropy.py does the fitting; this wraps it and draws logos.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logomaker

sys.path.insert(0, os.path.dirname(__file__))
from config import IEDB_MS_CSV, FIGURES_DIR, AA_20, L, Q_PEP, ANCHOR_POSITIONS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from entropy import convert_number, pwm_train, entropy_pwm


# negative log-likelihood of each encoded peptide under the position weight matrix, its independent-site energy
def pwm_energy_batch(seqs_encoded, pwm):
    return -np.sum(np.log(pwm[np.arange(L), seqs_encoded]), axis=1)


# draw one allele's PWM as a sequence logo, with the anchor columns shaded
def plot_pwm_logo(pwm, allele_name, save=True):
    pwm_df = pd.DataFrame(pwm, columns=AA_20)
    # information content at each position in bits, the uniform entropy minus
    # the entropy of the column, which is the height of the stack in the logo
    ic = np.log2(Q_PEP) + np.sum(pwm * np.log2(np.where(pwm > 0, pwm, 1)), axis=1)
    # scale each letter by the information content of its column
    ic_df = pwm_df.multiply(ic, axis=0)

    fig, ax = plt.subplots(figsize=(10, 3))
    for pos in ANCHOR_POSITIONS:
        ax.axvspan(pos - 0.5, pos + 0.5, color='lightyellow', zorder=0)
    logomaker.Logo(ic_df, ax=ax, color_scheme='chemistry')
    ax.set_xticks(range(L))
    ax.set_xticklabels([f'P{i+1}' for i in range(L)])
    ax.set_ylabel('Information (bits)')
    ax.set_title(f'Binding motif — {allele_name}  (PWM, MS data)')
    for pos in ANCHOR_POSITIONS:
        ax.text(pos, ax.get_ylim()[1] * 0.95, 'anchor',
                ha='center', va='top', fontsize=7, color='darkred')

    plt.tight_layout()
    if save:
        fname = allele_name.replace('*', '').replace(':', '')
        out = os.path.join(FIGURES_DIR, f'pwm_logo_{fname}.pdf')
        fig.savefig(out, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")
    else:
        plt.show()


def get_pwm_for_allele(allele, df_ms=None):
    '''Build a PWM from one allele's peptides and return it with the encoded sequences.'''
    if df_ms is None:
        df_ms = pd.read_csv(IEDB_MS_CSV)
    seqs = df_ms[df_ms['MHC allele'] == allele]['Peptide'].tolist()
    if len(seqs) == 0:
        raise ValueError(f"No peptides found for allele {allele}")
    print(f"{allele}: {len(seqs)} training peptides")
    seqs_encoded = convert_number(seqs)
    pwm = pwm_train(seqs_encoded)
    return pwm, seqs_encoded


if __name__ == '__main__':
    allele = 'HLA-B*15:02'
    pwm, seqs = get_pwm_for_allele(allele)
    print(f"PWM shape: {pwm.shape}")
    print(f"Training sequences: {len(seqs)}")
    E_pwm, E_anl, var_anl = entropy_pwm(seqs, ifprint=True)
    plot_pwm_logo(pwm, allele)
