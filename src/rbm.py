# Loading the pretrained RBMs and scoring peptides with them. The models were
# supplied already fitted, one per allele; nothing here trains them.

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logomaker

sys.path.insert(0, os.path.dirname(__file__))
from config import MODELS_DIR, IEDB_MS_CSV, FIGURES_DIR, AA_20, AA_DICT, L, Q_PEP, ANCHOR_POSITIONS

import RBM_utils  # from PGM, put on the path by config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from entropy import gen_data_rbm, convert_number, pwm_train


# every allele with a model file, keyed by its standard name
def list_available_alleles():
    alleles = {}
    for path in glob.glob(os.path.join(MODELS_DIR, 'model_HLA-*.data')):
        fname = os.path.basename(path)
        match = re.search(r'model_(HLA-[ABC]\d{4})', fname)
        if not match:
            continue
        # the filenames run the digits together, so HLA-B5101 becomes
        # HLA-B*51:01, which is how the IEDB tables spell it
        compact = match.group(1)
        gene = compact[4]
        digits = compact[5:]
        standard = f'HLA-{gene}*{digits[:2]}:{digits[2:]}'
        alleles[standard] = path
    return alleles


def get_model_path(allele):
    alleles = list_available_alleles()
    if allele not in alleles:
        raise FileNotFoundError(
            f"No pretrained RBM for {allele}. Available: {sorted(alleles.keys())}"
        )
    return alleles[allele]


def load_rbm(allele):
    path = get_model_path(allele)
    rbm = RBM_utils.loadRBM(path)
    print(f"Loaded RBM for {allele}: n_h={rbm.n_h}, n_v={rbm.n_v}")
    return rbm


# effective energy of each peptide in an encoded batch, the negative log probability of the peptide up to a constant
def rbm_energy_batch(seqs_encoded, rbm):
    return rbm.free_energy(seqs_encoded.astype(np.int16))


def compare_logos(rbm, allele, df_ms=None):
    '''Plot the ligand and generated sequence logos for one allele.'''
    if df_ms is None:
        df_ms = pd.read_csv(IEDB_MS_CSV)

    seqs = df_ms[df_ms['MHC allele'] == allele]['Peptide'].tolist()
    pwm_data = pwm_train(convert_number(seqs))

    gen_seqs = gen_data_rbm(rbm, df_train=None, N=len(seqs), Nc=100)
    pwm_gen = pwm_train(gen_seqs)

    def ic_df(pwm):
        '''Scale each row of the matrix by its information content.'''
        pwm_df = pd.DataFrame(pwm, columns=AA_20)
        ic = np.log2(Q_PEP) + np.sum(pwm * np.log2(np.where(pwm > 0, pwm, 1)), axis=1)
        return pwm_df.multiply(ic, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 3), sharey=True)
    for ax, mat, title in zip(
        axes,
        [pwm_data, pwm_gen],
        [f'Training data  (N={len(seqs)})', f'RBM generated  (N={len(seqs)})']
    ):
        for pos in ANCHOR_POSITIONS:
            ax.axvspan(pos - 0.5, pos + 0.5, color='lightyellow', zorder=0)
        logomaker.Logo(ic_df(mat), ax=ax, color_scheme='chemistry')
        ax.set_xticks(range(L))
        ax.set_xticklabels([f'P{i+1}' for i in range(L)])
        ax.set_ylabel('Information (bits)')
        ax.set_title(title)

    fig.suptitle(f'Binding motif — {allele}  (RBM quality check)', y=1.02)
    plt.tight_layout()

    fname = allele.replace('*', '').replace(':', '')
    out = os.path.join(FIGURES_DIR, f'rbm_logo_{fname}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    print("Available alleles with pretrained RBMs:")
    for a, path in sorted(list_available_alleles().items()):
        print(f"  {a}  ->  {os.path.basename(path)}")

    allele = 'HLA-A*02:01'
    rbm = load_rbm(allele)

    test_peptide = 'GILGFVFTL'  # A*02:01 binder from influenza M1
    seq_enc = np.array([AA_DICT[aa] for aa in test_peptide])
    print(f"\nEnergy of {test_peptide}: {rbm_energy_batch(seq_enc.reshape(1, -1), rbm)[0]:.4f}")

    gen = gen_data_rbm(rbm, df_train=None, N=100, Nc=100)
    print(f"Generated sequences shape: {gen.shape}")

    compare_logos(rbm, allele)
