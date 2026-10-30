# Chapter 3, what the RBM captures that a PWM cannot. Real ligands are scored
# against sequences generated from each model. A PWM treats the positions as
# independent, so its sequences sit at energies the real ligands do not reach,
# and that shift is the size of the correlations the RBM is picking up.

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_TICK
from entropy import gen_data_pwm, gen_data_rbm, convert_number

# PGM ships a module called rbm too and wins on sys.path, so load this one
# from its path rather than by name
_spec = importlib.util.spec_from_file_location("src_rbm", os.path.join(SRC, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

ALLELES = ['HLA-B*51:01', 'HLA-A*02:01']
N_GEN = 3000
SEED = 42
CACHE = os.path.join(FIGURES_DIR, 'pwm_vs_rbm_density.npz')

C_LIG = '#444444'
C_RBM = '#EA9542'
C_PWM = '#5C82D6'


def compute():
    '''Score real, RBM-generated and PWM-generated peptides under each RBM.'''
    df = pd.read_csv(IEDB_MS_CSV)
    store = {}
    for allele in ALLELES:
        np.random.seed(SEED)
        ligs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == 9]
        enc_lig = convert_number(ligs)
        rbm = load_rbm(allele)
        gen_pwm = gen_data_pwm(enc_lig, N=N_GEN)
        gen_rbm = gen_data_rbm(rbm, df_train=None, N=N_GEN, Nc=100)
        key = allele.replace('*', '').replace(':', '')
        store[f'{key}__lig'] = np.asarray(rbm_energy_batch(enc_lig, rbm))
        store[f'{key}__rbm'] = np.asarray(rbm_energy_batch(gen_rbm, rbm))
        store[f'{key}__pwm'] = np.asarray(rbm_energy_batch(gen_pwm, rbm))
        store[f'{key}__nh'] = np.array([int(rbm.n_h)])
        rbm.random_state = np.random.RandomState(SEED)
        rbm.AIS(M=100, n_betas=20000)
        store[f'{key}__logZ'] = np.array([float(rbm.log_Z_AIS)])
        print(f"  logZ[{allele}] = {rbm.log_Z_AIS:.3f} +/- {rbm.log_Z_AIS_std:.3f}")
    np.savez(CACHE, **store)
    print(f"Cached energies -> {CACHE}")


def ensure_logz():
    cache = dict(np.load(CACHE, allow_pickle=True))
    changed = False
    for allele in ALLELES:
        key = allele.replace('*', '').replace(':', '')
        if f'{key}__logZ' in cache:
            continue
        rbm = load_rbm(allele)
        rbm.random_state = np.random.RandomState(SEED)
        rbm.AIS(M=100, n_betas=20000)
        cache[f'{key}__logZ'] = np.array([float(rbm.log_Z_AIS)])
        print(f"  logZ[{allele}] = {rbm.log_Z_AIS:.3f} +/- {rbm.log_Z_AIS_std:.3f}")
        changed = True
    if changed:
        np.savez(CACHE, **cache)
        print(f"Augmented cache with logZ -> {CACHE}")
    return cache


def kde_curve(ax, vals, color, label, lw=2.2, fill=False):
    vals = np.asarray(vals, dtype=float)
    grid = np.linspace(vals.min() - 1, vals.max() + 1, 400)
    dens = gaussian_kde(vals)(grid)
    ax.plot(grid, dens, color=color, lw=lw, label=label)
    if fill:
        ax.fill_between(grid, dens, color=color, alpha=0.12)


def panel(ax, allele, cache, mode):
    '''Draw the density panel for one allele and return the PWM shift.'''
    key = allele.replace('*', '').replace(':', '')
    E_lig = np.asarray(cache[f'{key}__lig'], dtype=float)
    E_rbm = np.asarray(cache[f'{key}__rbm'], dtype=float)
    E_pwm = np.asarray(cache[f'{key}__pwm'], dtype=float)
    if mode == 'LL':
        logZ = float(cache[f'{key}__logZ'][0])  # log P(v) = -F(v) - logZ
        v_lig, v_rbm, v_pwm = -E_lig - logZ, -E_rbm - logZ, -E_pwm - logZ
        xlabel = 'RBM log-likelihood  (nats)'
    else:
        v_lig, v_rbm, v_pwm = E_lig, E_rbm, E_pwm
        xlabel = 'RBM effective energy  (nats)'
    m_rbm, m_pwm = float(np.mean(v_rbm)), float(np.mean(v_pwm))
    shift = m_pwm - m_rbm

    kde_curve(ax, v_lig, C_LIG, 'real ligands', lw=2.6)
    kde_curve(ax, v_rbm, C_RBM, 'RBM-generated', fill=True)
    kde_curve(ax, v_pwm, C_PWM, 'PWM-generated', fill=True)

    # the sampled gap and the reported Delta differ in the second digit
    ax.axvline(m_rbm, color=C_RBM, ls='--', lw=1.6, zorder=1)
    ax.axvline(m_pwm, color=C_PWM, ls='--', lw=1.6, zorder=1)
    lo, hi = sorted((m_rbm, m_pwm))
    ax.axvspan(lo, hi, color='0.35', alpha=0.16, zorder=0)  # keeps a small gap visible

    allv = np.concatenate([v_lig, v_rbm, v_pwm])
    ax.set_xlim(np.percentile(allv, 0.5), np.percentile(allv, 99.5))  # cuts the long thin tails

    ax.set_title(f'{allele}', fontsize=FS_TITLE)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel('density', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_TICK, loc='upper right', framealpha=0.9)
    return shift


# entry point that draws the density figure on both the energy and log-likelihood axes
def main():
    cache = ensure_logz()
    # both x axes use the same cached sample
    for mode, fname in [('E', 'pwm_vs_rbm_density.pdf'),
                        ('LL', 'pwm_vs_rbm_density_loglik.pdf')]:
        fig, axes = plt.subplots(1, len(ALLELES), figsize=(6.4 * len(ALLELES), 5.0))
        axes = np.atleast_1d(axes)
        for ax, allele in zip(axes, ALLELES):
            s = panel(ax, allele, cache, mode)
            print(f"[{mode}] {allele}: PWM-gen shift = {s:.2f} nats")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, fname)
        fig.savefig(out, bbox_inches='tight', dpi=200)
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Saved: {out}")


if __name__ == '__main__':
    if '--plot' not in sys.argv:
        compute()
    main()
