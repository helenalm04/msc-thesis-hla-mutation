# Chapter 5, the B*51:01 to B*27:05 case study: the PCA and specificity
# scatters, the energy profile, and the order the positions change in. The
# drawing itself lives in src/data_explore.py. The allele pair, the endpoints
# and the output label are read from TL_ALLELE_A, TL_ALLELE_B, TL_V0, TL_VT and
# TL_LABEL, so the same script covers the chapter's other pairs.

import os
import sys
import importlib.util

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV
# entropy imports PGM's rbm under that bare name, which is how the saved
# models unpickle, so this import is needed even where nothing here calls it
import entropy
# PGM ships a module called rbm too and wins on sys.path, so load this one
# from its path rather than by name
_spec = importlib.util.spec_from_file_location("src_rbm", os.path.join(SRC, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

from sampler import choose_endpoints_motif, run_sampler, save_paths
from encoding import convert_number, convert_letter
from data_explore import (plot_case_study_figure,
                          plot_energy_profile, plot_substitution_timing)


ALLELE_A = os.environ.get('TL_ALLELE_A', 'HLA-B*51:01')
ALLELE_B = os.environ.get('TL_ALLELE_B', 'HLA-B*27:05')
V0_FIX = os.environ.get('TL_V0', 'LPFEPVPVI')
VT_FIX = os.environ.get('TL_VT', 'SRLPFLFTL')  # pair 0 of the b51_either_sweep bulk run
SCORE = os.environ.get('TL_SCORE', 'either')
SEED = 42
N_BURN = 1000
N_ITER = 30000
N_THIN = 60
LABEL = os.environ.get('TL_LABEL', 'transition_B5101_B2705')


def main():
    df_ms = pd.read_csv(IEDB_MS_CSV)

    rbm_A, rbm_B = load_rbm(ALLELE_A), load_rbm(ALLELE_B)
    E_A = lambda s: rbm_energy_batch(s, rbm_A)
    E_B = lambda s: rbm_energy_batch(s, rbm_B)

    # leave TL_V0/TL_VT empty to pick the endpoints instead
    if V0_FIX and VT_FIX:
        v0, vT = convert_number([V0_FIX])[0], convert_number([VT_FIX])[0]
        T = int((v0 != vT).sum())  # positions where the two endpoints differ
    else:
        v0, vT, T = choose_endpoints_motif(ALLELE_A, ALLELE_B, df_ms, E_A, E_B,
                                           seed=SEED)
    print(f"v0={convert_letter(v0.reshape(1, -1))[0]}  "
          f"vT={convert_letter(vT.reshape(1, -1))[0]}  D={T}  score={SCORE}")
    samples, energies, rate, info = run_sampler(
        v0, vT, E_A, E_B, score=SCORE,
        n_burn=N_BURN, n_iter=N_ITER, n_thin=N_THIN, seed=SEED)
    print(f"T={T}  paths={len(samples)}  change_rate={rate:.3f}")
    save_paths(samples, energies, ALLELE_A, ALLELE_B, LABEL, info=info)

    # the same endpoints for every figure below
    plot_case_study_figure(ALLELE_A, ALLELE_B, rbm_A, rbm_B, df_ms,
                           v0=v0, vT=vT, paths=samples,
                           n_paths=20, path_alpha=0.7, n_pc=2, seed=SEED)

    plot_energy_profile(samples, E_A, E_B, ALLELE_A, ALLELE_B, df_ms,
                        center=True, seed=SEED)

    plot_substitution_timing(samples, ALLELE_A, ALLELE_B, seed=SEED)


if __name__ == '__main__':
    main()
