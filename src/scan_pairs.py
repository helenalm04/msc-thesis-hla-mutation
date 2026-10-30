# Chapter 4, the scan across allele pairs. Pairs are spread evenly over the
# range of pairwise JSD rather than drawn at random, which would cluster them
# wherever that distribution is densest, and each gets one summary row.
#
#     python -u src/scan_pairs.py

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (IEDB_MS_CSV, ENTROPY_CSV, JSD_CSV,
                    RESULTS_DIR, MODELS_DIR)
# entropy imports PGM's rbm under that bare name, which is how the saved
# models unpickle, so this import is needed even where nothing here calls it
import entropy

# PGM ships a module called rbm too and wins on sys.path, so load this one
# from its path rather than by name
_spec = importlib.util.spec_from_file_location("src_rbm", os.path.join(HERE, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch

from sampler import (choose_endpoints_transition,
                     run_sampler, save_paths)
from encoding import convert_number
from path_stats import summarise_pair, barrier_stats


# mean model energy of an allele's own measured ligands, the baseline its energies are centred on
def _ligand_offset(allele, df_ms, E_fn):
    seqs = df_ms[df_ms['MHC allele'] == allele]['Peptide'].tolist()
    return float(np.mean(E_fn(convert_number(seqs))))


# the allele name in the compact form the model filenames use
def _model_key(allele):
    return allele.replace('HLA-', '').replace('*', '').replace(':', '')


# an allele is only usable if all three sources have it: a fitted model, its
# own measured ligands, and a row in the entropy table
def valid_alleles(df_ms, df_E):
    model_files = os.listdir(MODELS_DIR)
    has_model = {a for a in df_E['HLA allele']
                 if any(_model_key(a) in f for f in model_files)}
    has_ligands = set(df_ms['MHC allele'].unique())
    has_entropy = set(df_E['HLA allele'])
    valid = has_model & has_ligands & has_entropy
    print(f"{len(valid)} alleles usable (model + ligands + entropy)")
    return valid


def select_pairs_spread(df_jsd, valid, n_pairs=12, jsd_col='JSD_corr_rbm'):
    '''Pick the pairs closest to evenly spaced values of the JSD column.'''
    m = (df_jsd['HLA allele1'].isin(valid) &
         df_jsd['HLA allele2'].isin(valid) &
         (df_jsd['HLA allele1'] != df_jsd['HLA allele2']))
    cand = df_jsd.loc[m].dropna(subset=[jsd_col]).sort_values(jsd_col)
    cand = cand.reset_index(drop=True)
    if len(cand) == 0:
        raise RuntimeError("no candidate pairs with both alleles usable")

    vals = cand[jsd_col].to_numpy()
    targets = np.linspace(vals.min(), vals.max(), min(n_pairs, len(cand)))
    chosen, used = [], set()
    for t in targets:
        for k in np.argsort(np.abs(vals - t)):
            if k not in used:
                used.add(int(k))
                chosen.append(int(k))
                break
    sel = cand.iloc[sorted(chosen)].copy()
    print(f"{len(cand)} candidate pairs; selected {len(sel)} spanning "
          f"{jsd_col} = {sel[jsd_col].min():.3f}..{sel[jsd_col].max():.3f}")
    return sel


def run_scan(n_pairs=12, n_burn=1000, n_iter=30000, n_thin=60,
             seed=42, top_k=40, beta=1.0, beta_init=None,
             out_csv=None, score='both', slack=0, lam=0.35):
    '''Sample paths for each selected pair and write the summary table.

    Args:
        score (str): Joint score, 'both' or 'either'.
        slack (int): Steps added to the direct path length, which frees
            interior nodes and makes the paths indirect.
        lam (float): Elastic cost per mutation, read only when slack > 0.

    Returns:
        out (pd.DataFrame): One summary row for each sampled pair.
    '''
    df_ms = pd.read_csv(IEDB_MS_CSV)
    df_E = pd.read_csv(ENTROPY_CSV, index_col=0)
    df_jsd = pd.read_csv(JSD_CSV, index_col=0)

    valid = valid_alleles(df_ms, df_E)
    pairs = select_pairs_spread(df_jsd, valid, n_pairs=n_pairs)

    ent_pwm = dict(zip(df_E['HLA allele'], df_E['E_corr_pwm']))
    ent_rbm = dict(zip(df_E['HLA allele'], df_E['E_corr_rbm']))

    rbm_cache = {}

    def get_rbm(allele):
        if allele not in rbm_cache:
            rbm_cache[allele] = load_rbm(allele)
        return rbm_cache[allele]

    rows = []
    for _, r in pairs.iterrows():
        A, B = r['HLA allele1'], r['HLA allele2']
        tag = f"{A} -> {B}"
        t0 = time.time()
        try:
            rbm_A, rbm_B = get_rbm(A), get_rbm(B)
            E_A_fn = lambda s, mdl=rbm_A: rbm_energy_batch(np.atleast_2d(s), mdl)
            E_B_fn = lambda s, mdl=rbm_B: rbm_energy_batch(np.atleast_2d(s), mdl)

            v0, vT, T = choose_endpoints_transition(
                A, B, df_ms, E_A_fn, E_B_fn, top_k=top_k)

            T_use = None if slack == 0 else T + slack
            samples, energies, rate, info = run_sampler(
                v0, vT, E_A_fn, E_B_fn, beta=beta, beta_init=beta_init,
                score=score, T=T_use, Lambda=(0.0 if slack == 0 else lam),
                n_burn=n_burn, n_iter=n_iter, n_thin=n_thin, seed=seed)

            label = f"scan_{_model_key(A)}_{_model_key(B)}" + ("" if slack == 0 else "_g")
            save_paths(samples, energies, A, B, label, info=info)

            row = summarise_pair(samples, E_A_fn, E_B_fn, extra={
                'allele_A': A, 'allele_B': B, 'label': label,
                'jsd_pwm': float(r.get('JSD_corr_pwm', np.nan)),
                'jsd_rbm': float(r.get('JSD_corr_rbm', np.nan)),
                'entropy_A_pwm': ent_pwm.get(A), 'entropy_B_pwm': ent_pwm.get(B),
                'entropy_A_rbm': ent_rbm.get(A), 'entropy_B_rbm': ent_rbm.get(B),
                'accept_rate': float(rate),
            })
            row['entropy_mean_pwm'] = np.nanmean(
                [row['entropy_A_pwm'], row['entropy_B_pwm']])
            row['entropy_mean_rbm'] = np.nanmean(
                [row['entropy_A_rbm'], row['entropy_B_rbm']])

            off_A = _ligand_offset(A, df_ms, E_A_fn)
            off_B = _ligand_offset(B, df_ms, E_B_fn)
            cE_A = lambda s, o=off_A, mdl=rbm_A: rbm_energy_batch(np.atleast_2d(s), mdl) - o
            cE_B = lambda s, o=off_B, mdl=rbm_B: rbm_energy_batch(np.atleast_2d(s), mdl) - o
            cbar = barrier_stats(samples, cE_A, cE_B)  # barriers on the ligand centred energies
            row['worst_either_centered'] = cbar['worst_either_mean']
            row['barrier_both_centered'] = cbar['barrier_both_mean']
            rows.append(row)
            print(f"[ok] {tag}: T={T} S_path={row['S_path']:.2f} "
                  f"n_eff={row['n_eff']:.1f} accept={rate:.2f} "
                  f"({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"[skip] {tag}: {type(e).__name__}: {e}")

    out = pd.DataFrame(rows)
    out_csv = out_csv or os.path.join(RESULTS_DIR, 'path_stats.csv')
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out)} rows -> {out_csv}")
    return out


def main():
    p = argparse.ArgumentParser(description="Scan the sampler across allele pairs.")
    p.add_argument('--n-pairs', type=int, default=12)
    p.add_argument('--n-burn', type=int, default=1000)
    p.add_argument('--n-iter', type=int, default=30000)
    p.add_argument('--n-thin', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--beta-init', type=float, default=None)
    p.add_argument('--smoke', action='store_true',
                   help="tiny run (3 pairs, short chains) to check the pipeline")
    p.add_argument('--out-csv', type=str, default=None)
    p.add_argument('--score', type=str, default='both', choices=['both', 'either'])
    p.add_argument('--slack', type=int, default=0, help='global paths: T = D + slack (0 = direct)')
    p.add_argument('--lam', type=float, default=0.35, help='elastic penalty for global paths')
    args = p.parse_args()

    if args.smoke:
        args.n_pairs, args.n_burn, args.n_iter, args.n_thin = 3, 200, 1500, 50
        args.out_csv = args.out_csv or os.path.join(RESULTS_DIR, 'path_stats_smoke.csv')
        print("SMOKE MODE")

    run_scan(n_pairs=args.n_pairs, n_burn=args.n_burn, n_iter=args.n_iter,
             n_thin=args.n_thin, seed=args.seed, top_k=args.top_k,
             beta=args.beta, beta_init=args.beta_init, out_csv=args.out_csv,
             score=args.score, slack=args.slack, lam=args.lam)


if __name__ == '__main__':
    main()
