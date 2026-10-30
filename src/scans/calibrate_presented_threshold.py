# Appendix, fixing the energy below which a peptide counts as presented. The
# models give an energy on an arbitrary scale, so it is anchored to something
# outside: each allele's ligands are scored against composition-matched
# background 9-mers, with MHCflurry as the external reference.
#
# MHCflurry pins an older numpy, so this runs in three stages. Both
# intermediate files ship, so stage1 and stage3 can be rerun alone.
#
#     python src/scans/calibrate_presented_threshold.py stage1
#     python src/scans/_mhcflurry_score.py <in> <out>      under MHCflurry
#     python src/scans/calibrate_presented_threshold.py stage3

import os
import sys
import importlib.util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, RESULTS_DIR
# entropy imports PGM's rbm under that bare name, which is how the saved
# models unpickle, so this import is needed even where nothing here calls it
import entropy
_spec = importlib.util.spec_from_file_location("src_rbm", os.path.join(SRC, "rbm.py"))
_src_rbm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_src_rbm)
load_rbm = _src_rbm.load_rbm
rbm_energy_batch = _src_rbm.rbm_energy_batch
from encoding import convert_number, convert_letter

ALLELES = ['HLA-B*51:01', 'HLA-B*27:05', 'HLA-A*02:01',
           'HLA-B*07:02', 'HLA-A*02:04', 'HLA-A*02:02']
N_LIG = 2500
N_BG = 5000  # random background 9-mers for each allele
SEED = 42
AA = 'ACDEFGHIKLMNPQRSTVWY'

IN_CSV = os.path.join(RESULTS_DIR, 'presented_calib_in.csv')
MHC_CSV = os.path.join(RESULTS_DIR, 'presented_calib_mhcflurry.csv')
OUT_CSV = os.path.join(RESULTS_DIR, 'presented_calib.csv')
ENERGY_CSV = os.path.join(RESULTS_DIR, 'presented_calib_energies.csv')


def background_composition(df_ms):
    '''Amino acid frequencies over all IEDB nonamers.'''
    enc = convert_number(df_ms['Peptide'].tolist())
    counts = np.bincount(enc.ravel(), minlength=20).astype(float)
    return counts / counts.sum()


def stage1():
    df_ms = pd.read_csv(IEDB_MS_CSV)
    rng = np.random.default_rng(SEED)
    p_bg = background_composition(df_ms)
    print("background composition (top 5): " + ", ".join(
        f"{AA[i]} {100 * p_bg[i]:.1f}%" for i in np.argsort(p_bg)[::-1][:5]))

    rows = []
    for allele in ALLELES:
        seqs = df_ms[df_ms['MHC allele'] == allele]['Peptide'].tolist()
        rbm = load_rbm(allele)
        E = lambda s, m=rbm: rbm_energy_batch(np.atleast_2d(s), m)

        enc_all = convert_number(seqs)
        offset = float(E(enc_all).mean())

        take = enc_all if len(enc_all) <= N_LIG else enc_all[
            rng.choice(len(enc_all), N_LIG, replace=False)]
        bg = rng.choice(20, size=(N_BG, enc_all.shape[1]), p=p_bg)

        for enc, kind in ((take, 'ligand'), (bg, 'background')):
            ec = E(enc) - offset
            for s, e in zip(convert_letter(enc), ec):
                rows.append((s, allele, kind, float(e)))
        print(f"{allele}: {len(take)} ligands + {N_BG} background, offset {offset:.3f}")

    df = pd.DataFrame(rows, columns=['peptide', 'allele', 'kind', 'Ec'])
    df = df.drop_duplicates(subset=['peptide', 'allele'])
    df.to_csv(ENERGY_CSV, index=False)
    df[['peptide', 'allele']].to_csv(IN_CSV, index=False)
    print(f"wrote {ENERGY_CSV} and {IN_CSV} ({len(df)} rows)")


def crossing(x, y, level):
    '''First x where the curve y crosses level.'''
    for i in range(1, len(y)):
        a, b = y[i - 1], y[i]
        if np.isnan(a) or np.isnan(b):
            continue
        if (a - level) * (b - level) <= 0 and a != b:
            t = (level - a) / (b - a)
            return x[i - 1] + t * (x[i] - x[i - 1])
    return float('nan')


def stage3():
    de = pd.read_csv(ENERGY_CSV)
    dm = pd.read_csv(MHC_CSV)
    df = de.merge(dm, on=['peptide', 'allele'], how='inner')
    print(f"joined {len(df)} peptides over {df['allele'].nunique()} alleles")

    out = []
    for allele, g in df.groupby('allele'):
        g = g.dropna(subset=['affinity_percentile', 'affinity'])
        edges = np.arange(-12, 12.01, 0.5)
        mids = 0.5 * (edges[:-1] + edges[1:])
        idx = np.digitize(g['Ec'].values, edges) - 1
        med_pct, med_aff = [], []
        for b in range(len(mids)):
            m = idx == b
            if m.sum() < 20:  # too few peptides to take a median
                med_pct.append(np.nan); med_aff.append(np.nan); continue
            med_pct.append(float(np.median(g['affinity_percentile'].values[m])))
            med_aff.append(float(np.median(g['affinity'].values[m])))
        med_pct, med_aff = np.array(med_pct), np.array(med_aff)

        near0 = g[np.abs(g['Ec']) < 0.25]
        pct_at_0 = float(np.median(near0['affinity_percentile'])) if len(near0) >= 20 else np.nan
        aff_at_0 = float(np.median(near0['affinity'])) if len(near0) >= 20 else np.nan

        # where the standard cuts land on the E^c scale
        ec_at_2 = crossing(mids, med_pct, 2.0)
        ec_at_05 = crossing(mids, med_pct, 0.5)
        ec_at_500 = crossing(mids, med_aff, 500.0)

        lig = g[g['kind'] == 'ligand']
        rho = g['Ec'].corr(np.log10(g['affinity']), method='spearman')

        out.append(dict(allele=allele, n=len(g),
                        pct_at_Ec0=pct_at_0, aff_nM_at_Ec0=aff_at_0,
                        Ec_at_pct2=ec_at_2, Ec_at_pct05=ec_at_05,
                        Ec_at_500nM=ec_at_500,
                        lig_frac_pct2=100 * (lig['affinity_percentile'] <= 2).mean(),
                        lig_frac_Ec0=100 * (lig['Ec'] < 0).mean(),
                        spearman_Ec_logaff=rho))

    res = pd.DataFrame(out).sort_values('allele')
    res.to_csv(OUT_CSV, index=False)
    pd.set_option('display.width', 200, 'display.max_columns', 20)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nwrote {OUT_CSV}")
    print("\nmedian over alleles:"
          f"%rank at E^c=0 = {res['pct_at_Ec0'].median():.2f}, "
          f"E^c at %rank 2 = {res['Ec_at_pct2'].median():.2f}, "
          f"E^c at %rank 0.5 = {res['Ec_at_pct05'].median():.2f}, "
          f"E^c at 500 nM = {res['Ec_at_500nM'].median():.2f}")


if __name__ == '__main__':
    {'stage1': stage1, 'stage3': stage3}[sys.argv[1]]()
