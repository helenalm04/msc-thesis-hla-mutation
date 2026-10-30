# Chapter 6, the table behind the detour correlation. The numbers were already
# printed by src/figures/mean_profile.py, so rather than resampling anything
# this reads them back out of that run's log and joins them to the JSD table.
# Run mean_profile.py first, keeping its output as the log this reads.

import os
import re
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

LOG = os.path.join(RESULTS_DIR, 'mean_profile_full.log')
OUT = os.path.join(RESULTS_DIR, 'detour_vs_overlap.csv')

LINE = re.compile(  # "[ok] A1101->A1102: direct peak -5.36  global peak -5.58"
    r"\[ok\]\s+(\w+)->(\w+):\s+direct peak\s+([+-]?[\d.]+)\s+global peak\s+([+-]?[\d.]+)")


def expand(short):
    '''A1101 -> HLA-A*11:01.'''
    m = re.fullmatch(r"([ABC])(\d{2})(\d{2})", short)
    if not m:
        raise ValueError(f"cannot expand allele name {short!r}")
    return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"


# parse the per-pair lines out of the log, ignoring the endpoints-mode run
def main():
    rows, in_pairs = [], False
    with open(LOG) as fh:
        for line in fh:
            if line.startswith('=== mode='):
                in_pairs = 'mode=pairs' in line  # the log also has a mode=endpoints run
                continue
            m = LINE.search(line)
            if m and in_pairs:
                a, b, d, g = m.groups()
                rows.append({'a': expand(a), 'b': expand(b),
                             'direct_peak': float(d), 'global_peak': float(g)})
    df = pd.DataFrame(rows)
    assert len(df) == df.drop_duplicates(subset=['a', 'b']).shape[0], "duplicate pairs"
    df['gap'] = df['direct_peak'] - df['global_peak']
    print(f"parsed {len(df)} pairs from {os.path.basename(LOG)}")
    print(f"mean gap {df['gap'].mean():+.3f} nats, "
          f"detour lower in {(df['gap'] > 0).sum()} of {len(df)}   "
          f"(Section 6.4 reports +0.42 and 29 of 40)")

    def join_sym(df, path, keycols, valcols):
        '''Join a pairwise table written in either allele order.'''
        t = pd.read_csv(path)
        t = t.rename(columns=dict(zip(keycols, ['a', 'b'])))
        both = pd.concat([t, t.rename(columns={'a': 'b', 'b': 'a'})], ignore_index=True)
        both = both.drop_duplicates(subset=['a', 'b'], keep='first')
        return df.merge(both[['a', 'b'] + valcols], on=['a', 'b'], how='left')

    # jsd_table_rbm is the column the run selected on
    df = join_sym(df, os.path.join(RESULTS_DIR, 'anchor_jsd_pairs.csv'),
                  ['allele_A', 'allele_B'], ['jsd_table_rbm', 'same_supertype'])
    df = df.rename(columns={'jsd_table_rbm': 'jsd'})
    df = join_sym(df, os.path.join(RESULTS_DIR, 'anchor_rule_vs_shared.csv'),
                  ['a', 'b'], ['shared', 'jaccard'])

    print("\ncoverage: " + ", ".join(
        f"{c} {df[c].notna().sum()}/{len(df)}" for c in ('jsd', 'shared', 'jaccard')))

    print("\ndetour benefit against overlap")
    for col, direction in [('jsd', 'higher = more divergent'),
                           ('shared', 'higher = more overlap'),
                           ('jaccard', 'higher = more overlap')]:
        sub = df[['gap', col]].dropna()
        if len(sub) < 5:
            print(f"{col:8s} too few pairs ({len(sub)})")
            continue
        rs, ps = stats.spearmanr(sub[col], sub['gap'])
        rp, pp = stats.pearsonr(sub[col], sub['gap'])
        print(f"{col:8s} n={len(sub):2d}  Spearman r_s={rs:+.3f} (p={ps:.3f})   "
              f"Pearson r={rp:+.3f} (p={pp:.3f})   [{direction}]")

    print("\nmedian split")
    for col in ('jsd', 'shared', 'jaccard'):
        sub = df[['gap', col]].dropna()
        if len(sub) < 5:
            continue
        med = sub[col].median()
        lo, hi = sub[sub[col] <= med]['gap'], sub[sub[col] > med]['gap']
        print(f"{col:8s} median {med:.4g}   "
              f"low {lo.mean():+.3f} +/- {lo.sem():.3f} (n={len(lo)})   "
              f"high {hi.mean():+.3f} +/- {hi.sem():.3f} (n={len(hi)})   "
              f"Mann-Whitney p={stats.mannwhitneyu(lo, hi).pvalue:.3f}")

    df.sort_values('gap', ascending=False).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
