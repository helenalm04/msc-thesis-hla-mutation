# Chapter 6 robustness check on the detour correlation. Distant alleles tend
# to need longer paths, so the correlation could really be about length. This
# holds the distance fixed by partial correlation and bootstraps the result,
# so the headline figure can be read against the nominal p-value.

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

RNG = np.random.default_rng(0)
NBOOT = 20000


# position of each value in the sorted sample, ties taking their average position
def rank(x):
    return stats.rankdata(x)


def partial_spearman(x, y, z):
    '''Spearman of x and y with z held fixed.'''
    # regress the ranks of both on the ranks of z, then correlate what is
    # left over, which is the correlation with z's influence taken out
    rx, ry, rz = rank(x), rank(y), rank(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


def main():
    d = pd.read_csv(os.path.join(RESULTS_DIR, 'detour_vs_overlap.csv'))
    n = len(d)
    rs0 = stats.spearmanr(d.jsd, d.gap)[0]
    print(f"n = {n} pairs.  headline: Spearman r_s = {rs0:+.3f} "
          f"(nominal p = {stats.spearmanr(d.jsd, d.gap)[1]:.3f})\n")

    print("1. IS IT JUST 'YOU CAN ONLY RELIEVE A BARRIER THAT EXISTS'?")
    r_bar = stats.spearmanr(d.jsd, d.direct_peak)
    print(f"   JSD vs direct barrier height: r_s = {r_bar[0]:+.3f} (p = {r_bar[1]:.2g})")
    r_gb = stats.spearmanr(d.direct_peak, d.gap)
    print(f"   direct barrier vs benefit:    r_s = {r_gb[0]:+.3f} (p = {r_gb[1]:.2g})")
    pr, pp = partial_spearman(d.jsd, d.gap, d.direct_peak)
    print(f"   >> JSD vs benefit, HOLDING THE BARRIER FIXED: r = {pr:+.3f} (p = {pp:.3f})")
    pr2, pp2 = partial_spearman(d.direct_peak, d.gap, d.jsd)  # the same test the other way round
    print(f"      barrier vs benefit, holding JSD fixed:     r = {pr2:+.3f} (p = {pp2:.3f})")

    print("\n2. THE PAIRS SHARE ALLELES, SO RESAMPLE ALLELES NOT PAIRS")
    alleles = sorted(set(d.a) | set(d.b))
    print(f"   {len(alleles)} alleles over {n} pairs")
    boot = []
    for _ in range(NBOOT):
        keep = set(RNG.choice(alleles, size=len(alleles), replace=True))
        s = d[d.a.isin(keep) & d.b.isin(keep)]
        if len(s) > 8 and s.jsd.nunique() > 3:
            boot.append(stats.spearmanr(s.jsd, s.gap)[0])
    boot = np.array(boot)
    print(f"   cluster bootstrap over alleles: median r_s = {np.median(boot):+.3f}, "
          f"95% CI [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}]")
    print(f"   fraction of bootstrap replicates with r_s <= 0: {np.mean(boot <= 0):.3f}")

    perm = np.array([stats.spearmanr(d.jsd, RNG.permutation(d.gap.values))[0]
                     for _ in range(NBOOT)])
    print(f"   permutation test (shuffle benefit): p = {np.mean(perm >= rs0):.4f}")

    print("\n3. HOW WIDE IS THE ESTIMATE?")
    pb = np.array([stats.spearmanr(*d.iloc[RNG.integers(0, n, n)][['jsd','gap']].values.T)[0]
                   for _ in range(NBOOT)])
    pb = pb[~np.isnan(pb)]
    print(f"   pair bootstrap: 95% CI [{np.percentile(pb,2.5):+.3f}, "
          f"{np.percentile(pb,97.5):+.3f}]")

    print("\n4. DOES JSD BEAT THE BINARY SUPERTYPE LABEL?")
    pr3, pp3 = partial_spearman(d.jsd, d.gap, d.same_supertype)
    print(f"   JSD vs benefit, controlling for same/cross supertype: r = {pr3:+.3f} (p = {pp3:.3f})")
    for v, g in d.groupby('same_supertype'):
        r = stats.spearmanr(g.jsd, g.gap)
        print(f"   within same_supertype={int(v)} (n={len(g):2d}): r_s = {r[0]:+.3f} (p = {r[1]:.3f})")

    print("\n5. TREND OR STEP? (quartiles of JSD)")
    d['q'] = pd.qcut(d.jsd, 4, labels=['Q1 low', 'Q2', 'Q3', 'Q4 high'])
    for q, g in d.groupby('q', observed=True):
        print(f"   {q}: JSD {g.jsd.min():.2f}-{g.jsd.max():.2f}  "
              f"benefit {g.gap.mean():+.3f} +/- {g.gap.sem():.3f}  (n={len(g)})")

    print("\n6. INFLUENCE")
    loo = [stats.spearmanr(d.drop(i).jsd, d.drop(i).gap)[0] for i in d.index]
    print(f"   jackknife r_s range: [{min(loo):+.3f}, {max(loo):+.3f}]")
    for k in (1, 2, 3, 4):
        sub = d.drop(d.nlargest(k, 'gap').index)
        r = stats.spearmanr(sub.jsd, sub.gap)
        print(f"   drop top {k} benefit(s): r_s = {r[0]:+.3f} (p = {r[1]:.3f}), n = {len(sub)}")


if __name__ == '__main__':
    main()
