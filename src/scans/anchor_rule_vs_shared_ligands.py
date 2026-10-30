# Chapter 3, a check that the anchor rule is not simply restating which
# alleles share ligands. Two alleles with the same anchors will usually
# present some of the same peptides, so both are computed over the
# well-sampled alleles and put side by side.

import os
import sys
import itertools
from collections import defaultdict

import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, RESULTS_DIR, L
import anchors

MIN_LIGANDS = 1000  # the allele panel used in Chapter 3
OUT = os.path.join(RESULTS_DIR, 'anchor_rule_vs_shared.csv')


def load():
    '''Return the nonamer rows of the IEDB eluted ligand table.'''
    df = pd.read_csv(IEDB_MS_CSV)
    return df[df['Peptide'].apply(lambda s: isinstance(s, str) and len(s) == L)]


def main():
    df = load()
    counts = df['MHC allele'].value_counts()
    alleles = sorted(counts[counts >= MIN_LIGANDS].index)
    lig = {a: set(g['Peptide']) for a, g in df.groupby('MHC allele') if a in alleles}
    A2 = {a: anchors.anchor_set(a, df, anchors.P2) for a in alleles}
    A9 = {a: anchors.anchor_set(a, df, anchors.P9) for a in alleles}

    print(f"{len(alleles)} alleles with >= {MIN_LIGANDS} nonamers")
    for a, b, expect in (('HLA-B*51:01', 'HLA-B*27:05', 18),
                         ('HLA-B*51:01', 'HLA-B*27:07', 1)):
        got = len(lig[a] & lig[b])
        print(f"  check {a} / {b}: {got} shared (Chapter 6 reports {expect})"
              f"{'  OK' if got == expect else '  MISMATCH'}")

    rows = []
    for a, b in itertools.combinations(alleles, 2):
        p2, p9 = bool(A2[a] & A2[b]), bool(A9[a] & A9[b])
        S, T = lig[a], lig[b]
        rows.append(dict(a=a, b=b, p2_ok=p2, p9_ok=p9, compatible=p2 and p9,
                         shared=len(S & T), jaccard=len(S & T) / len(S | T)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)

    yes, no = t[t.compatible].jaccard, t[~t.compatible].jaccard
    u, p = mannwhitneyu(yes, no)
    print(f"\nthe binary rule against measured sharing, {len(t)} pairs")
    print(f"  compatible   n={len(yes):4d}  median Jaccard {yes.median():.4f}")
    print(f"  incompatible n={len(no):4d}  median Jaccard {no.median():.4f}")
    print(f"  ratio {yes.median()/no.median():.1f}x, Mann-Whitney p={p:.2e}")
    print(f"  measured overlap separates the two classes at AUC "
          f"{roc_auc_score(t.compatible, t.jaccard):.3f}")
    print("  (that is overlap discriminating the classes; as a PREDICTOR of overlap")
    print("   the binary rule scores far less, see the table below)")

    jsd = pd.read_csv(os.path.join(RESULTS_DIR, 'anchor_jsd_pairs.csv'))
    key = {frozenset((r.a, r.b)): (r.jaccard, float(r.compatible))
           for r in t.itertuples()}
    jsd = jsd.assign(**{k: v for k, v in zip(
        ('jaccard', 'binary'),
        zip(*[key.get(frozenset((r.allele_A, r.allele_B)), (float('nan'),) * 2)
              for r in jsd.itertuples()]))})
    jsd = jsd.dropna(subset=['jaccard'])
    hi = (jsd.jaccard > jsd.jaccard.median()).astype(int)
    print(f"\npredicting measured overlap, {len(jsd)} pairs")
    print(f"  {'predictor':<30}{'Spearman':>10}{'AUC':>8}")
    for name, col, sign in (('anchor JSD (P2, P9)', 'anchor_jsd', -1),
                            ('non-anchor JSD (other seven)', 'nonanchor_jsd', -1),
                            ('all-position JSD', 'pos_jsd', -1),
                            ('whole-landscape JSD (RBM)', 'jsd_table_rbm', -1),
                            ('binary anchor rule', 'binary', 1),
                            ('same supertype', 'same_supertype', 1)):
        x = jsd[col].astype(float)
        r, _ = spearmanr(x, jsd.jaccard)
        print(f"  {name:<30}{r:>+10.3f}{roc_auc_score(hi, sign * x):>8.3f}")
    print("  the anchor columns carry cross-presentation and the other seven do not,")
    print("  which is the design-side counterpart of the escape comparison.")

    # a flag firing on k of n pairs caps its AUC against a median split
    n, npos = len(jsd), int(hi.sum())
    print(f"\nthe binary rows against their ceiling, {n} pairs")
    print(f"  {'predictor':<22}{'fires':>7}{'AUC':>7}{'ceiling':>9}"
          f"{'precision':>11}{'recall':>8}")
    for name, col in (('binary anchor rule', 'binary'), ('same supertype', 'same_supertype')):
        f = jsd[col].astype(float).astype(int)
        k = int(f.sum())  # the ceiling assumes every pair it fires on sits in the upper half
        ceiling = (min(k, npos) / npos + 1 - max(0, k - npos) / (n - npos)) / 2
        print(f"  {name:<22}{k:>7}{roc_auc_score(hi, f):>7.3f}{ceiling:>9.3f}"
              f"{hi[f == 1].mean():>11.3f}{f[hi == 1].mean():>8.3f}")
    print("  both sit near their ceiling, so neither flag picks the wrong pairs; the")
    print("  rule beats the supertype label on reach at equal precision.")

    hi_t = (t.jaccard > t.jaccard.median()).astype(int)
    n_t, npos_t = len(t), int(hi_t.sum())
    print(f"\nthe cut as a precision-recall trade, {n_t} pairs")
    print(f"  {'cut':>5}{'fires':>7}{'AUC':>7}{'ceiling':>9}{'precision':>11}{'recall':>8}")
    for cut in (0.06, 0.10, 0.13, 0.15, 0.20, 0.25):  # the cut is set on position separation elsewhere
        s2 = {a: anchors.anchor_set(a, df, anchors.P2, cut) for a in alleles}
        s9 = {a: anchors.anchor_set(a, df, anchors.P9, cut) for a in alleles}
        f = pd.Series([bool(s2[a] & s2[b]) and bool(s9[a] & s9[b])
                       for a, b in itertools.combinations(alleles, 2)]).astype(int)
        k = int(f.sum())
        ceiling = (min(k, npos_t) / npos_t + 1 - max(0, k - npos_t) / (n_t - npos_t)) / 2
        print(f"  {cut:>5.2f}{k:>7}{roc_auc_score(hi_t, f):>7.3f}{ceiling:>9.3f}"
              f"{hi_t[f == 1].mean():>11.3f}{f[hi_t == 1].mean():>8.3f}")
    print("  lowering the cut buys recall and sells precision; a design filter is")
    print("  judged on precision, which is where the cut sits, but this describes")
    print("  the operating point and does not justify the value.")

    print("\nwhich condition decides")
    print(f"  P2 intersects, P9 does not (P9 decisive): {int((t.p2_ok & ~t.p9_ok).sum())}")
    print(f"  P9 intersects, P2 does not (P2 decisive): {int((t.p9_ok & ~t.p2_ok).sum())}")
    print(f"  both intersect (compatible):              {int(t.compatible.sum())}")

    print("\nlargest allele sets one 9-mer could satisfy at both anchors")
    by2, by9 = defaultdict(set), defaultdict(set)
    for a in alleles:
        for r in A2[a]:
            by2[r].add(a)
        for r in A9[a]:
            by9[r].add(a)
    best = sorted(((len(s2 & s9), r2, r9, sorted(s2 & s9))
                   for r2, s2 in by2.items() for r9, s9 in by9.items()
                   if len(s2 & s9) >= 2), reverse=True)
    for n, r2, r9, s in best[:5]:
        loci = sorted({x.replace('HLA-', '')[0] for x in s})
        print(f"  {n:2d} alleles  P2={r2} P9={r9}  loci {loci}")
        print("" + ", ".join(x.replace('HLA-', '') for x in s))
    print("\n  (anchor condition only; the good-binder condition is not applied here,"
          "\n   so these are upper bounds on what a single peptide can achieve)")
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
