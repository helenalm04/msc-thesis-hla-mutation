# Stage 2 of the Appendix calibration, run on its own under the MHCflurry
# environment: stage 1 of calibrate_presented_threshold.py writes the
# peptides, this scores them, and stage 3 reads the scores back.
#
#     python src/scans/_mhcflurry_score.py <in.csv> <out.csv>

import sys
import pandas as pd
from mhcflurry import Class1PresentationPredictor


def main(inp, outp):
    df = pd.read_csv(inp)
    pred = Class1PresentationPredictor.load()
    # older versions do not expose the allele list, so carry on without it
    try:
        supported = set(pred.affinity_predictor.supported_alleles)
    except Exception:
        supported = None

    # score one allele at a time, skipping any MHCflurry has no model for
    out = []
    for allele, g in df.groupby('allele'):
        peps = sorted(g['peptide'].unique().tolist())
        if supported is not None and allele not in supported:
            print(f"[skip] {allele} not supported by MHCflurry", flush=True)
            continue
        res = pred.predict(peptides=peps, alleles=[allele],
                           include_affinity_percentile=True, verbose=0)
        keep = pd.DataFrame({
            'peptide': res['peptide'].values,
            'allele': allele,
            'affinity': res['affinity'].values,
            'affinity_percentile': (res['affinity_percentile'].values
                                    if 'affinity_percentile' in res else float('nan')),
            'presentation_score': res['presentation_score'].values,
        })
        out.append(keep)
        print(f"[ok] {allele}: scored {len(peps)} peptides", flush=True)

    if not out:
        raise SystemExit("no alleles scored (none supported?)")
    pd.concat(out, ignore_index=True).to_csv(outp, index=False)
    print(f"wrote {outp}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
