import pandas as pd

from config import L

FREQ_THRESH = 0.15  # fraction of an allele's ligands that carry the residue
P2, P9 = 1, L - 1  # counting from zero


# amino acid frequencies at a single position among one allele's ligands
def positional_freq(allele, df, pos):
    col = [s[pos] for s in df[df['MHC allele'] == allele]['Peptide']
           if isinstance(s, str) and len(s) == L]
    return pd.Series(col).value_counts(normalize=True)


# the residues an allele accepts at a position, those above the frequency threshold
def anchor_set(allele, df, pos, thresh=FREQ_THRESH):
    freq = positional_freq(allele, df, pos)
    return set(freq[freq >= thresh].index) or {freq.index[0]}
