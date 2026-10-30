# Encoding peptides as integers and back. The alphabet is the twenty amino
# acids alphabetically plus a gap, so a residue's code is its index here.

import numpy as np

curr_int = np.int16

def convert_number(seqs):
	'''Encode each sequence letter as its index in the amino acid alphabet.'''
	aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y','-']
	aadict = {aa[k]: k for k in range(len(aa))}

	msa_num = np.array(list(map(lambda x: [aadict[y] for y in x], seqs[0:])), dtype=curr_int, order="c")
	return msa_num

# turn encoded sequences back into letters, one string per sequence
def convert_letter(seqs_n):
	aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y','-']
	aadictinv = {k: aa[k] for k in range(len(aa))}
	seqs = []
	# a single sequence comes in as a 1D array of residues, a set of them as 2D
	if type(seqs_n[0]) == curr_int:
		seqs.append(''.join([aadictinv[e] for e in seqs_n]))
	else:
		for t in range(len(seqs_n)):
			seqs.append(''.join([aadictinv[e] for e in seqs_n[t]]))
	return seqs
