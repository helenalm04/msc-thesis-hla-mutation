# Every path the project uses and the constants shared across it. Importing
# this is what puts PGM on sys.path, so it has to come before entropy.py or
# anything that loads a model. Paths derive from the repo root, so the tree
# runs wherever it is unpacked and nothing here needs editing.

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PGM_DIR = os.path.join(ROOT, 'libs', 'PGM')
PGM_SOURCE = os.path.join(PGM_DIR, 'source')
PGM_UTILS = os.path.join(PGM_DIR, 'utilities')

sys.path.insert(0, PGM_SOURCE)
sys.path.insert(0, PGM_UTILS)

DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
IEDB_DIR = os.path.join(DATA_DIR, 'iedb')

IEDB_MS_CSV = os.path.join(IEDB_DIR, 'mhc_ligand_9unique_MS.csv')
IEDB_BA_CSV = os.path.join(IEDB_DIR, 'mhc_ligand_9unique_BA.csv')
ALLELE_SUPERTYPES = os.path.join(IEDB_DIR, 'allele_supertypes.csv')

JSD_CSV = os.path.join(DATA_DIR, 'df_jsd_total.csv')
ENTROPY_CSV = os.path.join(DATA_DIR, 'df_E_total.csv')

RESULTS_DIR = os.path.join(ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
PATHS_DIR = os.path.join(RESULTS_DIR, 'paths')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PATHS_DIR,   exist_ok=True)

AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
AA_20 = AA[:20]
Q = len(AA)
Q_PEP = 20
L = 9

AA_DICT = {aa: i for i, aa in enumerate(AA)}

# the two positions that hold a 9-mer in the binding groove, and so the ones
# an allele's motif is most particular about
ANCHOR_POSITIONS = [1, 8]  # p2 and p9, counting from zero
NON_ANCHOR_POSITIONS = [i for i in range(L) if i not in ANCHOR_POSITIONS]

FS_LABEL = 15  # large so shrunk figures stay readable
FS_TITLE = 15
FS_LEGEND = 15
FS_TICK = 13
