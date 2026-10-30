IEDB database

retrieved from (2025/07/03)
https://www.iedb.org/database_export_v3.php


***MS conditions:
- linear peptide (type)
- homo sapiens (species)
- class I
- positive quality
- mass spectrometry (method)
- single allelic (evidence)

remove peptides with special symbols
- only start with "HLA-A/B/C*"
- only first part before "+"
- remove peptides with "X/Z"

output
- mhc_ligand_MS.csv (MHC allele, Class, Peptide, Length)
- mhc_ligand_9unique_MS.csv (MHC allele, Class, Peptide)
- mhc_ligand_count_MS.csv (MHC allele, count)

***BA conditions:
- linear peptide (type)
- homo sapiens (species)
- class I
- binding assay/ fluorescence/ radioactivity (method)
- measure units "nM" & affinity < 500
- 3 exclusions
- no constraints on quality/evidence (?)

output
- mhc_ligand_BA.csv (MHC allele, Class, Peptide, Length, Affinity)
- mhc_ligand_9unique_BA.csv (MHC allele, Class, Peptide)
- mhc_ligand_count_BA.csv (MHC allele, count)

***others
mhc_ligand_pre.csv
- homo sapiens, linear peptide, class I
