# Mutational paths between HLA-I binding landscapes

Code for my MSc thesis *Mutational paths between HLA-I binding landscapes*. RBM
landscapes for 68 HLA-I alleles, a Markov chain sampler over paths in peptide
space, and the scans and plotting scripts behind the results chapters.

## Layout

    src/            landscapes, path sampler, path statistics, mean field
    src/scans/      one script per experiment, each writing a CSV or NPZ to results/
    src/figures/    plotting, reads results/, writes results/figures/
    src/b51/        the B*51:01 case study
    tests/          pytest suite
    entropy.py      not mine, see Attribution
    data/           IEDB mass-spec ligands, allele supertypes, and two supplied tables
    results/        the summary tables and caches the figure scripts read

`src/config.py` derives every path from the repo root, so the tree runs wherever
it is unpacked and nothing needs editing.

## Data

`data/` holds what the analyses here read:

    proteome.txt                    human proteome, the background for random
                                    nonamers, from Yinfei Yang
    iedb/mhc_ligand_9unique_MS.csv  IEDB mass-spec ligands, the PWM and anchor-rule input
    iedb/mhc_ligand_9unique_BA.csv  IEDB binding-affinity table
    iedb/allele_supertypes.csv      supertype assignments
    iedb/000README.txt              the IEDB query that produced the export

Two further tables are read but not included here, for the same reason as
the models, they were supplied to me rather than produced by this code.

    df_jsd_total.csv                pairwise Jensen-Shannon divergences
    df_E_total.csv                  per-allele entropies

Drop them into `data/` to run the scans that read them. The tests do not need
them, and neither does anything under `src/figures/` that reads only `results/`.

## The RBM models

The analyses read 68 allele-specific RBMs that were supplied to me already
trained.

## Environment

Python 3.12.

    pip install -r requirements.txt

numba is in there because the PGM engine needs it, not because anything here
imports it directly, but the RBM code will not load without it.

### PGM

The RBM effective energies come from Jerome Tubiana's PGM engine (Tubiana, Cocco
and Monasson 2019):

    git clone https://github.com/jertubiana/PGM.git libs/PGM

The Appendix calibration also needs MHCflurry 2.2.1, which pins an older numpy and
so wants its own environment. It runs in three stages: `stage1`, then
`_mhcflurry_score.py` under MHCflurry, then `stage3`. Both intermediate files ship.

## Tests

    python -m pytest tests/

The tests cover the three things my thesis argues for: that the sampler
reproduces the exact Boltzmann distribution, that the convergence diagnostics
behind the Appendix table are correctly estimated, and that the mean-field solver
agrees with the exact answer.

## Reproducing the figures

| Thesis figure | Chapter | Produced by |
|---|---|---|
| `binding_modes_b5101_h1.pdf` | 2 | `src/figures/plot_binding_modes_b5101.py` |
| `jsd_heatmap_supertype.pdf` | 3 | `src/figures/plot_jsd_map.py` |
| `anchor_jsd_columns.pdf`, `anchor_jsd_difficulty.pdf` | 3 | `src/scans/scan_anchor_jsd.py` |
| `pwm_vs_rbm_density.pdf` | 3 | `src/figures/plot_pwm_vs_rbm_density.py` |
| `lambda_regime.pdf` | 4 | `src/scans/scan_lambda_regime.py` |
| `lambda_vs_jsd.pdf` | 4 | `src/scans/scan_lambda_vs_jsd.py` |
| `design_superbinder.pdf` | 4 | `src/figures/plot_design_plane_lambda.py`, off `src/scans/design_lambda.py` |
| `pooled_rbm_either.pdf` | 4 | `src/scans/pooled_rbm_either.py` |
| `boltzmann_proof.pdf` | 4 | `src/figures/plot_boltzmann_proof.py` |
| `case_study_HLA-B5101_vs_HLA-B2705.png` | 5 | `src/figures/plot_transition_landscape.py`, drawing via `src/data_explore.py` |
| `mode_crossing_barrier.pdf`, `mode_crossing_barrier_B2705.pdf` | 5 | `src/scans/mode_crossing_barrier.py`, allele set by `MCB_ALLELE` |
| `direct_global_transition.pdf` | 6 | `src/figures/plot_direct_global_transition.py` |
| `indirect_hidden_b51.pdf` | 6 | `src/figures/plot_indirect_hidden.py` |
| `b51_either_levels.pdf` | 6 | `src/figures/plot_b51_sweep_levels.py` |
| `mean_profile.pdf` | 6 | `src/figures/plot_mean_profile_thesis.py` |
| `mean_field_paths.pdf` | 6 | `src/figures/plot_mean_field.py`, off `src/scans/mean_field_corridor.py` |

## Running everything else

Everything runs from the repo root and writes under `results/`:

    python -u src/scans/scan_anchor_jsd.py

Scripts that cache their own output take `--plot` or `--replot` to redraw without
recomputing. A few need more than that:

    src/scans/scan_direct_global_transition.py   run twice, plain and with --fixed-beta
    src/scans/mode_crossing_barrier.py           allele set by the MCB_ALLELE variable
    src/b51_either_sweep.py                      --bulk for the sweep behind fig:b51_corridor
    src/b51/b51_position_diff.py                 --bulk likewise
    src/scans/calibrate_presented_threshold.py   stage1 and stage3, see above
    src/scans/scan_detour_vs_overlap.py          reads the output of src/figures/mean_profile.py

`--bulk` picks endpoints from inside the bulk of an allele's repertoire instead
of from its lowest-energy ligands, which sit at the edge of it. Both figures use
the bulk runs, which write their own tables.

## Attribution

The RBM engine is Jerome Tubiana's PGM, used unmodified and not redistributed
here. See the Environment section for how to obtain it. Its licence permits
non-commercial use and asks that research using it be cited, which my thesis does
(Tubiana, Cocco and Monasson 2019, and the RBM-MHC pipeline of Bravi et al. 2021).

The code here is MIT licensed, see `LICENSE`. That covers my own work only:
the files noted below came from elsewhere and keep whatever terms they arrived
with.

`entropy.py` was written by Yinfei Yang and came to me with the models. I use
it as I received it.

The 68 RBM models the analyses read were supplied to me, not trained here, and
I do not redistribute them for that reason. No code here fits them.
`data/iedb/` is derived from IEDB (Vita et al.).

Nothing in `data/` was produced by this code. `df_jsd_total.csv` and
`df_E_total.csv` were supplied alongside the models and I do not redistribute
them here, `proteome.txt` came from Yinfei Yang. Every script here only reads
them.

Everything under `src/` is my own work.
