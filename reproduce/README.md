# Reproducing the manuscript

This folder holds the analysis scripts behind every figure and table of the manuscript, and `results/` holds the
result files they produced. Figures and the supplementary workbook are rebuilt from `results/` alone; the analyses
themselves need the public datasets listed in [`../data/README.md`](../data/README.md).

```bash
pip install -e ".[reproduce]"                                   # from the repository root
python reproduce/figures/make_figures.py --out figures          # Figures 1-7 and Figures S1-S16 (Additional file 1)
python reproduce/tables/build_supplementary_tables.py --out MetaboCast_Additional_file_2.xlsx   # Tables S1-S21
python reproduce/manifest.py                                    # rewrites docs/ANALYSIS_MANIFEST.md
```

Item numbers, file names and sheet names come from [`manifest.py`](manifest.py), which lists the figures and tables
in the order the manuscript first cites them, with the result files each reads and the analyses that wrote them;
[`docs/ANALYSIS_MANIFEST.md`](../docs/ANALYSIS_MANIFEST.md) is its rendered form. `make_figures.py` also writes
`figure_summary.json` (values quoted in the legends) and the source tables of selected panels; the table builder
writes `<out>_notes.json` with the values quoted in the text. [`results/README.md`](results/README.md) describes the
result files.

## Setup

All commands run from the repository root. Data locations are read from environment variables
(`reproduce/paths.py`):

```bash
export METABOCAST_DATA_ROOT=/path/to/data          # DESIUM/, SMA/, the HCC and GBM files (data/README.md)
export DESIUM_DIR=$METABOCAST_DATA_ROOT/DESIUM/Correlation_NMF_analysis/data
export SMA_DIR=$METABOCAST_DATA_ROOT/SMA/sma_h5ad    # written by pipeline/sma_to_metabocast.py (below)
export R=reproduce/results
export OMP_NUM_THREADS=1                           # one BLAS thread per process
DES="BC_515_Section_1 BC_515_Section_2 BC_525 BC_823 LC_091 LC_170 LC_276"
DHB="SMA_DHB_V11L12-038_A1 SMA_DHB_V11L12-038_B1"
FMP_MOUSE="SMA_FMP_V11L12-109_A1 SMA_FMP_V11L12-109_B1 SMA_FMP_V11L12-109_C1 SMA_FMP_V11T16-085_A1 SMA_FMP_V11T16-085_B1 SMA_FMP_V11T16-085_C1"
FMP_HUMAN="SMA_FMP_V11T17-102_A1 SMA_FMP_V11T17-102_B1 SMA_FMP_V11T17-102_C1 SMA_FMP_V11T17-102_D1"
SMA_ALL="$DHB SMA_9AA_V11L12-038_D1 $FMP_MOUSE $FMP_HUMAN"
```

Results do not depend on the number of threads beyond floating-point rounding, which can move individual per-ion
values in the third decimal through the tree splits.

## Analyses

`manifest.py` maps every figure and table to these scripts. `estimator.py` applies the MetaboCast estimator to
precomputed feature matrices for the scripts that build their own features (fold-specific PCA, ablated graphs); it
reproduces `MetaboCast.fit` and `predict_metabolome` to float32 precision.

### Within-section benchmark and controls

Five k-means spatial blocks per section; highly variable genes (3,000), PCA (100 components), the penalty search and
both regressors use the training blocks only. The panel is the 1,000 most variable ions detected in more than 20% of
spots; every detected ion is also scored.

```bash
python reproduce/within_benchmark.py --data $DESIUM_DIR --out $R/within_desium.json
python reproduce/within_controls.py --data $DESIUM_DIR --out $R/within_controls.json
python reproduce/within_benchmark.py --data $SMA_DIR --sections $SMA_ALL --maps --out $R/within_sma.json
python reproduce/annotate_ions.py --data $DESIUM_DIR --out $R/desium_ion_annotation.csv
```

`within_benchmark.py` writes `<out>.json`, `<out>_per_ion.npz`, `<out>_maps.npz` (held-out maps) and
`<out>_conformal.npz` (interval widths). `within_controls.py` adds the coordinate-only k-nearest-neighbour regressor
(k and weighting chosen by leave-one-block-out inside the training blocks) and the sensitivity analysis in which each
fold selects its 1,000-ion panel from its training blocks. `annotate_ions.py` assigns putative negative-mode
exact-mass candidates ([M-H]-, [M+Cl]-, [M+HCOO]-; ±0.01 Da) and keeps every candidate.

### Held-out sections, patients and animals

The model is fitted on the other sections of the same cohort (highly variable genes from the training sections) and
scored on ions detected in more than 20% of the held-out spots.

```bash
python reproduce/loso_benchmark.py --data $DESIUM_DIR --sections $DES --out $R/loso_desium.json
python reproduce/loso_benchmark.py --data $DESIUM_DIR --sections $DES --estimator ridge --out $R/loso_desium_ridge.json
python reproduce/loso_benchmark.py --data $DESIUM_DIR --sections $DES --use-kegg --out $R/loso_desium_kegg.json
python reproduce/loso_baselines.py --data $DESIUM_DIR --out $R/loso_baselines.json
python reproduce/coverage_outer.py --data $DESIUM_DIR --out $R/coverage/coverage_outer.json
for g in DHB FMP_MOUSE FMP_HUMAN; do
  n=$(echo $g | tr A-Z a-z)
  python reproduce/loso_benchmark.py --data $SMA_DIR --sections ${!g} --maps FMP557:557.2 --out $R/sma/loso_$n.json
  python reproduce/loso_benchmark.py --data $SMA_DIR --sections ${!g} --estimator ridge --out $R/sma/loso_${n}_ridge.json
  python reproduce/loso_benchmark.py --data $SMA_DIR --sections ${!g} --use-kegg --upper-genes --out $R/sma/loso_${n}_kegg.json
  python reproduce/loso_benchmark.py --data $SMA_DIR --sections ${!g} --upper-genes --out $R/sma/loso_${n}_upper.json
done
python reproduce/loso_benchmark.py --data $SMA_DIR --sections $FMP_MOUSE --hold-out subject --out $R/sma/loso_fmp_mouse_by_animal.json
python reproduce/per_ion_reliability.py --within $R/within_desium_per_ion.npz --loso $R/loso_desium_per_ion.npz --out $R
```

- `loso_baselines.py`: paired-supervision baselines under the same held-out-section design (the estimator without
  diffusion features, the estimator on KEGG pathway features, joint NMF of genes and metabolites, single-gene
  regression).
- `coverage_outer.py`: each patient (both BC_515 sections together) is excluded from fitting, penalty selection and
  width estimation; the released width procedure is repeated on the remaining patients and its coverage measured on
  the held-out patient, together with split calibration on one remaining patient. Holding out patients also gives
  the held-out-patient accuracy.
- `--hold-out subject` leaves out every section of the held-out section's animal or donor (`uns["sma"]["subject"]`).
- `--upper-genes` maps mouse gene symbols to their human upper-case form so that the human KEGG gene sets apply; the
  `_upper` runs are the matched comparison for the KEGG prior on mouse data.
- `per_ion_reliability.py` writes `desium_per_ion_accuracy.csv` (within-section and held-out-section accuracy per
  channel), which defines the reliability tiers.

### SMA training data and registration

The SMA analyses use the spot-to-pixel pairing published with the deposit, exported once and stored in
`pipeline/sma_published_pairs/`:

```bash
Rscript pipeline/sma_published_pairs.R $METABOCAST_DATA_ROOT/SMA/misc work/sma_pairs_export
python pipeline/sma_published_pairs.py --sma $METABOCAST_DATA_ROOT/SMA --export work/sma_pairs_export \
    --metadata $METABOCAST_DATA_ROOT/SMA/metadata.csv --out pipeline/sma_published_pairs
python pipeline/sma_to_metabocast.py $METABOCAST_DATA_ROOT/SMA --pairs pipeline/sma_published_pairs
python reproduce/sma_registration_audit.py --sma $METABOCAST_DATA_ROOT/SMA --out $R/sma/sma_registration_audit.csv \
    --convert-qc $SMA_DIR/sma_convert_qc.json [--coupling-h5ad <coupling-registered h5ad folder>]
```

The audit compares the published pairing with two alternatives: registration by tissue geometry alone, which does
not use the outcome, and the orientation that maximises transcriptome-metabolite coupling, which does (`--pairs none` with
`pipeline/sma_to_metabocast.py` builds the geometry-registered files). It writes the per-section agreement,
`sma_registration_audit_tic.csv.gz` (total ion count of the paired pixel per spot) and, with `--convert-qc`,
`results/sma_coregistration_qc.csv`.

### Transcriptome harmonisation and robustness

```bash
METABOCAST_WORK=$R python reproduce/harmonization.py
for s in splits spatial_control predictability baselines ceiling gcn uncertainty shift; do
  METABOCAST_WORK=$R/rigor python reproduce/rigor_$s.py
done
```

`rigor_predictability.py` reads the putative classes from `results/desium_ion_annotation.csv`. The model-selection
screen is in [`model_selection/`](model_selection/README.md).

### Ridge penalty and run time

```bash
python reproduce/penalty_grid_check.py --data $DESIUM_DIR --root $METABOCAST_DATA_ROOT --out $R/penalty_grid.json
python reproduce/crosscancer/penalty_sensitivity.py --root $METABOCAST_DATA_ROOT --out $R/crosscancer/penalty_sensitivity.json
python reproduce/timing_benchmark.py --data $DESIUM_DIR --root $METABOCAST_DATA_ROOT --out $R/timing.json
```

`penalty_grid_check.py` repeats the penalty selection on the default grid extended to 10¹⁰ for the shipped model,
each leave-one-section-out fold and the HCC/GBM application model; `penalty_sensitivity.py` refits the application
model at the optimum of a grid extended to 10¹² and compares its HCC contrasts with those of the released penalty.
`timing_benchmark.py` times fitting on one and on seven DESIUM sections and prediction for 9,356 HCC spots on one
thread and records the machine.

### Shipped model

```bash
python reproduce/build_shipped_model.py --data $DESIUM_DIR --within $R/within_desium_per_ion.npz \
    --coverage $R/coverage/coverage_outer.json --out metabocast_model.pkl \
    --curve $R/shipped_model/alpha_curve.json > $R/shipped_model/build_shipped_model.log
python reproduce/update_shipped_uncertainty.py        # coverage metadata only, without refitting
```

The model is `MetaboCast()` fitted on all seven DESIUM sections. Its per-ion reliability comes from held-out data:
within-section accuracy from the benchmark above, held-out-section accuracy and cross-fitted widths from seven refits
that each leave one section out, and the coverage of those widths on held-out patients from `coverage_outer.py`.
The released pickle was built by `build_shipped_model.py`; `update_shipped_uncertainty.py` then wrote the
outer-test coverage into its `uncertainty_` without changing the fitted model.

### HCC and glioblastoma

```bash
python reproduce/crosscancer/regenerate_crosscancer.py --root $METABOCAST_DATA_ROOT --desium $DESIUM_DIR \
    --accuracy $R/desium_per_ion_accuracy.csv --out work/crosscancer
python reproduce/crosscancer/derive_figure_sources.py --run work/crosscancer --root $METABOCAST_DATA_ROOT --out $R/crosscancer
python reproduce/crosscancer/summarise_crosscancer.py --run work/crosscancer --accuracy $R/desium_per_ion_accuracy.csv \
    --out $R/crosscancer
cp work/crosscancer/provenance.json $R/crosscancer/
```

One model is trained on all seven DESIUM sections on the genes shared by DESIUM, the five HCC samples and both GBM
sections, and predicts every HCC region and both GBM sections; `provenance.json` records inputs, hashes and
parameters. Every ion is labelled with its DESIUM tier, and the GBM niches use the transfer-supported channels.
`summarise_crosscancer.py` reports the DESIUM accuracy of the ions used and counts agreement with
literature-expected directions descriptively (the 28 calls share one model, seven metabolites and four patients and
are not independent replicates).

### Histology and cell-type ablations

```bash
python reproduce/histology_ablation.py --data $DESIUM_DIR --protocol loso --out $R/ablations/histology_loso.csv
python reproduce/histology_ablation.py --data $DESIUM_DIR --protocol within --out $R/ablations/histology_within.csv
python reproduce/histology_only_ablation.py --data $DESIUM_DIR --test all --out $R/ablations/histology_only_desium.csv
python reproduce/crosscancer/regenerate_s14_panelBC.py --root $METABOCAST_DATA_ROOT --crosscancer work/crosscancer --out work/s14
python reproduce/celltype_ablation.py --data $DESIUM_DIR --out $R/ablations
```

`--test all` runs the H&E-only model under leave-one-section-out, with shuffled features and within sections;
`ablations/histology_only_hcc_concordance.csv` holds the H&E columns of `work/s14/S14_panelC_regenerated.csv`.

### Flux test and protein markers

```bash
python reproduce/flux_validation.py --out $R/flux/flux_stroma.csv                 # tumour vs stroma, breast sections
python reproduce/flux_validation.py --labels rest --out $R/flux/flux_rest.csv     # tumour vs all other spots, all sections
METABOCAST_WORK=$R python reproduce/gbm_multimodal.py
```

### Transcriptome-only metabolic inference (scFEA, MGFEA)

```bash
git clone https://github.com/changwn/scFEA external/scFEA
git clone https://github.com/Sunwenzhilab/MGFEA external/MGFEA && (cd external/MGFEA && git apply ../../reproduce/baselines/MGFEA_patches.diff)
cp reproduce/baselines/scFEA_patched.py external/scFEA/src/scFEA.py
export SCFEA_REPO=$PWD/external/scFEA MGFEA_REPO=$PWD/external/MGFEA METABOCAST_BASELINE_DIR=$PWD/work/baselines
mkdir -p $METABOCAST_BASELINE_DIR/input $METABOCAST_BASELINE_DIR/output $METABOCAST_BASELINE_DIR/mgfea_out
(cd $METABOCAST_BASELINE_DIR && python "$OLDPWD/reproduce/baselines/prep.py")   # module-gene expression, MSI, coordinates
bash reproduce/baselines/run_scfea.sh                                            # scFEA, all seven sections
python reproduce/baselines/run_mgfea.py BC_823 LC_170                            # MGFEA, unsupervised spatial mode
python reproduce/baselines/compare_baselines.py --results $R --out $R/baselines
```

`scFEA_patched.py` replaces `DataFrame.append` and builds the module-gene matrix in NumPy for current pandas; on
BC_823 its output matched the original code (r = 0.99999). `MGFEA_patches.diff` fixes NumPy matrix, pandas
positional indexing and sparse stoichiometry issues.

### Not used in the manuscript

`magpie_lung_validation.py` builds paired data from the public MAGPIE DESI-MSI/Visium lung cohort. The registration
of those data cannot be recovered from the public files, and choosing an orientation by transcriptome-metabolite
coupling is not a valid registration criterion (it fails on SMA, where the published pairing is known), so the
script and `results/magpie/` are kept for reference only. `examples/metabolic_activity.py` computes
metabolite-grounded pathway coherence (`results/pathway_coherence_BC_515_Section_1.csv`).

## Notes on the protocols

- **Within-section scores** pool the out-of-fold predictions of the five fold models before each ion's correlation
  is computed, on the log1p intensity scale. Leave-one-section-out scores use one model per held-out section.
- **Detection matching.** Ions are scored where detected in more than 20% of the evaluated spots.
- **Uncertainty.** The released widths are cross-fitted (leave-one-section-out) residual quantiles; their coverage
  was measured on held-out patients (`coverage_outer.py`). `MetaboCast.calibrate()` uses the fitted predictor and
  finite-sample residual order statistics; section mode takes the largest residual per ion within each section and
  assumes independent, exchangeable sections. Serial sections are not independent subjects, and no guarantee is
  claimed under spatial dependence or distribution shift.
- **Statistics.** Effect sizes are paired differences of per-section medians with 5,000 paired bootstrap resamples;
  exact two-sided Wilcoxon signed-rank tests are used as win-consistency tests. No per-spot tests are reported for
  the HCC and GBM predictions, because spots within a region are not independent replicates.
