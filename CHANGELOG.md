# Changelog

## 1.0.0 (2026-10-26)

First public release.

### Prediction
- `MetaboCast`: diffusion features [P, ÂP, Â²P] on a section-local spatial neighbour graph; metabolite
  targets rank-normalised within each training section; ridge penalty chosen by inner cross-validation
  (leave-one-section-out, or pooled spatial blocks for a single section); ridge regression averaged with 100
  extremely randomized trees fitted to the target principal components; monotone back-transform to log1p
  intensity. Options: `estimator`, `alpha`, `alpha_grid`, `target_transform`, `use_kegg`, `extra_key`, `gene_scaling`,
  `n_jobs`. A `RuntimeWarning` reports a cross-validated penalty at the largest value of the grid.
- Memory-lean prediction for large queries (PCA scores computed in row chunks from the highly variable genes).
- `prepare_query()`: coordinates from Visium, Visium HD, Slide-seq, Stereo-seq, Xenium, CosMx and MERSCOPE data,
  Ensembl-to-symbol mapping, normalisation of raw counts and aggregation of single-cell or subcellular data into
  pseudo-spots; `gene_panel_overlap()` reports how much of a model's gene panel a query measures.
- `reliability_table()`: per-ion accuracy within sections and on unseen sections, uncertainty width, putative
  exact-mass candidates (negative-mode adducts; every candidate kept, nearest first, one name per formula),
  chemical class and a reliability tier (transfer-supported, within-section only, low, uncalibrated); optional
  cell-state summaries.
- `calibrate()` and `estimate_conformal()`: per-ion uncertainty widths for a fitted model from held-out sections,
  and per-ion accuracy on those sections as transfer evidence.
- `metabocast.annotation`: the curated exact-mass reference (84 entries) behind `reliability_table()`, the R
  interface and the manuscript tables.
- Pathway activity (`PathwayScorer`) and metabolite-grounded pathway activity (`MetabolicActivity`); optional
  histology and cell-type feature blocks; experimental H&E-only companion model.
- R/Seurat interface (`R/metabocast.R`): prediction into a `metabocast` assay and ion lookup by metabolite name.

### Shipped model
- `metabocast_model.pkl`: trained on the seven paired DESI-MSI/Visium DESIUM sections; 2,086 m/z channels with
  per-ion accuracy within sections and on unseen sections, cross-fitted uncertainty widths and their coverage on
  held-out patients (see `MODEL_CARD.md`).

### Reproduction
- `reproduce/`: the analysis scripts behind every figure and table of the manuscript, the cached results they
  produced, and builders that regenerate the figures and the supplementary workbook from those results.
- `pipeline/train_and_evaluate.py`: train and benchmark models on any folder of paired sections.
- `pipeline/sma_to_metabocast.py`: training data from the public SMA MALDI-MSI/Visium deposit with the
  spot-to-pixel pairing published with it (`pipeline/sma_published_pairs/`).
- `docs/ANALYSIS_MANIFEST.md`: every figure and table of the manuscript with the code, result files and settings
  behind it.
- `reproduce/penalty_grid_check.py` and `reproduce/crosscancer/penalty_sensitivity.py`: the ridge penalty beyond the
  default grid for the shipped model, the held-out-section benchmark and the HCC/GBM application;
  `reproduce/timing_benchmark.py`: single-core run time and memory of fitting and prediction.
