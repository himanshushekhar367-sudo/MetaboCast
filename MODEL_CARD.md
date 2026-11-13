# Model card: `metabocast_model.pkl`

What the shipped MetaboCast model is, what it should and should not be used for, and how it fails.

## Model details

- **Task.** Predict per-spot mass-spectrometry-imaging (MSI) ion intensities from spatial transcriptomics (gene
  expression and spot coordinates) alone.
- **Software.** MetaboCast 1.0.0 (`MetaboCast()` with the package defaults), built by
  `reproduce/build_shipped_model.py`; its coverage metadata were written by `reproduce/update_shipped_uncertainty.py`
  (metadata only, no refit). File size 35 MB; SHA-256
  `93ff87974e1b623e8868890746fb8fc4073cc2c52b98aa304bc963a52b2b8909`.
- **Architecture.** 3,000 highly variable genes, 100 principal components and diffusion features [P, ÂP, Â²P] from a
  degree-normalised six-nearest-neighbour spot graph. Each ion is rank-transformed to Gaussian scores within each
  training section. A multi-output ridge regression, with the penalty chosen by leave-one-section-out
  cross-validation (α = 10⁷, the top of the default grid; on a grid extended to 10¹⁰ the score peaks at 10⁸, less
  than 0.001 higher, and held-out-section accuracy is unchanged, `reproduce/results/penalty_grid.json`), and 100
  extremely randomized trees fitted to the 50 leading principal components of the targets are averaged per ion after
  scaling to equal spread. Predictions are mapped back to log1p intensity through each ion's pooled training
  quantiles, a monotone map that keeps the predicted spatial ordering. The optional KEGG prior is off. The trees are
  stored as plain arrays.
- **Training data.** DESIUM: seven paired DESI-MSI/Visium sections of human breast (BC_515 sections 1 and 2,
  BC_525, BC_823) and lung (LC_091, LC_170, LC_276) cancer from six patients, 14,816 spots, 15,620 genes (Godfrey
  et al., Angew. Chem. Int. Ed. 2025; Harvard Dataverse [10.7910/DVN/GZFCWC](https://doi.org/10.7910/DVN/GZFCWC)).
- **Output.** 2,086 negative-mode DESI-MSI m/z channels (m/z 89-1,290). These are ion channels, not identified
  metabolites. Names attached to them are putative exact-mass candidates from the curated reference of
  `metabocast/annotation.py` (84 common tissue metabolites and lipid species; the one the manuscript uses), with the
  negative-mode adducts [M−H]⁻, [M+Cl]⁻ and [M+HCOO]⁻ within ±0.01 Da; every candidate is kept, nearest first, and
  isomers that share a formula carry one group name (hexose for glucose, fructose, galactose or inositol).

## Reliability shipped with the model

All values come from held-out data:

- `train_predictability_`: per-ion Spearman correlation under within-section spatial-block cross-validation, mean
  over the seven sections (median 0.121; 565 channels reach r ≥ 0.30). Within-section values are raised by spatial
  autocorrelation and do not show that an ion transfers.
- `transfer_predictability_`: per-ion Spearman correlation on unseen sections (leave-one-section-out, mean over the
  seven held-out sections; median 0.050). This is the guide for new tissue.
- **Tiers** (`reliability_table()`): *transfer-supported*, held-out-section r ≥ 0.20 (310 channels);
  *within-section only*, within-section r ≥ 0.30 with weaker transfer (300 channels); *low* (1,474); *uncalibrated*,
  no held-out estimate (2). The thresholds are arguments (`within_min`, `transfer_min`).
- `conf_width_`: per-ion 90% quantile of absolute leave-one-section-out residuals (log1p units). When each patient
  was excluded from fitting, penalty selection and width estimation and the procedure was repeated on the remaining
  patients, the widths covered 0.88 of the held-out patient's spots on average (0.80-0.92 across sections; 0.89
  averaged over patients), close to the nominal 0.90; calibrating on a single patient instead under-covered (0.80).
  `uncertainty_["outer_test_coverage"]` holds these values. `uncertainty_["internal_check_coverage"]` holds an
  earlier per-section check that reuses the same residuals and is not an independent test. The widths are empirical,
  not a coverage guarantee: tissue unlike the training sections can fall outside them.
  `.uns["metabocast_conf_width_adj"]` widens the width in proportion to the training genes missing from a query, a
  heuristic.

## Accuracy

Median Spearman correlation between predicted and measured intensity, manuscript protocols.

| Setting | Median Spearman |
|---|---|
| DESIUM, within section (spatial-block CV, 1,000 best-measured ions, mean of 7 sections) | 0.337 ± 0.078 |
| DESIUM, unseen section (leave-one-section-out, ions detected in > 20% of spots, mean of 7) | 0.045 (breast 0.062, lung 0.023) |
| DESIUM, unseen patient (both BC_515 sections held out together) | 0.042 |

Under the same held-out-section design, joint NMF (0.019) and single-gene regression (0.010) transferred worse in
every section, the estimator without diffusion features matched MetaboCast (0.044), and the estimator on KEGG
pathway features averaged 0.063: better in three breast sections, worse in the fourth breast section and the three
lung sections. On the 1,000-ion panel, an H&E-only model on 13 patch statistics reached 0.139 against 0.106 for the
transcriptome-only model (higher in four of six sections; Wilcoxon P = 0.22), and adding the H&E features to the
transcriptome improved transfer in all six sections (mean 0.158; P = 0.031); see `MorphologyMetabolitePredictor`
and `extra_key`.

The same estimator trained on the SMA MALDI-MSI/Visium cohort (Vicari et al., Nat. Biotechnol. 2024), with the
spot-to-pixel pairing published with that deposit, transferred to unseen sections with median 0.32 for DHB lipids
(two sections of one mouse), 0.29 for FMP neurotransmitters in mouse substantia nigra and 0.10 in mouse striatum
(three mice; over all six mouse FMP sections, 0.199 with whole animals held out against 0.196 with sections held
out) and 0.15 in human striatum (four sections of one donor). The cohorts differ in chemistry, species, tissue and design, so these numbers are not
a ranking of tissues.

## Intended use

- Hypothesis generation on transcriptome-only tissue: prioritising spatial metabolite hypotheses for targeted MSI or
  MS/MS.
- Most informative for tissue resembling the human tumour training data, for transfer-supported ions and for
  spatially structured predicted maps.

## Known failure modes and limits

- **Weak transfer between tumour patients** (median 0.045 on unseen sections). Rank normalisation removes monotone
  intensity differences between sections, not differences in tissue composition.
- **Within-section-only ions.** 300 channels predict well inside a section but not on unseen sections (for example
  putative glutathione, ascorbate, palmitate and stearate). Read their predictions on new tissue as leads at most.
- **Poorly predicted classes.** Putative amino acids, sugar phosphates and other polar channels were predicted worse
  than lipids and nucleotides in DESIUM. Predictability tracked spatial autocorrelation and varied across classes
  and cohorts; these comparisons do not isolate biochemical regulation from acquisition and tissue effects. Check
  each ion's held-out accuracy rather than its class.
- **Tissue absent from training.** The model has seen no brain. On glioblastoma, predicted N-acetylaspartate, a
  neuronal metabolite, followed the proliferation marker PCNA rather than neural tissue, a likely transfer failure.
- **Gene-panel overlap.** Genes absent from a query are imputed at their training mean; low overlap
  (`last_gene_overlap_`, `last_hvg_overlap_`) means extrapolation.
- **Other species.** Mouse or other non-human data share few gene symbols with the model; retrain on paired data from
  the species (`pipeline/train_and_evaluate.py`) rather than relying on upper-cased symbols.
- **Identity claims.** Named-metabolite statements need MS/MS or FDR-controlled annotation (for example METASPACE).

## Before interpreting a prediction

Treat a prediction as a hypothesis worth measuring only when the query resembles the training tissue and platform,
gene-panel overlap is high, the ion is transfer-supported in `reliability_table()` and its predicted map is
spatially structured. These checks do not validate a prediction.

## Provenance and safety

Trained on public, de-identified human cancer data under the source licence. Predictions are computational and are
not a clinical or diagnostic device. The model is a Python pickle: load pickle files only from trusted sources. It
was saved with the package versions in `requirements-lock.txt` (scikit-learn 1.8.0); other scikit-learn versions
load it with a single warning.
