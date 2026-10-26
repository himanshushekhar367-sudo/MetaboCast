# Result files

Outputs of the analysis scripts in `reproduce/` (commands in [`../README.md`](../README.md)). The figure and table
builders read only these files; [`../../docs/ANALYSIS_MANIFEST.md`](../../docs/ANALYSIS_MANIFEST.md) lists which
figure or table reads which file. Spearman correlations are per ion between predicted and measured log1p intensity;
"median" is the median over ions detected in more than 20% of the evaluated spots unless stated. Item names below
are manifest keys (`reproduce/manifest.py`), which carry the current figure and table numbers.

| File | Contents | Script |
|---|---|---|
| `within_desium.json`, `within_sma.json` | within-section benchmark: per section and method, the median over the 1,000-ion panel and over all detected ions, on the log1p scale and on each model's training-target scale; selected penalties | `within_benchmark.py` |
| `within_*_per_ion.npz` | per-ion Spearman per method and section (`<method>__<section>`), m/z axis (`mz__<section>`) | `within_benchmark.py` |
| `within_*_maps.npz` | measured and held-out predicted maps of the ions shown in Figs 3 and 4 and SuppFig_maps / SuppFig_maps_shared_scale | `within_benchmark.py` |
| `within_*_conformal.npz` | per-ion interval widths and coverage behind Fig. 4, and block coordinates | `within_benchmark.py` |
| `within_controls.json` (+ `_per_ion.npz`) | coordinate-only k-nearest-neighbour regressor (k and weighting tuned inside the training blocks) and the training-block-only panel selection (within_controls table) | `within_controls.py` |
| `loso_desium.json`, `loso_desium_ridge.json`, `loso_desium_kegg.json` (+ `_per_ion.npz`) | DESIUM leave-one-section-out: default estimator, ridge half only, with the KEGG prior | `loso_benchmark.py` |
| `loso_baselines.json` (+ `_per_ion.npz`) | paired-supervision baselines under the same held-out-section design: no-diffusion estimator (key `graph-free`), KEGG features + regressor, joint NMF, single gene | `loso_baselines.py` |
| `coverage/coverage_outer.json` (+ `.npz`) | each patient held out of fitting, penalty selection and width estimation: coverage of the released widths and of single-patient split calibration, and held-out-patient accuracy | `coverage_outer.py` |
| `sma/loso_<cohort>[_ridge, _kegg, _upper].json` (+ `_per_ion.npz`) | the same as `loso_desium*` for the SMA DHB, mouse FMP and human FMP cohorts with the published spot-to-pixel pairing; `_maps.npz` holds the FMP ion maps of SuppFig_neurotransmitter | `loso_benchmark.py` |
| `sma/loso_fmp_mouse_by_animal.json` (+ `_per_ion.npz`) | mouse FMP with every section of the held-out animal excluded from training | `loso_benchmark.py --hold-out subject` |
| `sma/sma_registration_audit.csv`, `sma/sma_registration_audit_tic.csv.gz` | agreement of geometry registration and of the coupling-chosen orientation with the published pairing; total ion count of the paired pixel per spot | `sma_registration_audit.py` |
| `sma_coregistration_qc.csv` | per SMA section: chemistry, subject, region, spots, m/z channels, transcriptome-metabolite coupling and shuffled-spot null under the published pairing | `sma_registration_audit.py --convert-qc` |
| `desium_per_ion_accuracy.csv` | per-ion mean within-section and leave-one-section-out Spearman over the seven DESIUM sections; defines the reliability tiers | `per_ion_reliability.py` |
| `desium_ion_annotation.csv` | putative negative-mode exact-mass candidates of the 2,086 DESIUM channels from the reference in `metabocast/annotation.py` ([M-H]-, [M+Cl]-, [M+HCOO]-; ±0.01 Da; every candidate kept, nearest first; error = observed − theoretical; class only when all candidates share it) | `annotate_ions.py` |
| `penalty_grid.json` | inner cross-validation score of every penalty on the default grid extended to 10¹⁰ for the shipped model, each leave-one-section-out fold (with held-out accuracy under both choices) and the HCC/GBM application model (robustness table) | `penalty_grid_check.py` |
| `crosscancer/penalty_sensitivity.json`, `crosscancer/penalty_sensitivity_contrasts.csv` | HCC contrasts of the application model at the released penalty and at the optimum of a grid extended to 10¹²; agreement summaries and every contrast of the upper-tier channels (hcc_patients table, sheet b) | `crosscancer/penalty_sensitivity.py` |
| `timing.json` | single-thread run time and peak memory of fitting on one and seven DESIUM sections and of prediction for 9,356 HCC spots, with the machine | `timing_benchmark.py` |
| `harmonization.json` | leave-one-section-out under five transcriptome representations (harmonisation table, SuppFig_harmony) | `harmonization.py` |
| `rigor/` | learner families and PCA size, diffusion depth and validation splits, extended baselines, determinants of predictability, uncertainty under shift, distribution shift, graph convolution, spatial controls (SuppFig_predictability, SuppFig_spatial_controls, SuppFig_robustness, robustness table) | `rigor_*.py` |
| `model_selection/` | model-selection screen (model_selection table; SuppFig_robustness panel e) | `model_selection/` |
| `shipped_model/` | build log of `metabocast_model.pkl` (penalty, held-out accuracy, internal width check), the inner cross-validation score of every penalty, and the log of the coverage-metadata update | `build_shipped_model.py`, `update_shipped_uncertainty.py` |
| `crosscancer/` | HCC and GBM run. File names are the run's own labels: `S8_*` (HCC-2 normal to PVTT; hcc_pvtt table), `S9_and_Fig7c_source.csv` (contrasts across patients; hcc_patients table), `S10_*` (GBM protein co-localisation; gbm_protein table); figure sources `Fig7*`, `HCC2_*`, `HCC_*`, `GBM_Visium_maps.csv`; DESIUM accuracy and tier of the ions used and descriptive direction counts (`crosscancer_summary.json`, `crosscancer_ion_reliability.csv`); run provenance (`provenance.json`: input SHA-256 hashes, library versions and parameters; its `git_head` and script hash identify the working copy that ran the analysis, whose prediction code is identical to this release, which differs only in names, comments and log messages) | `crosscancer/*.py` |
| `ablations/histology_loso.csv`, `histology_within.csv` | MetaboCast with and without H&E features (histology table) | `histology_ablation.py` |
| `ablations/histology_only_desium.csv` | H&E-only model on DESIUM: leave-one-section-out, shuffled features, within-section (histology table) | `histology_only_ablation.py` |
| `ablations/histology_only_hcc_concordance.csv` | H&E-only normal-to-tumour directions in HCC (crosscancer_summary table) | `crosscancer/regenerate_s14_panelBC.py` |
| `ablations/celltype_ablation_*` | cell-type feature ablation per section, by class and per ion (celltype table) | `celltype_ablation.py` |
| `flux/flux_stroma.csv`, `flux/flux_rest.csv` | Warburg flux test, tumour versus stroma and versus all other spots (flux_baselines table) | `flux_validation.py` |
| `gbm_multimodal.json` | how much of each GBM CytAssist protein marker the transcriptome reconstructs (robustness table) | `gbm_multimodal.py` |
| `baselines/` | scFEA and MGFEA against measured MSI, with MetaboCast on the same ions (flux_baselines table, SuppFig_flux) | `baselines/` |
| `magpie/` | MAGPIE lung analysis with coupling-selected orientations; not used in the manuscript (the registration cannot be recovered from the public files) | `magpie_lung_validation.py` |
| `pathway_coherence_BC_515_Section_1.csv` | enzyme-metabolite coherence per KEGG pathway with measured metabolites; example output, not used in the manuscript | `examples/metabolic_activity.py` |
