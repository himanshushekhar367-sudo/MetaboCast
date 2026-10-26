"""Numbering of the manuscript's display items and the analysis manifest.

One list per kind of item, in the order the manuscript first cites them; supplementary figures and tables are
numbered from that order, so moving an entry renumbers every builder, file name and sheet consistently. Each entry names the builder that draws or
writes the item, the result files it reads (reproduce/results) and the analysis scripts that wrote those files,
with their settings. reproduce/figures and reproduce/tables take their file and sheet names from here, and
`python reproduce/manifest.py` writes docs/ANALYSIS_MANIFEST.md.

Figures 1-7 are the main figures; FigureS<n> are in Additional file 1 and Table S<n> in Additional file 2.
"""
import os
import sys

FIGURES = [
    dict(key="Figure1", label="Figure 1", title="MetaboCast design", builder="figures/fig_overview.py: fig1",
         reads=[], analyses=[]),
    dict(key="Figure2", label="Figure 2", title="Within-section benchmark", builder="figures/fig_benchmark.py: fig2",
         reads=["within_desium.json", "within_desium_per_ion.npz", "within_controls.json"],
         analyses=["within_benchmark.py (DESIUM; 5 k-means spatial blocks; 3,000 training-block HVGs; 100 PCs; top-1,000 panel)",
                   "within_controls.py (tuned coordinate-only kNN, k 1-320, uniform or inverse-distance weights, chosen by "
                   "leave-one-block-out inside the training blocks; training-block-only panel selection)"]),
    dict(key="Figure3", label="Figure 3", title="Spatial predictions and putative classes", builder="figures/fig_benchmark.py: fig3",
         reads=["within_desium_maps.npz", "within_desium_conformal.npz", "within_desium_per_ion.npz", "desium_ion_annotation.csv"],
         analyses=["within_benchmark.py", "annotate_ions.py (reference of metabocast/annotation.py; negative-mode adducts [M-H]-, [M+Cl]-, [M+HCOO]-; ±0.01 Da; all candidates kept, nearest first)"]),
    dict(key="Figure4", label="Figure 4", title="Uncertainty widths", builder="figures/fig_benchmark.py: fig4",
         reads=["within_desium_conformal.npz", "coverage/coverage_outer.json", "desium_ion_annotation.csv"],
         analyses=["within_benchmark.py (within-section widths: refit on a random 75% of the spots in the training blocks, 90% quantile on the other 25%)",
                   "coverage_outer.py (each patient held out of fitting, penalty selection and width estimation)"]),
    dict(key="Figure5", label="Figure 5", title="Transfer and reliability tiers", builder="figures/fig_transfer.py: fig5",
         reads=["loso_desium.json", "coverage/coverage_outer.json", "loso_baselines.json", "ablations/histology_loso.csv",
                "ablations/histology_only_desium.csv", "desium_per_ion_accuracy.csv",
                "crosscancer/HCC_NtoT_annotated_by_patient.csv"],
         analyses=["loso_benchmark.py (leave one section out; 3,000 training-section HVGs)", "coverage_outer.py (leave one patient out)",
                   "loso_baselines.py (no diffusion, KEGG features + regressor, joint NMF, single gene)",
                   "histology_ablation.py, histology_only_ablation.py (13 H&E patch features)", "per_ion_reliability.py",
                   "crosscancer/regenerate_crosscancer.py + derive_figure_sources.py"]),
    dict(key="Figure6", label="Figure 6", title="Transfer across cohorts", builder="figures/fig_transfer.py: fig6",
         reads=["loso_desium.json", "sma/loso_dhb.json", "sma/loso_fmp_mouse.json", "sma/loso_fmp_human.json",
                "sma/loso_fmp_mouse_by_animal.json", "within_desium.json", "within_sma.json"],
         analyses=["../pipeline/sma_to_metabocast.py (published spot-pixel pairing)",
                   "loso_benchmark.py (SMA cohorts; --hold-out subject for whole animals)", "within_benchmark.py (SMA)"]),
    dict(key="Figure7", label="Figure 7", title="Transcriptome-only HCC and GBM", builder="figures/fig_crosscancer.py: fig7",
         reads=["crosscancer/HCC2_maps.csv", "crosscancer/S8_full_all_ions.csv", "crosscancer/S9_and_Fig7c_source.csv",
                "crosscancer/S10_metabolite_x_protein.csv", "crosscancer/Fig7e_niches.csv", "crosscancer/Fig7f_metabolite_by_niche.csv"],
         analyses=["crosscancer/regenerate_crosscancer.py (one model on all seven DESIUM sections; genes shared with HCC and GBM; "
                   "100 PCs; k = 6)", "crosscancer/derive_figure_sources.py"]),
    # Additional file 1
    dict(key="SuppFig_flux", title="Comparison with transcriptome-only metabolic inference",
         builder="figures/fig_overview.py: s9", reads=["baselines/baseline_comparison_per_ion.csv"],
         analyses=["baselines/ (scFEA v1.1, MGFEA unsupervised spatial mode) + baselines/compare_baselines.py"]),
    dict(key="SuppFig_maps", title="Measured and held-out predicted maps", builder="figures/fig_benchmark.py: supp_maps",
         reads=["within_desium_maps.npz", "within_desium_conformal.npz", "desium_ion_annotation.csv"], analyses=["within_benchmark.py"]),
    dict(key="SuppFig_maps_shared_scale", title="Maps on a shared intensity scale",
         builder="figures/fig_benchmark.py: supp_shared_scale", reads=["within_desium_maps.npz", "within_desium_conformal.npz"],
         analyses=["within_benchmark.py"]),
    dict(key="SuppFig_best_worst", title="Best- and worst-predicted putative identities",
         builder="figures/fig_benchmark.py: supp_best_worst", reads=["within_desium_per_ion.npz", "desium_ion_annotation.csv"],
         analyses=["within_benchmark.py", "annotate_ions.py"]),
    dict(key="SuppFig_predictability", title="Determinants of predictability",
         builder="figures/fig_robustness.py: ed5", reads=["rigor/rigor_predictability.json", "rigor/rigor_shift.json", "rigor/rigor_gcn.json"],
         analyses=["rigor_predictability.py", "rigor_shift.py", "rigor_gcn.py"]),
    dict(key="SuppFig_spatial_controls", title="Spatial-structure controls",
         builder="figures/fig_robustness.py: ed6", reads=["rigor/rigor_spatial_control.json"], analyses=["rigor_spatial_control.py"]),
    dict(key="SuppFig_robustness", title="Learners, diffusion depth, validation design and uncertainty under shift",
         builder="figures/fig_robustness.py: ed4",
         reads=["rigor/rigor_ceiling.json", "rigor/rigor_splits.json", "rigor/rigor_baselines.json", "rigor/rigor_uncertainty.json",
                "model_selection/r3_loso_desium.json", "model_selection/loso2_desium.json"],
         analyses=["rigor_ceiling.py", "rigor_splits.py", "rigor_baselines.py", "rigor_uncertainty.py", "model_selection/"]),
    dict(key="SuppFig_harmony", title="Transcriptome harmonisation", builder="figures/fig_transfer.py: s1",
         reads=["harmonization.json"], analyses=["harmonization.py"]),
    dict(key="SuppFig_SMA_registration", title="SMA registration", builder="figures/fig_supplementary.py: s3",
         reads=["sma/sma_registration_audit.csv", "sma/sma_registration_audit_tic.csv.gz", "sma_coregistration_qc.csv"],
         analyses=["../pipeline/sma_published_pairs.R + sma_published_pairs.py (pairing from the deposit)",
                   "../pipeline/sma_to_metabocast.py", "sma_registration_audit.py"]),
    dict(key="SuppFig_SMA_within", title="Within-section benchmark on DESIUM and SMA",
         builder="figures/fig_overview.py: s8", reads=["within_desium.json", "within_sma.json"], analyses=["within_benchmark.py"]),
    dict(key="SuppFig_neurotransmitter", title="SMA neurotransmitter maps",
         builder="figures/fig_transfer.py: supp_neurotransmitter", reads=["sma/loso_fmp_mouse_maps.npz", "sma/loso_*.json"],
         analyses=["loso_benchmark.py --maps FMP557:557.2"]),
    dict(key="SuppFig_HCC2_subsections", title="HCC-2 subsections", builder="figures/fig_crosscancer.py: s5",
         reads=["crosscancer/HCC2_maps.csv", "crosscancer/HCC2_trajectory.csv", "crosscancer/S8_supported_annotated.csv"],
         analyses=["crosscancer/regenerate_crosscancer.py", "crosscancer/derive_figure_sources.py"]),
    dict(key="SuppFig_GSH_patients", title="Predicted glutathione by patient",
         builder="figures/fig_crosscancer.py: ed3", reads=["crosscancer/HCC_GSH_per_patient.csv"],
         analyses=["crosscancer/derive_figure_sources.py"]),
    dict(key="SuppFig_HCC_typelevel", title="HCC contrasts by histological spot type",
         builder="figures/fig_crosscancer.py: s6",
         reads=["crosscancer/HCC_TypeLevel_TumorCell_vs_NormalCell.csv", "crosscancer/HCC_TypeLevel_LeadingEdge.csv"],
         analyses=["crosscancer/derive_figure_sources.py"]),
    dict(key="SuppFig_GBM_maps", title="Predicted GBM maps", builder="figures/fig_crosscancer.py: s7",
         reads=["crosscancer/GBM_Visium_maps.csv", "crosscancer/S8_full_all_ions.csv"], analyses=["crosscancer/regenerate_crosscancer.py"]),
    dict(key="SuppFig_cv_folds", title="Spatial cross-validation blocks", builder="figures/fig_supplementary.py: s2",
         reads=["within_desium_conformal.npz"], analyses=["within_benchmark.py (k-means blocks, seed 0)"]),
]

TABLES = [
    dict(key="benchmark", title="Within-section benchmark (DESIUM): per-method summary and effect sizes",
         reads=["within_desium.json", "within_controls.json"], analyses=["within_benchmark.py", "within_controls.py"]),
    dict(key="within_controls", title="Within-section controls: coordinate-only predictor and ion-panel selection",
         reads=["within_controls.json"], analyses=["within_controls.py"]),
    dict(key="flux_baselines", title="MetaboCast versus transcriptome-only metabolic inference (scFEA, MGFEA) and the Warburg test",
         reads=["baselines/baseline_comparison_per_ion.csv", "flux/flux_stroma.csv", "flux/flux_rest.csv"],
         analyses=["baselines/compare_baselines.py", "flux_validation.py"]),
    dict(key="per_ion", title="Per-ion accuracy, reliability tier and putative candidates (sheet b: all 2,086 channels)",
         reads=["within_desium_per_ion.npz", "loso_desium_per_ion.npz", "desium_ion_annotation.csv"],
         analyses=["within_benchmark.py", "loso_benchmark.py", "annotate_ions.py"]),
    dict(key="robustness", title="Source values of the robustness analyses (sheet b: feature ranking of the predictability model)",
         reads=["rigor/*.json", "penalty_grid.json", "gbm_multimodal.json"],
         analyses=["rigor_*.py (rigor_predictability.py: gradient boosting scored on held-out sections and held-out ions; "
                   "permutation importance)", "penalty_grid_check.py", "gbm_multimodal.py"]),
    dict(key="coverage", title="Coverage of the released uncertainty widths on held-out patients",
         reads=["coverage/coverage_outer.json", "within_desium.json"], analyses=["coverage_outer.py", "within_benchmark.py"]),
    dict(key="transfer", title="Transfer to unseen DESIUM sections and patients, with paired-supervision baselines",
         reads=["loso_desium.json", "loso_desium_ridge.json", "coverage/coverage_outer.json", "loso_baselines.json"],
         analyses=["loso_benchmark.py", "coverage_outer.py", "loso_baselines.py"]),
    dict(key="harmonisation", title="Transcriptome harmonisation under held-out-section transfer",
         reads=["harmonization.json"], analyses=["harmonization.py"]),
    dict(key="histology", title="Histology: H&E features and H&E-only models",
         reads=["ablations/histology_loso.csv", "ablations/histology_within.csv", "ablations/histology_only_desium.csv"],
         analyses=["histology_ablation.py", "histology_only_ablation.py"]),
    dict(key="sma_registration", title="SMA sections, subjects and registration audit",
         reads=["sma/sma_registration_audit.csv", "sma_coregistration_qc.csv"],
         analyses=["../pipeline/sma_published_pairs.R/.py", "../pipeline/sma_to_metabocast.py", "sma_registration_audit.py"]),
    dict(key="sma_within", title="Within-section benchmark on the SMA cohorts", reads=["within_sma.json"],
         analyses=["within_benchmark.py"]),
    dict(key="sma_transfer", title="SMA transfer: held-out sections and held-out animals",
         reads=["sma/loso_*.json", "sma/loso_fmp_mouse_by_animal.json"], analyses=["loso_benchmark.py"]),
    dict(key="crosscancer_summary", title="HCC and GBM applications: DESIUM accuracy of the ions used and direction counts",
         reads=["crosscancer/crosscancer_summary.json", "crosscancer/crosscancer_ion_reliability.csv",
                "ablations/histology_only_hcc_concordance.csv"],
         analyses=["crosscancer/summarise_crosscancer.py", "crosscancer/regenerate_s14_panelBC.py"]),
    dict(key="hcc_pvtt", title="HCC-2 normal to PVTT: predicted differences with DESIUM accuracy tiers",
         reads=["crosscancer/S8_supported_annotated.csv"], analyses=["crosscancer/regenerate_crosscancer.py"]),
    dict(key="hcc_patients", title="HCC tumour-normal contrasts across patients (sheet b: ridge-penalty sensitivity)",
         reads=["crosscancer/S9_and_Fig7c_source.csv", "crosscancer/HCC_NtoT_annotated_by_patient.csv",
                "crosscancer/penalty_sensitivity.json", "crosscancer/penalty_sensitivity_contrasts.csv"],
         analyses=["crosscancer/regenerate_crosscancer.py", "crosscancer/derive_figure_sources.py",
                   "crosscancer/penalty_sensitivity.py"]),
    dict(key="hcc_literature", title="Predicted HCC changes, published HCC metabolism and proposed tests",
         reads=["crosscancer/S9_and_Fig7c_source.csv", "crosscancer/S8_full_all_ions.csv"], analyses=[]),
    dict(key="gbm_protein", title="GBM CytAssist: predicted metabolites and measured proteins (exploratory)",
         reads=["crosscancer/S10_metabolite_x_protein.csv"], analyses=["crosscancer/regenerate_crosscancer.py"]),
    dict(key="kegg_prior", title="KEGG pathway prior", reads=["loso_desium_kegg.json", "sma/loso_*_kegg.json", "sma/loso_*_upper.json"],
         analyses=["loso_benchmark.py --use-kegg"]),
    dict(key="celltype", title="Cell-type signature features", reads=["ablations/celltype_ablation_*.csv"],
         analyses=["celltype_ablation.py"]),
    dict(key="model_selection", title="Model-selection screen", reads=["model_selection/*.json"],
         analyses=["model_selection/"]),
    dict(key="statistics", title="Statistics register: units, sample sizes, effects, 95% confidence intervals and Holm-adjusted P values",
         reads=["the results read by the tables above"],
         analyses=["tables/build_supplementary_tables.py (paired bootstrap intervals, Wilcoxon tests, Holm adjustment within families)"]),
]

_n = 0
for _f in FIGURES:                      # supplementary figures and tables are numbered in list (= citation) order
    if not _f["key"].startswith("Figure"):
        _n += 1
        _f["label"] = f"Figure S{_n}"
for _i, _t in enumerate(TABLES, 1):
    _t["label"] = f"Table S{_i}"

FIG = {f["key"]: f for f in FIGURES}
TAB = {t["key"]: t for t in TABLES}


def figure_stem(key):
    """File stem of a figure: the main figures keep their label; supplementary figures become FigureS<n>_<key>."""
    f = FIG.get(key)
    if f is None or not f["label"].startswith("Figure S"):
        return key
    return f"FigureS{f['label'].split('S')[1]}_{key.replace('SuppFig_', '')}"


def table_sheet(key):
    return TAB[key]["label"].replace("Table ", "")


def table_ref(key):
    """'Table S<n>' of a manifest table key."""
    return TAB[key]["label"]


def figure_ref(key, short=True):
    """'Fig. S<n>' (or 'Figure S<n>') of a manifest supplementary figure key; 'Fig. <n>' for main figures."""
    lab = FIG[key]["label"]
    return lab.replace("Figure", "Fig.") if short else lab


def write_manifest(path):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    from metabocast import __version__
    lines = ["# Analysis manifest", "",
             f"Every figure and supplementary table of the manuscript (MetaboCast {__version__}), the code that builds it, the",
             "result files it reads (`reproduce/results/`) and the analyses that wrote those files. Commands and data locations",
             "are in [`reproduce/README.md`](../reproduce/README.md); package versions are pinned in "
             "[`requirements-lock.txt`](../requirements-lock.txt).", "",
             "Generated by `python reproduce/manifest.py` from `reproduce/manifest.py`.", "",
             "## Figures", "", "| Item | Content | Built by | Reads | Written by (settings) |", "|---|---|---|---|---|"]
    for f in FIGURES:
        lines.append(f"| {f['label']} | {f['title']} | `reproduce/{f['builder']}` (file `{figure_stem(f['key'])}`) | "
                     + ", ".join(f"`{r}`" for r in f["reads"]) + " | " + "; ".join(f["analyses"]) + " |")
    lines += ["", "## Supplementary tables (Additional file 2)", "",
              "Built by `reproduce/tables/build_supplementary_tables.py` (one sheet per table).", "",
              "| Item | Content | Reads | Written by |", "|---|---|---|---|"]
    for t in TABLES:
        lines.append(f"| {t['label']} | {t['title']} | " + ", ".join(f"`{r}`" for r in t["reads"]) + " | "
                     + "; ".join(t["analyses"]) + " |")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_manifest(os.path.join(root, "docs", "ANALYSIS_MANIFEST.md"))
