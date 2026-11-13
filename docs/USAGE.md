# Using MetaboCast

This guide covers prediction on your own spatial transcriptomics data, training on paired data, optional inputs,
pathway activity and the interpretation of the output. The [README](../README.md) has the short version.

## 1. Install

```bash
git clone https://github.com/himanshushekhar367-sudo/MetaboCast.git
cd MetaboCast
pip install -e .
```

## 2. Which route fits your data?

| You have | Do this | You get |
|---|---|---|
| Spatial transcriptomics only | apply a trained model, such as the shipped `metabocast_model.pkl` | predicted metabolite maps with per-ion reliability, as hypotheses for follow-up |
| Spatial transcriptomics and MSI on the same sections | train a model on your sections, then predict new sections from the same system | a model for your m/z panel, tissue and species, with its own held-out accuracy |

The shipped model was trained on seven paired DESI-MSI/Visium sections of human breast and lung cancer (DESIUM)
and predicts 2,086 negative-mode DESI-MSI m/z channels. The public DESIUM and SMA cohorts
([`../data/README.md`](../data/README.md)) can be used to train models for other panels.

## 3. Prepare a query

A query needs gene expression and two-dimensional coordinates per spot, cell or bin. Read your data into AnnData
(for example with `scanpy.read_visium`, `scanpy.read_10x_h5`, `squidpy.read.vizgen`, `squidpy.read.nanostring` or
`spatialdata-io`) and pass it to `prepare_query`:

```python
import anndata as ad
from metabocast import prepare_query

q = prepare_query(ad.read_h5ad("section.h5ad"))
```

`prepare_query` returns a new AnnData with gene symbols as `var_names`, log1p(CP10k) expression in
`layers["log1p"]` and coordinates in `obsm["spatial"]`, and records its choices in `uns["metabocast_input"]`.

| Argument | Use |
|---|---|
| `spatial_key` | coordinates stored elsewhere: an `obsm` key, or `"xcol,ycol"` naming two `obs` columns |
| `symbol_column` | the `var` column holding gene symbols (found automatically when `var_names` are Ensembl IDs) |
| `normalize` | `"auto"` (default): normalise raw counts, keep an existing `layers["log1p"]`; `"raw"`: always normalise; `"none"`: `X` is already log1p(CP10k) |
| `counts_layer` | the layer holding raw counts, if `X` holds something else |
| `bin_size` | sum observations into square pseudo-spots of this side length, in coordinate units (for example 50 for Xenium or MERSCOPE coordinates in micrometres) |
| `section_key` | `obs` column separating the sections of a multi-section object; bins and the spatial graph stay within sections |
| `uppercase_genes` | upper-case gene symbols for approximate cross-species matching (training on paired data from the query species is better) |

Coordinates are found automatically in `obsm["spatial"]` (Visium, Visium HD, Slide-seq, Stereo-seq), in the Xenium
(`x_centroid`, `y_centroid`), CosMx (`CenterX_global_px`, `CenterY_global_px`) and MERSCOPE (`center_x`, `center_y`)
cell tables, in Space Ranger pixel columns and in generic `x`/`y` columns.

Why bin single-cell data? The shipped model was trained on Visium spots, each pooling transcripts from several
cells. Single cells measured on a targeted panel are much sparser, and pooling them into pseudo-spots of about
50 µm brings their profiles closer to the training data. Coordinates must describe the real tissue geometry,
because the model diffuses expression over a spatial neighbour graph.

## 4. Predict

Command line:

```bash
python examples/predict_on_your_data.py --model metabocast_model.pkl --query section.h5ad --out results/ --plot
# single-cell platforms, for example Xenium:
python examples/predict_on_your_data.py --model metabocast_model.pkl --query xenium.h5ad --bin-size 50 --out results/
```

The script prints the gene-panel overlap and warns when the query looks like another species or lacks most of the
model's highly variable genes. It writes `query_with_predictions.h5ad`, `predicted_metabolites.csv` (one row per
m/z channel, ordered by predicted spatial variation) and, with `--plot`, `predicted_maps.png`.
`examples/quickstart.py section.h5ad` is a minimal version that writes the spots-by-ions predictions to a CSV file.

Python:

```python
from metabocast import MetaboCast, gene_panel_overlap

model = MetaboCast.load("metabocast_model.pkl")
overlap = gene_panel_overlap(model, q)          # fractions of training genes and highly variable genes measured
pred = model.predict_metabolome(q)              # spots x m/z channels, log1p intensity
rel = model.reliability_table(q)                # one row per m/z channel
mz = model.mz_                                  # m/z of each prediction column
```

| Where | What |
|---|---|
| `q.obsm["metabocast_pred"]` | predicted log1p intensity (spots x m/z channels) |
| `q.obsm["metabocast_pred_rank"]` | the same predictions on the rank (Gaussian score) scale used in training |
| `q.uns["metabocast_conf_width"]` | per-ion 90% half-width from leave-one-section-out residuals (log1p units) |
| `q.uns["metabocast_conf_width_adj"]` | the half-width widened in proportion to the training genes missing from the query |
| `q.uns["metabocast_uncertainty"]` | how the widths were estimated and their coverage on held-out patients (`outer_test_coverage`) |
| `model.last_gene_overlap_`, `model.last_hvg_overlap_` | fractions of training genes and highly variable genes present in the last query |

`reliability_table()` columns: `mz`, `pred_mean`, `pred_std`, `conf_width`, `train_predictability_spearman`
(accuracy within DESIUM sections), `transfer_predictability_spearman` (accuracy on unseen DESIUM sections),
`annotation` (putative exact-mass candidates), `chemical_class`, `trust_tier` (transfer-supported, within-section
only, low or uncalibrated), `uncertainty_method`, `calibration_unit`. `focus="lipid_redox"` keeps
lipid, nucleotide and redox ions; `celltype_key` adds cell-state summaries (section 6).

R and Seurat:

```r
source("R/metabocast.R")
model <- mc_load_model("metabocast_model.pkl")
seu   <- mc_run_rds("section.RDS", model, section_col = "sample")      # or mc_predict(seu, model)
mc_spatial_plot(seu, "Glutathione")                                    # map of an ion by name or m/z (objects with images)
mc_feature(seu, "DHA")                                                 # feature name of an ion in the "metabocast" assay
head(mc_annotation_table(seu))                                         # annotated ions: feature | m/z | metabolite
head(seu@misc$metabocast_reliability)                                  # per-ion reliability table
```

The R functions call the same Python model through reticulate (1.36 or later for NumPy 2). Expression is read from the `data` layer of the
default assay (raw counts are normalised to log1p(CP10k)); coordinates come from the object's images or from
`imagecol`/`imagerow`, `x`/`y` or `array_col`/`array_row` columns of `meta.data`. `Rscript examples/predict_on_your_data.R`
runs the same steps from the command line.

## 5. Train on paired data

Each training section needs:

| Field | Contents | Shape |
|---|---|---|
| `layers["log1p"]` or `X` | gene expression, log1p(CP10k) | spots x genes |
| `obsm["spatial"]` | coordinates | spots x 2 |
| `uns["msi"]` | MSI intensities (non-negative) | spots x ions |
| `uns["mz_features"]` | m/z of each ion | ions |

Sections pooled in one model must share one m/z axis. `pipeline/train_and_evaluate.py` finds the paired sections
in a folder, groups them into cohorts that can be pooled (same m/z axis, same gene-symbol case), saves one model
per cohort and reports leave-one-section-out accuracy:

```bash
python pipeline/train_and_evaluate.py ./paired_sections --out results/training      # options: --use-kegg, --estimator ridge
```

In Python:

```python
import anndata as ad
from metabocast import MetaboCast

train = [ad.read_h5ad(f) for f in ["sectionA.h5ad", "sectionB.h5ad", "sectionC.h5ad"]]
model = MetaboCast().fit(train)
model.save("my_model.pkl")
```

`examples/train_on_paired_data.py --data ./sections --out my_model.pkl` does the same from the command line.

How the default model works. Genes are reduced to principal components fitted on the training sections (3,000
highly variable genes, 100 components). Each component is diffused over a six-nearest-neighbour spot graph,
giving the diffusion features [P, ÂP, Â²P], a simple encoding of local spatial context. Each ion is rank-transformed to Gaussian scores within each
training section, which removes differences in intensity scale between MSI acquisitions. A ridge regression, whose
penalty is chosen by leave-one-section-out cross-validation (or over five spatial blocks for a single section),
and 100 extremely randomized trees, fitted to 50 principal components of the targets, are averaged after scaling
to equal spread. Predictions are mapped back to log1p intensity through each ion's training quantiles.

| Option | Effect |
|---|---|
| `estimator="ridge"` | linear half only (faster, slightly less accurate) |
| `alpha=<number>` | fixed ridge penalty instead of cross-validation |
| `alpha_grid=[...]` | penalties searched by the cross-validation (default `metabocast.metabocast.ALPHA_GRID`, 10⁻¹ to 10⁷); a `RuntimeWarning` reports a selection at the largest value, where a wider grid shows whether larger penalties fit better |
| `target_transform=None` | fit per-ion standardised log1p intensities instead of rank-normalised targets |
| `use_kegg=True` | add KEGG metabolic-pathway activity features (bundled human gene sets; `kegg_gmt=` for your own) |
| `gene_scaling="section"` | z-score genes within each section; can help transfer between batches but removes real between-section differences, so avoid it for contrasts between sections |
| `extra_key="<obsm key>"` | append a per-spot modality (section 6) |
| `n_jobs=-1` | fit the trees on all cores |

Uncertainty for a retrained model. Keep independent calibration sections out of training, then run

```python
model.calibrate(calibration_sections, q=0.90, unit="section")
```

Section mode takes the largest residual of each ion within each calibration section and assumes the sections are
independent and exchangeable; with fewer than nine sections a 90% width is infinite. `unit="spot"` assumes
exchangeable spots, which neighbouring tissue spots rarely are. Calibration also stores each ion's accuracy on
the calibration sections, averaged over sections, as `transfer_predictability_`; with the within-section score
`train_predictability_` it sets the tiers of `reliability_table()`.

## 6. Optional inputs

Per-spot modalities measured on the same spots (antibody-derived protein, H&E features, cell-type signatures) are
appended with `extra_key`:

```python
model = MetaboCast(extra_key="protein").fit(train)          # train[i].obsm["protein"]: spots x markers
```

Cell-type signatures computed from the transcriptome:

```python
from metabocast import MetaboCast, add_celltype_features

for a in train + [query]:
    add_celltype_features(a)                                 # obsm["celltype"]: lineage marker-set scores
model = MetaboCast(extra_key="celltype").fit(train)
pred = model.predict_metabolome(query)
rel = model.reliability_table(query, focus="lipid_redox", celltype_key="celltype")
```

H&E patch features from Visium images (`add_histology_features`, `examples/add_histology_features.py`) are used the
same way. `MorphologyMetabolitePredictor` is a separate, experimental model that predicts from H&E features alone;
treat its output as morphology-metabolite coherence, not metabolomics. The shipped model uses the transcriptome
only.

## 7. Metabolic pathway activity

Per-spot pathway activity from enzyme expression, in the style of scMetabolism, with optional spatial smoothing
(gene-programme scores, not metabolite measurements):

```bash
python examples/pathway_scores.py --query section.h5ad --gmt data/genesets/scMetab_KEGG.gmt --out pathways/ --method aucell --smooth 1 --plot
```

```python
from metabocast import PathwayScorer
scores = PathwayScorer("data/genesets/scMetab_KEGG.gmt", method="aucell").score(adata, spatial_smooth=1)
```

`method="aucell"` follows scMetabolism's rank-based AUCell scoring and `"mean_z"` averages z-scored member genes;
`spatial_smooth` diffuses the scores over the tissue neighbour graph. Bundled gene sets: `scMetab_KEGG.gmt` (85 KEGG
metabolic pathways) and `scMetab_REACT.gmt`.

Enzyme expression alone does not show whether a pathway's products are present. `MetabolicActivity` scores a
pathway where its enzymes and its annotated metabolites agree, using predicted or measured metabolites:

```bash
python examples/metabolic_activity.py --query section.h5ad --model metabocast_model.pkl --out metabolic/ --plot
python examples/metabolic_activity.py --query paired_section.h5ad --measured --out metabolic/     # measured MSI
```

```python
from metabocast import MetabolicActivity
activity, confidence, source = MetabolicActivity("data/genesets/scMetab_KEGG.gmt", adducts="neg").score(adata, model=model)
```

`confidence` ranks pathways by enzyme-metabolite coherence and flags whether metabolites supported the score.
Coherence between enzyme expression and metabolite abundance is not a flux measurement, and predicted metabolites
carry their own error. Annotation uses negative-mode adducts by default (`adducts="pos"` for positive mode).

## 8. Interpreting predictions

Treat a prediction as a hypothesis worth measuring when all of the following hold:

1. the query resembles the training tissue and platform, and its species matches the model;
2. gene-panel overlap is high (`gene_panel_overlap`; the command-line tool warns below half of the model's highly
   variable genes);
3. the ion is transfer-supported in `reliability_table()` (accuracy on unseen sections,
   `transfer_predictability_spearman`, at least 0.20); a within-section-only ion is a lead at most;
4. the predicted map is spatially structured.

What to expect. Within paired sections the shipped model reaches a median Spearman correlation of 0.34 on
well-measured ions, part of it spatial structure that a coordinate-only smoother also captures (0.23). On unseen
DESIUM sections the median over all detected ions is 0.045; 310 of the 2,086 channels reach 0.20, and adding H&E
features improved transfer in all six sections tested (an H&E-only model was better than the transcriptome in four
of six). In the SMA MALDI cohorts, with the spot-to-pixel pairing published
with the deposit, transfer reached 0.32 for DHB lipids between two sections of one mouse and 0.10-0.29 for FMP
neurotransmitters in mouse brain; those cohorts differ from DESIUM in chemistry, species, tissue and design.
Predictability tracked spatial autocorrelation and varied across putative classes and cohorts; these comparisons do
not isolate biochemical regulation from acquisition and tissue effects. Names are putative exact-mass candidates;
confirm identities by MS/MS or FDR-controlled annotation such as METASPACE.

## 9. End-to-end example

```bash
# train on the public DESIUM cohort (data/README.md) and report leave-one-section-out accuracy
python pipeline/train_and_evaluate.py /path/to/DESIUM/Correlation_NMF_analysis/data --out results/desium

# predict a Visium section with the shipped model
python examples/predict_on_your_data.py --model metabocast_model.pkl --query my_visium_section.h5ad --out my_results/ --plot
```
