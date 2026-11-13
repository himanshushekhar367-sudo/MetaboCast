# =====================================================================================
# predict_on_your_data.R: apply a trained MetaboCast model to spatial transcriptomics stored as a
# Seurat .RDS. No MSI and no training are needed.
#
#   Rscript examples/predict_on_your_data.R <model.pkl> <query.RDS> [out.RDS] [section column]
#
# The query needs gene expression (gene symbols) and spatial coordinates (images, or imagecol/imagerow
# or x/y columns in meta.data). For objects holding several tissue sections, give the meta.data column
# that names the sections. Predictions are added as a "metabocast" assay; the per-ion reliability table
# is written next to the output object.
# Set METABOCAST_REPO to your MetaboCast folder if the model file is not in it.
# =====================================================================================
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript predict_on_your_data.R <model.pkl> <query.RDS> [out.RDS] [section column]")
model_pkl <- args[[1]]; query_rds <- args[[2]]
out_rds   <- if (length(args) >= 3) args[[3]] else sub("\\.RDS$", "_with_metabocast.RDS", query_rds, ignore.case = TRUE)
section   <- if (length(args) >= 4) args[[4]] else NULL

REPO_DIR <- Sys.getenv("METABOCAST_REPO", dirname(normalizePath(model_pkl)))
source(file.path(REPO_DIR, "R", "metabocast.R"))

model <- mc_load_model(model_pkl)
obj   <- mc_run_rds(query_rds, model, section_col = section, out_rds = out_rds,
                    out_csv = sub("\\.RDS$", "_metabocast_ions.csv", out_rds, ignore.case = TRUE))

cat(sprintf("\nGene-panel overlap with the model: %.1f%%\n", 100 * obj@misc$metabocast_gene_overlap))
cat("Predicted metabolome saved to:\n  ", out_rds, "\n")
cat("Address ions by name or m/z, e.g.:\n")
cat("  DefaultAssay(obj) <- 'metabocast'\n")
cat("  SpatialFeaturePlot(obj, mc_feature(obj, 'Glutathione'))   # or mc_spatial_plot(obj, 'Glutathione')\n")
cat("  head(mc_annotation_table(obj), 20)                         # every annotated ion: feature | m/z | metabolite\n")
cat("  head(obj@misc$metabocast_reliability)                      # per-ion accuracy and trust tier\n")
