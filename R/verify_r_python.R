# =====================================================================================
# verify_r_python.R: check, in R, that the shipped MetaboCast model reproduces the direction of the
# HCC-2 normal -> portal-vein tumour thrombus (PVTT) changes reported in the manuscript for the
# seven literature-expected metabolites. R is a thin layer over the Python model (reticulate calls the
# same object); this script checks the result from R.
#
# Usage: Rscript R/verify_r_python.R <MetaboCast repository> <HCC-2 Seurat .RDS> [out.RDS]
#   The Seurat object needs the HCC-2 expression, spot coordinates (images or imagecol/imagerow in
#   meta.data) and the region labels in meta.data$sample.ident (HCC-2N, -2L, -2P, -2T).
# =====================================================================================
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript R/verify_r_python.R <repository> <HCC-2.RDS> [out.RDS]")
REPO <- args[[1]]; HCC2 <- args[[2]]
OUT  <- if (length(args) >= 3) args[[3]] else NULL
source(file.path(REPO, "R", "metabocast.R"))

model <- mc_load_model(file.path(REPO, "metabocast_model.pkl"))
obj   <- mc_run_rds(HCC2, model, section_col = "sample.ident", out_csv = NULL, out_rds = OUT)

## region means (N -> L -> P -> T), ions addressed by name
D   <- SeuratObject::GetAssayData(obj, assay = "metabocast", layer = "data")
sec <- factor(sub("^.*([NLPT])$", "\\1", as.character(obj$sample.ident)), levels = c("N", "L", "P", "T"))
mean_by <- function(name) { f <- mc_feature(obj, name); tapply(as.numeric(D[f, ]), sec, mean) }

report <- function(name, updown) {
  m <- mean_by(name); d <- as.numeric(m["P"] - m["N"])
  ok <- if (updown == "up") d > 0 else d < 0
  cat(sprintf("  %-20s N/L/P/T = %6.3f %6.3f %6.3f %6.3f   P - N = %+.3f   %s\n",
              name, m["N"], m["L"], m["P"], m["T"], d, if (ok) "as expected" else "NOT as expected"))
  ok
}
cat(sprintf("gene-panel overlap with the model: %.1f%%\n", 100 * obj@misc$metabocast_gene_overlap))
cat("\nHCC-2 predicted normal -> PVTT changes (literature-expected direction in brackets):\n")
res <- c(
  report("Glutathione",         "up"),
  report("Ascorbate",           "up"),
  report("Arachidonate FA20:4", "up"),
  report("DHA FA22:6",          "up"),
  report("Palmitate FA16:0",    "down"),
  report("Stearate FA18:0",     "down"),
  report("Hexose",              "down")
)
cat(sprintf("\n%d of %d expected directions reproduced.\n", sum(res), length(res)))
