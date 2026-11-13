#!/usr/bin/env Rscript
# Export the spot-to-pixel pairing published with the SMA deposit (Vicari et al., Nat Biotechnol 2024).
#
# Usage:
#   Rscript pipeline/sma_published_pairs.R <folder with the deposit's misc/ files> <out dir>
#   then: python pipeline/sma_published_pairs.py --sma <folder containing the 'sma' tree> --export <out dir> \
#             --metadata <deposit metadata.csv> --out pipeline/sma_published_pairs
#
# The two inputs are in misc.zip of the deposit (misc/knn_spatial_df_filtered_list, misc/hPDStr.multi.list).
#
# Mouse brain, 9 sections: knn_spatial_df_filtered_list holds, per section, the MALDI pixel ("<x>x<y>", the x and y
#   columns of the section's *_smamsi.csv raster) paired with each Visium spot (barcode + "_1"). The authors aligned the
#   MALDI and Visium images by hand in STUtility and paired each spot with its nearest pixel, dropping pairs more than
#   35 image pixels apart. Written as <out>/<section>.csv (barcode, msi_x, msi_y).
# Human striatum, 4 sections: hPDStr.multi.list holds one Seurat object per section whose MSI assay carries, for each
#   spot, the spectrum of the pixel the authors paired with it. The spectra are written to
#   <out>/human_msi/<section>.csv.gz; sma_published_pairs.py finds the raster pixel with the identical spectrum.
#
# Needs R with SeuratObject and Matrix.
suppressMessages({ library(SeuratObject); library(Matrix) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript sma_published_pairs.R <misc folder> <out dir>")
src <- args[1]; out <- args[2]
dir.create(file.path(out, "human_msi"), recursive = TRUE, showWarnings = FALSE)

knn <- readRDS(file.path(src, "knn_spatial_df_filtered_list"))
for (sec in names(knn)) {
  d <- knn[[sec]]
  stopifnot(!anyDuplicated(d$to), !anyDuplicated(d$from))
  xy <- do.call(rbind, strsplit(as.character(d$from), "x", fixed = TRUE))
  df <- data.frame(barcode = sub("_[0-9]+$", "", d$to), msi_x = as.integer(xy[, 1]), msi_y = as.integer(xy[, 2]))
  write.csv(df, file.path(out, paste0(sec, ".csv")), row.names = FALSE, quote = FALSE)
  cat(sprintf("%s: %d pairs (largest distance %.1f px)\n", sec, nrow(df), max(d$distance)))
}

assay_counts <- function(o) {
  tryCatch(GetAssayData(o, assay = "MSI", layer = "counts"),
           error = function(e) GetAssayData(o, assay = "MSI", slot = "counts"))
}
hum <- readRDS(file.path(src, "hPDStr.multi.list"))
for (o in hum) {
  sec <- unique(as.character(o@meta.data$sample_id)); stopifnot(length(sec) == 1)
  m <- assay_counts(o)
  df <- data.frame(barcode = sub("_[0-9]+$", "", colnames(m)), as.matrix(t(m)), check.names = FALSE)
  gz <- gzfile(file.path(out, "human_msi", paste0(sec, ".csv.gz")), "w")
  write.csv(df, gz, row.names = FALSE); close(gz)
  cat(sprintf("%s: %d spots x %d m/z exported\n", sec, ncol(m), nrow(m)))
}
