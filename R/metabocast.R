# MetaboCast: R interface (a thin reticulate layer over the Python model).
#
# Predictions are made by the same pickled Python model; this layer only converts the Seurat object.
# Requires reticulate (1.36 or later for NumPy 2) and a Python environment with metabocast installed;
# select that environment before sourcing, e.g.  reticulate::use_condaenv("metabocast", required = TRUE)
#
#   source("R/metabocast.R")
#   model <- mc_load_model("metabocast_model.pkl")
#   seu   <- mc_run_rds("your_section.RDS", model, section_col = "sample")   # adds a "metabocast" assay
#   mc_spatial_plot(seu, "Glutathione")                                       # address ions by name or m/z
#
# Notes
#  * Expression is read from the "data" layer of the chosen assay (log-normalised, as after
#    Seurat::NormalizeData); raw counts are converted to log1p(CP10k). Genes are matched by symbol.
#  * Coordinates come from the object's images or, without images, from meta.data columns
#    (imagecol/imagerow, x/y, col/row, array_col/array_row).
#  * Multi-section objects: pass section_col (the meta.data column naming each tissue section) so the
#    spatial graph is built within sections.

suppressMessages({
  library(reticulate)
})

.mc_np <- NULL
.mc_sp <- NULL
.mc_ad <- NULL
.mc_mod <- NULL

.mc_init <- function() {
  if (is.null(.mc_np))  .mc_np  <<- import("numpy", delay_load = TRUE)
  if (is.null(.mc_sp))  .mc_sp  <<- import("scipy.sparse", delay_load = TRUE)
  if (is.null(.mc_ad))  .mc_ad  <<- import("anndata", delay_load = TRUE)
  if (is.null(.mc_mod)) .mc_mod <<- import("metabocast", delay_load = TRUE)
}

#' Load a saved MetaboCast model (the shipped bundle or a model saved with MetaboCast.save()).
mc_load_model <- function(path = "metabocast_model.pkl") {
  .mc_init()
  .mc_mod$MetaboCast$load(path)
}

# ------------------------------------------------------------------------------------------------
# metabolite names -> m/z for lookups by name (negative-mode adducts; exact-mass candidates, not confirmed
# identities); annotation of m/z values uses the package reference through mc_annotate_mz()
# ------------------------------------------------------------------------------------------------
.MC_MASS <- c(
  "Lactate" = 90.0317, "Pyruvate" = 88.0160, "Alanine" = 89.0477, "Serine" = 105.0426, "Glycine" = 75.0320,
  "Succinate" = 118.0266, "Fumarate" = 116.0110, "Malate" = 134.0215, "Aspartate" = 133.0375,
  "Glutamate" = 147.0532, "Glutamine" = 146.0691, "alpha-Ketoglutarate" = 146.0215,
  "Citrate/Isocitrate" = 192.0270, "Hexose" = 180.0634, "Inositol" = 180.0634, "Glucose-6-phosphate" = 260.0297,
  "Taurine" = 125.0147, "Creatine" = 131.0695, "Creatinine" = 113.0589, "Hypoxanthine" = 136.0385,
  "Inosine" = 268.0808, "Adenosine" = 267.0968, "AMP" = 347.0631, "ADP" = 427.0294, "ATP" = 506.9957,
  "Glutathione" = 307.0838, "Glutathione disulfide" = 612.1519, "Ascorbate" = 176.0321, "Urate" = 168.0283,
  "Palmitate FA16:0" = 256.2402, "Stearate FA18:0" = 284.2715, "Oleate FA18:1" = 282.2559,
  "Linoleate FA18:2" = 280.2402, "Arachidonate FA20:4" = 304.2402, "DHA FA22:6" = 328.2402,
  "Sphingosine" = 299.2824, "UDP-GlcNAc" = 607.0817, "Carnitine" = 161.1052, "Acetylcarnitine" = 203.1158,
  "Kynurenine" = 208.0848)
.MC_NEG_ADDUCTS <- c("[M-H]-" = -1.0073, "[M+Cl]-" = 34.9694, "[M+HCOO]-" = 44.9982)

#' Names of the metabolites that can be addressed by name.
mc_list_metabolites <- function() sort(names(.MC_MASS))

# exact (case-insensitive) name first, then a unique partial match
.mc_resolve_name <- function(name) {
  nm <- names(.MC_MASS)
  hit <- nm[tolower(nm) == tolower(name)]
  if (!length(hit)) hit <- nm[grepl(tolower(name), tolower(nm), fixed = TRUE)]
  if (length(hit) != 1)
    stop(sprintf("'%s' %s; see mc_list_metabolites()", name,
                 if (length(hit)) paste("is ambiguous:", paste(hit, collapse = ", ")) else "is not a known metabolite"))
  hit
}

# m/z of a named metabolite for one negative-mode adduct ([M-H]- by default)
.mc_name_to_mz <- function(name, adduct = "[M-H]-") {
  unname(.MC_MASS[.mc_resolve_name(name)] + .MC_NEG_ADDUCTS[[adduct]])
}

#' Putative annotation of m/z values ("" when nothing matches within tol, in Da), from the package's reference
#' (metabocast.annotation, the one the manuscript uses): every candidate within tol, nearest first, one name per
#' formula (Hexose covers glucose, fructose, galactose and inositol).
mc_annotate_mz <- function(mz, tol = 0.01) {
  .mc_init()
  ann <- reticulate::import("metabocast.annotation")
  as.character(unlist(ann$candidate_labels(as.numeric(mz), tol = tol)))
}

.mc_feature_mz <- function(features) suppressWarnings(as.numeric(sub("^mz[-_]", "", features)))

.mc_features <- function(obj, assay) {
  f <- rownames(obj[[assay]])
  if (is.null(f)) stop(sprintf("assay '%s' has no feature names", assay))
  f
}

# full-precision m/z stored by mc_predict(), else the value encoded in the feature names
.mc_mz_of <- function(obj, assay, feats) {
  mz <- tryCatch(obj@misc$metabocast_mz, error = function(e) NULL)
  if (!is.null(mz) && length(mz) == length(feats)) as.numeric(mz) else .mc_feature_mz(feats)
}

#' Feature name of an ion in the "metabocast" assay, addressed by metabolite name, m/z or feature name.
#' Names resolve to the first negative-mode adduct ([M-H]-, [M+Cl]-, [M+HCOO]-) present within tol (Da).
mc_feature <- function(obj, x, assay = "metabocast", tol = 0.01) {
  feats <- .mc_features(obj, assay)
  if (is.character(x) && x %in% feats) return(x)
  fmz <- .mc_mz_of(obj, assay, feats)
  targets <- if (is.numeric(x)) x else vapply(names(.MC_NEG_ADDUCTS), function(a) .mc_name_to_mz(x, a), numeric(1))
  for (t in targets) {
    d <- abs(fmz - t)
    if (any(d <= tol, na.rm = TRUE)) return(feats[which.min(d)])
  }
  stop(sprintf("no ion within %.3f Da of %s in assay '%s'", tol, if (is.numeric(x)) format(x) else sQuote(x), assay))
}

#' Every annotated ion of the "metabocast" assay: feature | m/z | putative metabolite.
mc_annotation_table <- function(obj, assay = "metabocast", tol = 0.01) {
  feats <- .mc_features(obj, assay)
  mz <- .mc_mz_of(obj, assay, feats)
  ann <- mc_annotate_mz(mz, tol)
  keep <- ann != ""
  data.frame(feature = feats[keep], mz = mz[keep], metabolite = ann[keep], stringsAsFactors = FALSE)
}

#' Spatial map of a predicted ion addressed by name or m/z (needs an object with images).
mc_spatial_plot <- function(obj, x, assay = "metabocast", ...) {
  f <- mc_feature(obj, x, assay = assay)
  Seurat::DefaultAssay(obj) <- assay
  Seurat::SpatialFeaturePlot(obj, features = f, ...)
}

# ------------------------------------------------------------------------------------------------
# Seurat -> AnnData and prediction
# ------------------------------------------------------------------------------------------------
# pandas DataFrame -> R data.frame through a temporary CSV (independent of pandas/reticulate versions)
.mc_py_df <- function(df) {
  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp))
  df$to_csv(tmp, index = FALSE)
  utils::read.csv(tmp, stringsAsFactors = FALSE, check.names = FALSE)
}

.mc_layer <- function(seu, assay, layer) {
  tryCatch(SeuratObject::GetAssayData(seu, assay = assay, layer = layer),
           error = function(e) SeuratObject::GetAssayData(seu, assay = assay, slot = layer))
}

.mc_is_counts <- function(X) {
  v <- if (methods::is(X, "sparseMatrix")) X@x else as.vector(X)
  v <- v[seq_len(min(length(v), 20000L))]
  length(v) > 0 && all(v >= 0) && all(v == round(v))
}

.mc_coordinates <- function(seu) {
  cells <- colnames(seu)
  xy <- tryCatch({
    tc <- Seurat::GetTissueCoordinates(seu)
    if ("cell" %in% colnames(tc)) rownames(tc) <- tc$cell
    tc <- tc[, setdiff(colnames(tc), "cell"), drop = FALSE]
    if (all(c("imagecol", "imagerow") %in% colnames(tc))) tc <- tc[, c("imagecol", "imagerow")]
    as.matrix(tc[cells, 1:2])
  }, error = function(e) NULL)
  if (!is.null(xy) && nrow(xy) == length(cells) && all(is.finite(xy))) return(xy)
  md <- seu@meta.data
  for (p in list(c("imagecol", "imagerow"), c("x", "y"), c("col", "row"), c("array_col", "array_row"))) {
    if (all(p %in% colnames(md))) return(as.matrix(md[cells, p]))
  }
  stop("no spatial coordinates: the object has no images and meta.data has no imagecol/imagerow, x/y, col/row ",
       "or array_col/array_row columns")
}

.mc_seurat_to_anndata <- function(seu, assay = NULL, section_col = NULL) {
  .mc_init()
  assay <- if (is.null(assay)) SeuratObject::DefaultAssay(seu) else assay
  X <- .mc_layer(seu, assay, "data")                                     # genes x spots
  if (!length(X) || .mc_is_counts(X)) {                                  # not normalised: log1p(CP10k)
    C <- methods::as(.mc_layer(seu, assay, "counts"), "CsparseMatrix")
    X <- methods::as(C %*% Matrix::Diagonal(x = 1e4 / pmax(Matrix::colSums(C), 1)), "CsparseMatrix")
    X@x <- log1p(X@x)
    dimnames(X) <- dimnames(C)
  }
  X <- methods::as(Matrix::t(methods::as(X, "CsparseMatrix")), "CsparseMatrix")   # spots x genes
  ad <- .mc_ad$AnnData(X = .mc_sp$csr_matrix(r_to_py(X)),
                       obs = r_to_py(data.frame(row.names = rownames(X))))
  ad$var_names <- as.list(colnames(X))
  ad$layers["log1p"] <- ad$X
  ad$obsm["spatial"] <- .mc_np$asarray(.mc_coordinates(seu)[rownames(X), , drop = FALSE], dtype = "float64")
  if (!is.null(section_col)) {
    if (!section_col %in% colnames(seu@meta.data)) stop(sprintf("section_col '%s' is not a meta.data column", section_col))
    ad$obs[section_col] <- as.character(seu@meta.data[rownames(X), section_col])
  }
  ad
}

#' Predict the spatial metabolome of a Seurat object and add a "metabocast" assay (ions x spots).
#' @param seu         Seurat object with spatial coordinates (images or meta.data columns)
#' @param model       a model returned by mc_load_model()
#' @param assay       expression assay (default: the object's default assay)
#' @param section_col optional meta.data column naming each tissue section
#' @return the object with a "metabocast" assay (features named "mz-<m/z>") and, in misc, the per-ion
#'         reliability table, the m/z axis and the gene-panel overlap
mc_predict <- function(seu, model, assay = NULL, section_col = NULL) {
  ad <- .mc_seurat_to_anndata(seu, assay = assay, section_col = section_col)
  pred <- py_to_r(model$predict_metabolome(ad, section_key = section_col))        # spots x ions
  mz <- as.numeric(py_to_r(model$mz_))
  colnames(pred) <- make.unique(sprintf("mz-%.3f", mz))
  rownames(pred) <- unlist(py_to_r(ad$obs_names$tolist()))
  seu[["metabocast"]] <- SeuratObject::CreateAssayObject(data = t(pred))
  overlap <- tryCatch(as.numeric(py_to_r(model$last_gene_overlap_)), error = function(e) NA_real_)
  if (is.finite(overlap) && overlap < 0.5)
    warning(sprintf("only %.0f%% of the model's genes are in the query; predictions are extrapolative", 100 * overlap))
  rel <- tryCatch(.mc_py_df(model$reliability_table(ad)), error = function(e) NULL)
  if (!is.null(rel)) {
    rel$feature <- colnames(pred)
    seu@misc$metabocast_reliability <- rel
  }
  seu@misc$metabocast_mz <- mz
  seu@misc$metabocast_gene_overlap <- overlap
  seu
}

#' Read a Seurat .RDS, predict its metabolome and optionally save the per-ion table and the object.
mc_run_rds <- function(rds_path, model, assay = NULL, section_col = NULL,
                       out_csv = "predicted_metabolites.csv", out_rds = NULL) {
  seu <- mc_predict(readRDS(rds_path), model, assay = assay, section_col = section_col)
  if (!is.null(out_csv) && !is.null(seu@misc$metabocast_reliability))
    utils::write.csv(seu@misc$metabocast_reliability, out_csv, row.names = FALSE)
  if (!is.null(out_rds)) saveRDS(seu, out_rds)
  seu
}
