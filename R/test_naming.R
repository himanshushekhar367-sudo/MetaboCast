# =====================================================================================
# test_naming.R — reproducibility unit test for the metabolite-naming layer in metabocast.R
# Needs NO Python, Seurat, or data — it stubs a tiny object and checks the pure-R logic.
# Run from the R/ folder:   Rscript test_naming.R
# Expect: every line PASS, final "N/N checks passed.", exit status 0.
# =====================================================================================
requireNamespace <- function(...) TRUE          # stub package loading so the naming layer runs without reticulate
library          <- function(...) invisible(NULL)
source("metabocast.R")

.chk <- new.env(); .chk$pass <- 0L; .chk$n <- 0L
ok <- function(cond, msg) {
  .chk$n <- .chk$n + 1L; if (isTRUE(cond)) .chk$pass <- .chk$pass + 1L
  cat(if (isTRUE(cond)) "  PASS " else "  FAIL ", msg, "\n")
}

cat("mc_list_metabolites():\n")
ml <- mc_list_metabolites()
ok(is.character(ml) && "Glutathione" %in% ml && "DHA FA22:6" %in% ml, "sorted names incl. Glutathione & DHA FA22:6")

cat("\n.mc_name_to_mz():\n")
ok(abs(.mc_name_to_mz("Glutathione") - (307.0838 - 1.0073)) < 1e-6, "Glutathione -> 306.0765 ([M-H]-)")
ok(abs(.mc_name_to_mz("glutathione") - 306.0765) < 1e-6,            "case-insensitive")
ok(abs(.mc_name_to_mz("DHA") - (328.2402 - 1.0073)) < 1e-6,         "partial 'DHA' -> DHA FA22:6 (327.2329)")
ok(inherits(try(.mc_name_to_mz("unobtainium"), silent = TRUE), "try-error"), "unknown name errors")

cat("\nmc_annotate_mz():\n")
a <- mc_annotate_mz(c(306.076, 327.231, 999.0))
ok(a[1] == "Glutathione [M-H]-" && a[2] == "DHA FA22:6 [M-H]-" && a[3] == "", "annotates 306/327, blanks unknown")

cat("\nmc_feature() with a stub object (list; obj[[assay]] has rownames):\n")
fake <- list(metabocast = matrix(0, 3, 2,
        dimnames = list(c("mz-306.076", "mz-327.231", "mz-255.231"), NULL)))
ok(mc_feature(fake, "Glutathione") == "mz-306.076", "name  'Glutathione' -> mz-306.076")
ok(mc_feature(fake, "DHA")         == "mz-327.231", "name  'DHA'         -> mz-327.231")
ok(mc_feature(fake, 255.23)        == "mz-255.231", "number 255.23       -> mz-255.231")
ok(mc_feature(fake, "mz-327.231")  == "mz-327.231", "exact 'mz-...' string kept")

cat(sprintf("\n%d/%d checks passed.\n", .chk$pass, .chk$n))
quit(status = if (.chk$pass == .chk$n) 0L else 1L)
