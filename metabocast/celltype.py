"""Cell-type-aware feature block for MetaboCast (optional input).

REDCAT (Li et al., Nat. Methods 2026) showed that metabolic state in intact tissue is
strongly cell-type-specific, most visibly for lipids and redox species. This module gives
MetaboCast a cheap, reference-free approximation of that signal: per-spot lineage-signature
scores computed directly from the transcriptome (mean of z-scored marker-gene expression per
major lineage). Attach them to `adata.obsm[key]` and pass `extra_key=key` to MetaboCast, which
concatenates the block and one neighbour-diffused copy onto the diffusion features — i.e. the
same value + diffusion treatment the diffusion features use, so the cell-type signal is spatially
smoothed consistently.

The shipped model remains transcriptome-only; this is an opt-in retraining/ablation input.
Use `add_celltype_features(adata)` before `fit(...)` / `predict_metabolome(...)`.
"""
import numpy as np

# Major lineage marker sets (broad, tumour-tissue oriented; extend/replace per tissue).
LINEAGE_MARKERS = {
    "Tcell_NK":    ["CD3D", "CD3E", "CD8A", "CD2", "NKG7", "GZMB", "IL7R", "CCL5"],
    "B_plasma":    ["MS4A1", "CD79A", "CD79B", "IGHG1", "MZB1", "JCHAIN"],
    "Myeloid":     ["LYZ", "CD68", "CD14", "ITGAX", "C1QA", "C1QB", "AIF1", "FCGR3A"],
    "Epithelial":  ["EPCAM", "KRT8", "KRT18", "KRT19", "ELF3", "CDH1", "KRT5"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "CDH5", "EGFL7"],
    "Fibroblast":  ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB", "ACTA2", "THY1"],
    "Immune_all":  ["PTPRC"],
}
LINEAGE_NAMES = list(LINEAGE_MARKERS)


def _log1p_matrix(adata):
    if "log1p" in adata.layers:
        X = adata.layers["log1p"]
    else:
        X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    if "log1p" not in adata.layers and float(X.max()) > 50.0:   # looks like raw counts
        X = np.log1p(X)
    return X


def add_celltype_features(adata, key="celltype", markers=None, layer="log1p"):
    """Compute per-spot lineage-signature scores and store them in ``adata.obsm[key]``.

    Each column is the mean of the per-section z-scored expression of that lineage's markers
    present in ``adata.var_names`` (missing markers are skipped; a lineage with no markers
    present yields a zero column). Returns the (n_spots x n_lineages) array. Column order is
    ``[name for name in (markers or LINEAGE_MARKERS)]`` and is also stored in
    ``adata.uns[key + '_names']``.
    """
    mk = markers or LINEAGE_MARKERS
    X = _log1p_matrix(adata)
    gi = {g: i for i, g in enumerate(adata.var_names)}
    cols, names = [], []
    for lin, genes in mk.items():
        idx = [gi[g] for g in genes if g in gi]
        names.append(lin)
        if not idx:
            cols.append(np.zeros(adata.n_obs, np.float32)); continue
        sub = X[:, idx]
        z = (sub - sub.mean(0)) / (sub.std(0) + 1e-8)
        cols.append(z.mean(1).astype(np.float32))
    block = np.vstack(cols).T.astype(np.float32)
    adata.obsm[key] = block
    adata.uns[key + "_names"] = names
    return block
