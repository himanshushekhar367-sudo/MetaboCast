"""Bring spatial transcriptomics data from any platform into the form MetaboCast expects.

MetaboCast needs, per spot (or cell, or bin):
  * gene expression with gene SYMBOLS as ``var_names``, as log1p(CP10k) in ``layers["log1p"]``;
  * two-dimensional coordinates in ``obsm["spatial"]``.

Platforms store these differently. :func:`prepare_query` finds the coordinates (Visium/Visium HD,
Slide-seq, Stereo-seq, Xenium, CosMx, MERSCOPE/MERFISH and generic ``x``/``y`` columns), swaps
Ensembl identifiers for symbols when a symbol column is present, normalises raw counts, and can
aggregate single-cell or subcellular data into square pseudo-spots so that a model trained on
spot-level data sees profiles of comparable depth. Nothing here changes how a model predicts; it
only prepares the input and records what was done in ``uns["metabocast_input"]``.
"""
import re
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

# coordinate locations, most specific first
COORD_OBSM_KEYS = ("spatial", "X_spatial", "spatial_coords", "spatial_coordinates", "X_spatial_coords")
COORD_OBS_PAIRS = (
    ("x_centroid", "y_centroid"),                    # 10x Xenium (micrometres)
    ("CenterX_global_px", "CenterY_global_px"),      # NanoString CosMx (pixels)
    ("center_x", "center_y"),                        # Vizgen MERSCOPE (micrometres)
    ("x_location", "y_location"),                    # 10x Xenium transcript tables
    ("pxl_col_in_fullres", "pxl_row_in_fullres"),    # 10x Visium (Space Ranger)
    ("imagecol", "imagerow"),                        # Seurat / Loupe exports
    ("x", "y"), ("X", "Y"),
    ("array_col", "array_row"),                      # Visium array grid (fallback)
)
SYMBOL_COLUMNS = ("gene_symbols", "gene_symbol", "gene_name", "gene_names", "feature_name", "feature_names",
                  "symbol", "Symbol", "SYMBOL", "name")
_ENSEMBL = re.compile(r"^ENS[A-Z]*G\d{6,}(\.\d+)?$")


def _as_array(x):
    return np.asarray(x.toarray() if sp.issparse(x) else x)


def find_coordinates(adata, spatial_key=None):
    """Return (coords, source): an (n_obs, 2) float array and a description of where it came from.

    ``spatial_key`` may name an ``obsm`` entry or a pair ``"xcol,ycol"`` of ``obs`` columns."""
    if spatial_key:
        if spatial_key in adata.obsm:
            xy = np.asarray(adata.obsm[spatial_key], float)
            return xy[:, :2], f"obsm['{spatial_key}']"
        cols = [c.strip() for c in str(spatial_key).split(",")]
        if len(cols) == 2 and all(c in adata.obs for c in cols):
            return adata.obs[cols].to_numpy(float), f"obs['{cols[0]}', '{cols[1]}']"
        raise ValueError(f"spatial_key {spatial_key!r} is neither an obsm key nor two obs columns")
    for k in COORD_OBSM_KEYS:
        if k in adata.obsm:
            xy = np.asarray(adata.obsm[k], float)
            if xy.ndim == 2 and xy.shape[1] >= 2:
                return xy[:, :2], f"obsm['{k}']"
    for xa, ya in COORD_OBS_PAIRS:
        if xa in adata.obs and ya in adata.obs:
            return adata.obs[[xa, ya]].to_numpy(float), f"obs['{xa}', '{ya}']"
    raise ValueError("no spatial coordinates found: put (x, y) in adata.obsm['spatial'] or pass spatial_key")


def _looks_ensembl(names, frac=0.5):
    names = [str(n) for n in names[:2000]]
    return bool(names) and np.mean([bool(_ENSEMBL.match(n)) for n in names]) >= frac


def gene_symbols(adata, symbol_column=None):
    """Return (symbols, source). Uses ``symbol_column`` if given; otherwise keeps ``var_names`` unless
    they look like Ensembl identifiers and a symbol column exists in ``var``."""
    if symbol_column:
        if symbol_column not in adata.var:
            raise ValueError(f"symbol_column {symbol_column!r} is not a column of adata.var")
        return adata.var[symbol_column].astype(str).to_numpy(), f"var['{symbol_column}']"
    if _looks_ensembl(list(adata.var_names)):
        for c in SYMBOL_COLUMNS:
            if c in adata.var and not _looks_ensembl(list(adata.var[c].astype(str))):
                return adata.var[c].astype(str).to_numpy(), f"var['{c}']"
        warnings.warn("var_names look like Ensembl identifiers and no gene-symbol column was found; "
                      "MetaboCast matches genes by symbol, so overlap with a model will be low")
    return np.asarray(adata.var_names, str), "var_names"


def looks_like_counts(X, n=20000):
    """True when the matrix holds non-negative integers (raw counts)."""
    v = X.data if sp.issparse(X) else np.asarray(X).ravel()
    v = np.asarray(v[:n], float)
    return bool(v.size) and bool(np.all(v >= 0)) and bool(np.allclose(v, np.round(v)))


def symbol_case(names):
    """'upper' (human-style symbols such as GFAP), 'title' (mouse-style, Gfap) or 'mixed'."""
    alpha = [g for g in map(str, names[:5000]) if g[:1].isalpha() and g.isalnum()]
    if not alpha:
        return "mixed"
    up = np.mean([g.isupper() for g in alpha])
    title = np.mean([g[:1].isupper() and g[1:].lower() == g[1:] and not g.isupper() for g in alpha])
    return "upper" if up > 0.6 else ("title" if title > 0.6 else "mixed")


def _cp10k_log1p(C):
    C = sp.csr_matrix(C, dtype=np.float32) if sp.issparse(C) else np.asarray(C, np.float32)
    tot = np.asarray(C.sum(1)).ravel()
    scale = np.divide(1e4, tot, out=np.zeros_like(tot, dtype=np.float64), where=tot > 0).astype(np.float32)
    if sp.issparse(C):
        C = sp.diags(scale) @ C
        C.data = np.log1p(C.data)
        return C.tocsr().astype(np.float32)
    return np.log1p(C * scale[:, None]).astype(np.float32)


def bin_spots(counts, coords, bin_size, groups=None):
    """Sum ``counts`` (observations x genes) over square bins of side ``bin_size`` (coordinate units).

    Bins never cross ``groups`` (e.g. tissue sections). Returns (binned, centres, size, first), where
    ``centres`` are member centroids, ``size`` the number of observations per bin and ``first`` the index
    of each bin's first member."""
    if not bin_size or bin_size <= 0:
        raise ValueError("bin_size must be a positive number in the units of the coordinates")
    coords = np.asarray(coords, float)
    g = np.zeros(len(coords), np.int64) if groups is None else pd.factorize(np.asarray(groups))[0].astype(np.int64)
    cell = np.floor((coords - coords.min(0)) / float(bin_size)).astype(np.int64)
    codes, uniq = pd.factorize(pd.MultiIndex.from_arrays([g, cell[:, 0], cell[:, 1]]))
    n_bins, n = len(uniq), len(codes)
    agg = sp.csr_matrix((np.ones(n, np.float32), (codes, np.arange(n))), shape=(n_bins, n))
    C = sp.csr_matrix(counts, dtype=np.float32) if sp.issparse(counts) else np.asarray(counts, np.float32)
    binned = agg @ C
    size = np.bincount(codes, minlength=n_bins)
    centres = np.asarray(agg @ coords) / size[:, None]
    first = np.full(n_bins, -1, np.int64)
    first[codes[::-1]] = np.arange(n)[::-1]
    return (sp.csr_matrix(binned) if sp.issparse(binned) else np.asarray(binned)), centres, size, first


def prepare_query(adata, symbol_column=None, spatial_key=None, normalize="auto", counts_layer=None,
                  bin_size=None, section_key=None, uppercase_genes=False, verbose=True):
    """Return a new AnnData ready for :meth:`MetaboCast.predict_metabolome`.

    Parameters
    ----------
    adata : AnnData
        Spots, cells or bins from any spatial transcriptomics platform.
    symbol_column : str, optional
        ``var`` column holding gene symbols (detected automatically when ``var_names`` are Ensembl IDs).
    spatial_key : str, optional
        ``obsm`` key, or ``"xcol,ycol"`` naming two ``obs`` columns, holding the coordinates.
    normalize : {"auto", "raw", "none"}
        ``"auto"`` (default) keeps a supplied ``layers["log1p"]``, otherwise applies CP10k + log1p when the
        data look like raw counts; ``"raw"`` always normalises; ``"none"`` treats ``X`` as log1p(CP10k).
    counts_layer : str, optional
        Layer with raw counts (for example ``"counts"``); used for normalisation and binning.
    bin_size : float, optional
        Aggregate observations into square pseudo-spots of this side length (coordinate units) before
        normalising. Recommended for single-cell and subcellular platforms (Xenium, CosMx, MERSCOPE,
        Visium HD 2 um bins, Stereo-seq), so that profiles approach the depth of the spot-level training data.
    section_key : str, optional
        ``obs`` column separating tissue sections; bins never cross sections and the column is kept.
    uppercase_genes : bool
        Upper-case gene symbols (approximate symbol-level ortholog matching, e.g. mouse data against a
        human-trained model). Retraining on paired data from the same species is preferable.
    """
    import anndata as ad

    log = print if verbose else (lambda *a, **k: None)
    info = {}
    names, info["gene_symbols"] = gene_symbols(adata, symbol_column)
    if uppercase_genes:
        names = np.char.upper(names.astype(str)); info["gene_symbols"] += " (upper-cased)"
    coords, info["coordinates"] = find_coordinates(adata, spatial_key)
    if not np.isfinite(coords).all():
        raise ValueError("coordinates contain missing or infinite values")

    if counts_layer:
        if counts_layer not in adata.layers:
            raise ValueError(f"counts_layer {counts_layer!r} not found in adata.layers")
        M, is_counts, src = adata.layers[counts_layer], True, f"layers['{counts_layer}']"
    elif normalize == "auto" and "log1p" in adata.layers and not bin_size:
        M, is_counts, src = adata.layers["log1p"], False, "layers['log1p']"
    elif normalize == "none":
        M, is_counts, src = (adata.layers["log1p"] if "log1p" in adata.layers else adata.X), False, "log1p input"
    else:
        M = adata.X
        is_counts = normalize == "raw" or looks_like_counts(M)
        src = "X"
        if not is_counts and "log1p" in adata.layers:
            M, src = adata.layers["log1p"], "layers['log1p']"
    M = M.tocsr() if sp.issparse(M) else np.asarray(M)

    if section_key and section_key not in adata.obs:
        raise ValueError(f"section_key {section_key!r} is not a column of adata.obs")
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    obsm = {k: v for k, v in adata.obsm.items() if k != "spatial"}

    if bin_size:
        if is_counts:
            counts = M
        elif sp.issparse(M):
            counts = M.copy(); counts.data = np.expm1(counts.data)
        else:
            counts = np.expm1(M)
        groups = obs[section_key].to_numpy() if section_key else None
        C, coords, size, first = bin_spots(counts, coords, bin_size, groups)
        n_in = len(obs)
        new_obs = pd.DataFrame(index=[f"bin{i}" for i in range(C.shape[0])])
        new_obs["n_observations"] = size
        if section_key:
            new_obs[section_key] = obs[section_key].to_numpy()[first]
        obs, obsm = new_obs, {}
        X = _cp10k_log1p(C)
        info.update(bin_size=float(bin_size), n_input_observations=int(n_in), n_bins=int(C.shape[0]),
                    normalisation=f"summed {'counts' if is_counts else 'expm1(log1p) values'} per bin, CP10k + log1p")
        log(f"  binned {n_in} observations into {C.shape[0]} pseudo-spots of side {bin_size:g} "
            f"(median {int(np.median(size))} per pseudo-spot)")
    elif is_counts:
        X = _cp10k_log1p(M)
        info["normalisation"] = f"CP10k + log1p applied to {src}"
    else:
        X = M.astype(np.float32) if sp.issparse(M) else np.asarray(M, np.float32)
        info["normalisation"] = f"none ({src} used as log1p(CP10k))"

    var = pd.DataFrame(index=pd.Index(names, dtype=str))
    q = ad.AnnData(X=X, obs=obs, var=var)
    for k, v in obsm.items():
        q.obsm[k] = v
    if not q.var_names.is_unique:
        q.var_names_make_unique()
        info["duplicate_symbols"] = "made unique with numeric suffixes"
    q.layers["log1p"] = q.X
    q.obsm["spatial"] = np.asarray(coords, float)
    info["symbol_case"] = symbol_case(list(q.var_names))
    q.uns["metabocast_input"] = info
    log(f"  input: {q.n_obs} observations x {q.n_vars} genes | coordinates from {info['coordinates']} | "
        f"symbols from {info['gene_symbols']} | {info['normalisation']}")
    return q


def gene_panel_overlap(model, adata):
    """Fractions of the model's training genes and of its highly variable genes present in ``adata``."""
    have = set(map(str, adata.var_names))
    genes = list(getattr(model, "genes_", []))
    hvg = list(getattr(model, "hvg_genes_", None) or [])
    out = {"training_genes": float(np.mean([g in have for g in genes])) if genes else float("nan"),
           "hvg": float(np.mean([g in have for g in hvg])) if hvg else float("nan")}
    out["model_symbol_case"] = symbol_case(genes) if genes else "mixed"
    out["query_symbol_case"] = symbol_case(list(adata.var_names))
    return out
