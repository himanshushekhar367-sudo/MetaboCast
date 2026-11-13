"""Query preparation for spatial transcriptomics data from different platforms (metabocast.io)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anndata as ad
from metabocast import MetaboCast, prepare_query, gene_panel_overlap, find_coordinates
from metabocast.io import bin_spots, looks_like_counts, symbol_case
from test_regression_fixes import paired


def counts_section(n=200, genes=None, seed=0, obs_xy=("x_centroid", "y_centroid")):
    r = np.random.RandomState(seed)
    genes = genes or [f"G{i}" for i in range(30)]
    a = ad.AnnData(X=sp.csr_matrix(r.poisson(2.0, (n, len(genes))).astype(np.float32)))
    a.var_names = genes
    a.obs_names = [f"c{i}" for i in range(n)]
    xy = r.rand(n, 2) * 200
    a.obs[obs_xy[0]] = xy[:, 0]
    a.obs[obs_xy[1]] = xy[:, 1]
    return a


@pytest.mark.parametrize("cols", [("x_centroid", "y_centroid"), ("CenterX_global_px", "CenterY_global_px"),
                                  ("center_x", "center_y"), ("x", "y")])
def test_coordinates_are_found_for_common_platform_columns(cols):
    a = counts_section(obs_xy=cols)
    xy, src = find_coordinates(a)
    assert xy.shape == (a.n_obs, 2) and cols[0] in src
    np.testing.assert_allclose(xy[:, 0], a.obs[cols[0]].to_numpy())


def test_raw_counts_are_normalised_to_log1p_cp10k():
    a = counts_section()
    assert looks_like_counts(a.X)
    q = prepare_query(a, verbose=False)
    X = q.layers["log1p"].toarray()
    expect = np.log1p(a.X.toarray() / a.X.toarray().sum(1, keepdims=True) * 1e4)
    np.testing.assert_allclose(X, expect, rtol=1e-5, atol=1e-5)
    assert q.obsm["spatial"].shape == (a.n_obs, 2)
    assert "CP10k" in q.uns["metabocast_input"]["normalisation"]


def test_ensembl_ids_are_replaced_by_a_symbol_column():
    a = counts_section(genes=[f"ENSG{i:011d}" for i in range(30)])
    a.var["gene_symbols"] = [f"G{i}" for i in range(30)]
    q = prepare_query(a, verbose=False)
    assert list(q.var_names[:3]) == ["G0", "G1", "G2"]
    assert q.uns["metabocast_input"]["gene_symbols"] == "var['gene_symbols']"


def test_binning_preserves_counts_and_respects_sections():
    a = counts_section(n=400)
    a.obs["section"] = np.where(np.arange(a.n_obs) < 200, "s1", "s2")
    q = prepare_query(a, bin_size=50, section_key="section", verbose=False)
    assert q.n_obs < a.n_obs
    assert q.obs["n_observations"].sum() == a.n_obs
    assert set(q.obs["section"]) == {"s1", "s2"}
    C, centres, size, first = bin_spots(a.X, find_coordinates(a)[0], 50, a.obs["section"].to_numpy())
    np.testing.assert_allclose(np.asarray(C.sum()), a.X.sum())
    assert (size > 0).all() and (first >= 0).all()


def test_symbol_case_and_uppercasing():
    assert symbol_case(["GFAP", "TH", "SLC6A3", "ACTB"]) == "upper"
    assert symbol_case(["Gfap", "Th", "Slc6a3", "Actb"]) == "title"
    a = counts_section(genes=["Gfap", "Th", "Slc6a3", "Actb"])
    assert list(prepare_query(a, uppercase_genes=True, verbose=False).var_names) == ["GFAP", "TH", "SLC6A3", "ACTB"]


def test_prepared_query_predicts_and_reports_panel_overlap():
    train = [paired(seed=s) for s in range(2)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(train)
    q0 = paired(seed=7)
    raw = ad.AnnData(X=sp.csr_matrix(np.round(np.expm1(np.asarray(q0.layers["log1p"])) * 10)))
    raw.var_names = list(q0.var_names)
    raw.obs["x"] = q0.obsm["spatial"][:, 0]
    raw.obs["y"] = q0.obsm["spatial"][:, 1]
    q = prepare_query(raw, verbose=False)
    pred = m.predict_metabolome(q)
    assert pred.shape == (q.n_obs, len(m.mz_)) and np.isfinite(pred).all()
    ov = gene_panel_overlap(m, q)
    assert ov["training_genes"] == 1.0 and ov["hvg"] == 1.0
    assert m.last_hvg_overlap_ == 1.0
    half = q[:, list(q.var_names[: q.n_vars // 2])].copy()
    m.predict_metabolome(half)
    assert 0.0 < m.last_hvg_overlap_ < 1.0


def test_missing_coordinates_fail_clearly():
    a = counts_section()
    a.obs = pd.DataFrame(index=a.obs_names)
    with pytest.raises(ValueError, match="no spatial coordinates"):
        prepare_query(a, verbose=False)
