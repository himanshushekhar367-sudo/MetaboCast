"""Tests for the training objective: rank-Gaussian targets per training section, ridge penalty chosen
by inner cross-validation, the ridge + ExtraTrees ensemble (stored as a compact forest) and the monotone
back-transform to log1p intensity."""
import os
import sys

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anndata as ad
from metabocast import MetaboCast, ALPHA_GRID, select_alpha
from metabocast.metabocast import _CompactForest


def paired(nspots=120, ngenes=60, nmz=8, seed=0, scale=1.0):
    """Section whose metabolites depend on a few genes, so there is real signal to learn."""
    r = np.random.RandomState(seed)
    X = np.log1p(r.gamma(1.0, 1.0, (nspots, ngenes))).astype(np.float32)
    W = np.zeros((ngenes, nmz)); W[:4] = r.normal(size=(4, nmz))
    msi = np.exp(X @ W * 0.5 + 0.2 * r.normal(size=(nspots, nmz))) * scale
    a = ad.AnnData(X=X)
    a.var_names = [f"G{i}" for i in range(ngenes)]
    a.layers["log1p"] = a.X
    a.obsm["spatial"] = r.rand(nspots, 2) * 50
    a.uns["msi"] = msi.astype(np.float32)
    a.uns["mz_features"] = np.linspace(100.0, 900.0, nmz)
    return a


def test_defaults_are_rank_targets_auto_alpha_and_ensemble():
    m = MetaboCast()
    assert m.target_transform == "rank" and m.alpha == "auto" and m.estimator == "ensemble"
    with pytest.raises(ValueError):
        MetaboCast(alpha="best")
    with pytest.raises(ValueError):
        MetaboCast(target_transform="log")


def test_auto_alpha_uses_sections_blocks_or_falls_back():
    secs = [paired(seed=s) for s in range(3)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(secs)
    assert m.alpha_selection_ == "leave-one-section-out"
    assert m.alpha_ in ALPHA_GRID and m.reg_.alpha == m.alpha_
    assert m.alpha_cv_scores_.shape == (len(ALPHA_GRID), 3)
    one = MetaboCast(n_hvg=40, n_pcs=8).fit([paired(nspots=150, seed=4)])
    assert one.alpha_selection_ == "leave-one-spatial-block-out (pooled)"
    assert one.alpha_ in ALPHA_GRID and one.alpha_cv_scores_.shape == (len(ALPHA_GRID), 1)
    tiny = MetaboCast(n_hvg=40, n_pcs=8).fit([paired(nspots=30, seed=5)])
    assert tiny.alpha_selection_.startswith("fallback") and tiny.alpha_ == 200.0


def test_fixed_penalty_standardised_targets_match_plain_ridge():
    secs = [paired(seed=s) for s in range(2)]
    q = paired(seed=9)
    m = MetaboCast(n_hvg=40, n_pcs=8, alpha=200.0, target_transform=None, estimator="ridge").fit(secs)
    assert m.alpha_selection_ == "fixed" and m.alpha_ == 200.0 and m.trees_ is None
    X = lambda a: np.asarray(a[:, m.genes_].layers["log1p"])       # columns in the model's gene order
    F = np.vstack([m._features(X(a), m.genes_, np.asarray(a.obsm["spatial"]), None) for a in secs])
    Y = np.vstack([np.log1p(np.asarray(a.uns["msi"], float)) for a in secs])
    mu, sd = Y.mean(0), Y.std(0) + 1e-8
    ref = Ridge(alpha=200.0).fit(F, (Y - mu) / sd)
    Fq = m._features(X(q), m.genes_, np.asarray(q.obsm["spatial"]), None)
    expected = np.maximum(ref.predict(Fq) * sd + mu, 0)
    np.testing.assert_allclose(m.predict_metabolome(q.copy()), expected, rtol=1e-4, atol=1e-4)
    assert "metabocast_pred_rank" not in q.obsm


def test_back_transform_is_monotone_nonnegative_and_in_range():
    secs = [paired(seed=s) for s in range(3)]
    q = paired(seed=11)
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(secs)
    pred = m.predict_metabolome(q)
    z = q.obsm["metabocast_pred_rank"]
    for j in range(pred.shape[1]):          # monotone per ion (ties only from float32 rounding)
        o = np.argsort(z[:, j], kind="stable")
        assert (np.diff(pred[o, j]) >= -1e-6).all()
        assert spearmanr(pred[:, j], z[:, j])[0] > 0.9999
    assert (pred >= 0).all()
    hi = np.vstack([np.log1p(np.asarray(a.uns["msi"], float)) for a in secs]).max(0)
    assert (pred <= hi + 1e-3).all()


def test_rank_objective_is_invariant_to_section_intensity_scaling():
    """A monotone intensity shift in one training section (a batch effect) must not change the
    rank-harmonised model, whereas a model fitted to standardised log intensities is affected by it."""
    base = [paired(seed=s) for s in range(3)]
    shifted = [paired(seed=s, scale=(25.0 if s == 1 else 1.0)) for s in range(3)]
    q = paired(seed=12)
    a = MetaboCast(n_hvg=40, n_pcs=8, alpha=1000.0).fit(base)
    b = MetaboCast(n_hvg=40, n_pcs=8, alpha=1000.0).fit(shifted)
    a.predict_metabolome(qa := q.copy()); b.predict_metabolome(qb := q.copy())
    np.testing.assert_allclose(qa.obsm["metabocast_pred_rank"], qb.obsm["metabocast_pred_rank"], rtol=1e-5, atol=1e-5)
    la = MetaboCast(n_hvg=40, n_pcs=8, alpha=1000.0, target_transform=None, estimator="ridge").fit(base).predict_metabolome(q.copy())
    lb = MetaboCast(n_hvg=40, n_pcs=8, alpha=1000.0, target_transform=None, estimator="ridge").fit(shifted).predict_metabolome(q.copy())
    assert not np.allclose(la, lb)


def test_select_alpha_prefers_less_shrinkage_when_signal_is_strong():
    r = np.random.RandomState(0)
    W = r.normal(size=(5, 3))
    units = []
    for s in range(4):
        F = r.normal(size=(80, 5)); T = F @ W + 0.01 * r.normal(size=(80, 3))
        units.append((F, T, np.ones(3, bool)))
    alpha, scores = select_alpha(units)
    assert alpha <= 100 and np.isfinite(scores).all()
    alpha_p, scores_p = select_alpha(units, pooled=True)
    assert alpha_p <= 100 and scores_p.shape == (len(ALPHA_GRID), 1) and np.isfinite(scores_p).all()


def test_pooled_selection_rewards_getting_block_differences_right():
    """Blocks whose mean metabolite level differs, with the difference carried by the features: scoring
    each block on its own cannot see that a heavily shrunk model misses the between-block differences,
    pooling the held-out blocks can."""
    r = np.random.RandomState(3)
    units = []
    for b in range(5):
        F = r.normal(size=(60, 6)); F[:, 0] += 3.0 * b                 # block identity carried by feature 0
        T = (0.8 * F[:, :1] + 0.5 * r.normal(size=(60, 4)))             # level follows feature 0
        units.append((F, T, np.ones(4, bool)))
    grid = np.array([1.0, 1e3, 1e5, 1e7])
    _, per_block = select_alpha(units, grid)
    alpha_p, pooled = select_alpha(units, grid, pooled=True)
    assert alpha_p == 1.0
    assert pooled[0, 0] > pooled[-1, 0] + 0.2       # the collapsed (huge-penalty) model loses the block levels


def test_estimate_conformal_uses_the_same_objective():
    secs = [paired(seed=s) for s in range(3)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(secs)
    m.estimate_conformal(secs)
    assert m.conf_width_.shape == (len(m.mz_),) and np.isfinite(m.conf_width_).all()
    assert m.transfer_predictability_ is not None


def test_ensemble_estimator_trains_predicts_and_roundtrips(tmp_path):
    secs = [paired(seed=s) for s in range(3)]
    q = paired(seed=13)
    m = MetaboCast(n_hvg=40, n_pcs=8, estimator="ensemble", n_trees=10).fit(secs)
    assert m.trees_ is not None
    p1 = m.predict_metabolome(q.copy())
    assert p1.shape == (q.n_obs, 8) and (p1 >= 0).all() and np.isfinite(p1).all()
    path = tmp_path / "ens.pkl"; m.save(str(path))
    p2 = MetaboCast.load(str(path)).predict_metabolome(q.copy())
    np.testing.assert_allclose(p1, p2)
    with pytest.raises(ValueError):
        MetaboCast(estimator="forest")


def test_section_gene_scaling_is_invariant_to_per_section_gene_shifts():
    secs = [paired(seed=s) for s in range(3)]
    q = paired(seed=14)
    q2 = q.copy(); q2.layers["log1p"] = q2.layers["log1p"] * 1.7 + 0.3; q2.X = q2.layers["log1p"]
    m = MetaboCast(n_hvg=40, n_pcs=8, gene_scaling="section").fit(secs)
    m.predict_metabolome(q); m.predict_metabolome(q2)
    np.testing.assert_allclose(q.obsm["metabocast_pred_rank"], q2.obsm["metabocast_pred_rank"], rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError):
        MetaboCast(gene_scaling="global")


def test_compact_forest_matches_sklearn_including_threshold_ties():
    from sklearn.ensemble import ExtraTreesRegressor
    r = np.random.RandomState(0)
    F = r.normal(size=(600, 12)).astype(np.float32)
    T = (F[:, :3] @ r.normal(size=(3, 4)) + 0.3 * r.normal(size=(600, 4))).astype(np.float32)
    et = ExtraTreesRegressor(n_estimators=8, min_samples_leaf=5, max_features=0.33, random_state=0).fit(F, T)
    cf = _CompactForest(et)
    Xq = r.normal(size=(300, 12)).astype(np.float32)
    t = et.estimators_[0].tree_
    for k, i in enumerate(np.where(t.children_left != -1)[0][:40]):   # inputs sitting exactly on split thresholds
        Xq[k, t.feature[i]] = np.float32(t.threshold[i])
    np.testing.assert_allclose(cf.predict(Xq), et.predict(Xq), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        cf.predict(Xq[:, :5])


def test_ridge_estimator_is_the_ridge_half_of_the_ensemble():
    secs = [paired(seed=s) for s in range(3)]
    q = paired(seed=15)
    ens = MetaboCast(n_hvg=40, n_pcs=8, n_trees=10).fit(secs)
    rid = MetaboCast(n_hvg=40, n_pcs=8, estimator="ridge").fit(secs)
    assert rid.trees_ is None and ens.trees_ is not None and ens.alpha_ == rid.alpha_
    ens.predict_metabolome(qe := q.copy()); rid.predict_metabolome(qr := q.copy())
    F = rid._features(np.asarray(q[:, rid.genes_].layers["log1p"]), rid.genes_, np.asarray(q.obsm["spatial"]), None)
    np.testing.assert_allclose(qr.obsm["metabocast_pred_rank"], rid.reg_.predict(F), rtol=1e-4, atol=1e-5)
    e = ens.tpca_.inverse_transform(ens.trees_.predict(F).astype(np.float32)) * ens.t_sd_ + ens.t_mu_
    r = ens.reg_.predict(F)
    w_r, w_e, m_r, m_e = ens.ens_w_r_, ens.ens_w_e_, ens.ens_m_r_, ens.ens_m_e_
    np.testing.assert_allclose(qe.obsm["metabocast_pred_rank"], 0.5 * (m_r + m_e) + w_r * (r - m_r) + w_e * (e - m_e),
                               rtol=1e-4, atol=1e-4)


def test_equal_weights_balance_the_two_halves_and_fall_back_when_one_is_constant():
    from metabocast.metabocast import _equal_weights
    r = np.random.RandomState(1)
    R = r.normal(size=(400, 3)) * np.array([0.01, 1.0, 0.0]); E = r.normal(size=(400, 3)) * np.array([1.0, 1.0, 1.0]) + 0.2
    w_r, w_e, m_r, m_e = _equal_weights(R, E)
    np.testing.assert_allclose((w_r * R.std(0))[:2], (w_e * E.std(0))[:2], rtol=1e-5)   # equal spread
    assert w_r[2] == 0.0 and w_e[2] == 1.0                                                 # ridge constant


def test_memory_lean_prediction_matches_the_full_gene_path():
    """predict_metabolome projects large queries in row chunks from the HVG columns only; the result must equal
    the full-panel path, including genes missing from the query (imputed at the training mean)."""
    import scipy.sparse as sp
    secs = [paired(seed=s) for s in range(3)]
    m = MetaboCast(n_hvg=30, n_pcs=8, n_trees=10).fit(secs)
    q = paired(nspots=300, seed=21)
    q = q[:, [g for g in q.var_names if g not in ("G3", "G7")]].copy()          # two training genes absent
    q.layers["log1p"] = sp.csr_matrix(q.layers["log1p"])
    lean = m.predict_metabolome(q.copy())
    m._lean_predict = False
    full = m.predict_metabolome(q.copy())
    np.testing.assert_allclose(lean, full, rtol=1e-5, atol=1e-5)
    m._lean_predict = True
    P_chunks = m._query_pca_scores(q, chunk=37)                                   # chunking does not change the scores
    np.testing.assert_allclose(P_chunks, m._query_pca_scores(q, chunk=10_000), rtol=1e-5, atol=1e-5)


def test_load_reports_a_scikit_learn_version_mismatch_once(tmp_path, monkeypatch):
    """A model saved under another scikit-learn version loads with one explanatory warning instead of one
    InconsistentVersionWarning per pickled estimator, and predicts as before."""
    import warnings
    import sklearn.base
    secs = [paired(seed=s) for s in range(2)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(secs)
    q = paired(seed=7)
    ref = m.predict_metabolome(q.copy())
    monkeypatch.setattr(sklearn.base, "__version__", "0.0.0")       # pretend the model was saved elsewhere
    path = m.save(tmp_path / "model.pkl")
    monkeypatch.undo()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = MetaboCast.load(path)
    messages = [str(w.message) for w in caught]
    assert not any(type(w.message).__name__ == "InconsistentVersionWarning" for w in caught)
    assert sum("scikit-learn 0.0.0" in s for s in messages) == 1
    np.testing.assert_allclose(loaded.predict_metabolome(q.copy()), ref, rtol=1e-6, atol=1e-6)
