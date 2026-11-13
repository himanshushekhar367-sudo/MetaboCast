"""Regression tests for model persistence, uncertainty calibration and m/z-axis checks.

  * estimate_conformal() works on pickles that lack optional attributes (kegg_gmt_, use_kegg, extra_key);
  * load() does not carry over the saving session's per-query gene overlap;
  * reliability_table() does not invent trust tiers when no held-out predictability exists;
  * fit() and estimate_conformal() refuse sections whose m/z axes differ in value, not only in length.
  * fit() warns when the cross-validated ridge penalty is the largest value of the grid.

Run: pytest -q tests/test_regression_fixes.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anndata as ad
from metabocast import MetaboCast, common_mz_axis, group_sections_by_mz_axis

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(REPO, "metabocast_model.pkl")
needs_shipped_model = pytest.mark.skipif(not os.path.exists(PKL), reason="shipped model missing")


def paired(nspots=60, ngenes=80, nmz=12, seed=0, mz=None):
    """Minimal MetaboCast-format section: log1p genes + msi targets + m/z axis + coordinates."""
    r = np.random.RandomState(seed)
    a = ad.AnnData(X=np.log1p(r.gamma(1.0, 1.0, (nspots, ngenes)).astype(np.float32)))
    a.var_names = [f"G{i}" for i in range(ngenes)]
    a.layers["log1p"] = a.X
    a.obsm["spatial"] = r.rand(nspots, 2) * 50
    a.uns["msi"] = r.gamma(1.0, 1.0, (nspots, nmz)).astype(np.float32)
    a.uns["mz_features"] = np.linspace(100.0, 900.0, nmz) if mz is None else np.asarray(mz, float)
    return a


def query(genes, nspots=40, seed=7, frac=0.8):
    r = np.random.RandomState(seed)
    use = list(genes[: max(1, int(len(genes) * frac))])
    q = ad.AnnData(X=np.log1p(r.gamma(1.0, 1.0, (nspots, len(use))).astype(np.float32)))
    q.var_names = use
    q.layers["log1p"] = q.X
    q.obsm["spatial"] = r.rand(nspots, 2) * 50
    return q


# --------------------------------------------------------------------------- #
# estimate_conformal() must not require attributes newer than the pickle
# --------------------------------------------------------------------------- #
def test_estimate_conformal_on_pickle_without_kegg_gmt_attr():
    """A model pickled without kegg_gmt_/use_kegg/extra_key must still calibrate."""
    tr = [paired(seed=s) for s in range(3)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(tr)
    for stale in ("kegg_gmt_", "use_kegg", "extra_key", "kegg"):
        m.__dict__.pop(stale, None)
    assert not hasattr(m, "kegg_gmt_")

    out = m.estimate_conformal(tr)

    assert out is m
    assert m.conf_width_ is not None
    assert m.conf_width_.shape == (len(m.mz_),)
    assert np.isfinite(m.conf_width_).all() and (m.conf_width_ >= 0).all()
    assert m.transfer_predictability_ is not None          # held-out-section accuracy is transfer evidence


def test_estimate_conformal_reuses_parent_kegg_sets_without_reloading():
    """The calibration refit gets the parent's exact gene sets even when no .gmt path is recorded,
    and does not emit the "no KEGG gene sets could be loaded" warning while doing so."""
    import warnings

    tr = [paired(seed=s) for s in range(2)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(tr)
    m.use_kegg = True
    m.kegg = {"FAKE_PATHWAY": ["G1", "G2", "G3", "G4"]}
    m.__dict__.pop("kegg_gmt_", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.estimate_conformal(tr)

    assert not [w for w in caught if "KEGG" in str(w.message)]
    assert m.conf_width_ is not None and m.conf_width_.shape == (len(m.mz_),)


@needs_shipped_model
def test_shipped_model_recalibrates_on_its_own_axis():
    """Compatibility check against the shipped pickle (rank targets, ridge + ExtraTrees ensemble).

    Calibration sections are synthetic but carry the model's OWN 2,086-ion axis, which is what
    a legitimate recalibration would use. Pickles without kegg_gmt_ are covered by
    test_estimate_conformal_on_pickle_without_kegg_gmt_attr.
    """
    m = MetaboCast.load(PKL)
    assert m.estimator == "ensemble" and m.target_transform == "rank"
    genes = list(m.genes_[:60])
    cal = [paired(nspots=40, ngenes=len(genes), nmz=len(m.mz_), seed=s, mz=m.mz_) for s in (1, 2)]
    for a in cal:
        a.var_names = genes

    m.estimate_conformal(cal)

    assert m.conf_width_ is not None and m.conf_width_.shape == (len(m.mz_),)


# --------------------------------------------------------------------------- #
# estimate_conformal() must refuse a calibration cohort on another axis
# --------------------------------------------------------------------------- #
def test_estimate_conformal_rejects_calibration_on_a_different_mz_axis():
    """conf_width_ is indexed by self.mz_, so calibrating on another axis mislabels every ion."""
    axis = np.linspace(100.0, 900.0, 10)
    m = MetaboCast(n_hvg=40, n_pcs=8).fit([paired(nmz=10, seed=s, mz=axis) for s in range(2)])
    m.estimate_conformal([paired(nmz=10, seed=s, mz=axis) for s in range(2)])
    good_widths = m.conf_width_.copy()
    good_tp = m.transfer_predictability_.copy()

    other = [paired(nmz=10, seed=s, mz=axis + 0.5) for s in (5, 6)]   # same length, different ions
    with pytest.raises(ValueError, match="calibration m/z axis does not match"):
        m.estimate_conformal(other)

    # the guard must fail before overwriting anything
    assert np.array_equal(m.conf_width_, good_widths), "conf_width_ was clobbered by a bad call"
    assert np.array_equal(m.transfer_predictability_, good_tp, equal_nan=True)


def test_estimate_conformal_rejects_calibration_with_a_different_ion_count():
    axis = np.linspace(100.0, 900.0, 10)
    m = MetaboCast(n_hvg=40, n_pcs=8).fit([paired(nmz=10, seed=s, mz=axis) for s in range(2)])
    with pytest.raises(ValueError, match="calibration m/z axis does not match"):
        m.estimate_conformal([paired(nmz=6, seed=s) for s in (5, 6)])
    assert getattr(m, "conf_width_", None) is None


def test_estimate_conformal_rejects_internally_inconsistent_calibration_cohort():
    """A calibration cohort that disagrees with itself is caught by the same axis check."""
    axis = np.linspace(100.0, 900.0, 10)
    m = MetaboCast(n_hvg=40, n_pcs=8).fit([paired(nmz=10, seed=s, mz=axis) for s in range(2)])
    mixed = [paired(nmz=10, seed=5, mz=axis), paired(nmz=10, seed=6, mz=axis + 0.5)]
    with pytest.raises(ValueError, match="m/z axis mismatch"):
        m.estimate_conformal(mixed)


@needs_shipped_model
def test_shipped_model_rejects_calibration_on_an_unrelated_cohort():
    """The realistic form of the mistake: recalibrating the 2,086-ion model on a 12-ion cohort."""
    m = MetaboCast.load(PKL)
    shipped_widths = np.asarray(m.conf_width_, np.float32).copy()
    genes = list(m.genes_[:60])
    cal = [paired(nspots=40, ngenes=len(genes), nmz=12, seed=s) for s in (1, 2)]
    for a in cal:
        a.var_names = genes

    with pytest.raises(ValueError, match="calibration m/z axis does not match"):
        m.estimate_conformal(cal)

    assert np.array_equal(np.asarray(m.conf_width_, np.float32), shipped_widths), \
        "the shipped conformal widths must survive a rejected recalibration"


# --------------------------------------------------------------------------- #
# load() must not carry over the saving session's gene overlap
# --------------------------------------------------------------------------- #
def test_load_resets_stale_gene_overlap():
    import tempfile

    tr = [paired(seed=s) for s in range(2)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(tr)
    m.predict_metabolome(query(m.genes_, frac=0.5))
    saved_overlap = m.last_gene_overlap_
    assert np.isfinite(saved_overlap) and saved_overlap < 1.0, "precondition: a stale value is saved"

    path = os.path.join(tempfile.mkdtemp(), "m.pkl")
    m.save(path)
    reloaded = MetaboCast.load(path)
    assert np.isnan(reloaded.last_gene_overlap_), "load() must not report the saving session's overlap"

    raw = os.path.join(tempfile.mkdtemp(), "raw.pkl")
    m.save(raw, bundle=False)
    assert np.isnan(MetaboCast.load(raw).last_gene_overlap_), "non-bundle pickles must reset too"

    # and a real predict on the reloaded model repopulates it
    p = reloaded.predict_metabolome(query(reloaded.genes_, frac=0.5))
    assert 0.0 < reloaded.last_gene_overlap_ <= 1.0 and np.isfinite(p).all()


def test_stale_overlap_does_not_widen_conformal_interval_before_predict():
    """conf_width_adj is derived from last_gene_overlap_; a stale 0.5 would inflate it by 1.5x."""
    tr = [paired(seed=s) for s in range(3)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(tr)
    m.estimate_conformal(tr)
    m.predict_metabolome(query(m.genes_, frac=0.5))       # bakes a ~0.5 overlap into the pickle
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "m.pkl")
    m.save(path)

    reloaded = MetaboCast.load(path)
    q = query(reloaded.genes_, frac=1.0)                  # full-overlap query
    reloaded.predict_metabolome(q)
    base = np.asarray(q.uns["metabocast_conf_width"], float)
    adj = np.asarray(q.uns["metabocast_conf_width_adj"], float)
    assert np.allclose(adj, base), "full-overlap query must not inherit the saved query's widening"


@needs_shipped_model
def test_shipped_model_overlap_is_nan_until_predicted():
    m = MetaboCast.load(PKL)
    assert np.isnan(m.last_gene_overlap_), "the shipped bundle must not report a stored overlap"
    m.predict_metabolome(query(m.genes_))
    assert 0.0 <= m.last_gene_overlap_ <= 1.0 and np.isfinite(m.last_gene_overlap_)


# --------------------------------------------------------------------------- #
# no artificial terciles when train_predictability_ is absent
# --------------------------------------------------------------------------- #
def test_reliability_tier_is_uncalibrated_without_predictability():
    tr = [paired(nmz=9, seed=s) for s in range(2)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(tr)
    # conformal widths present and spread out, but no held-out predictability
    m.conf_width_ = np.linspace(0.1, 5.0, len(m.mz_)).astype(np.float32)
    m.__dict__.pop("train_predictability_", None)

    rel = m.reliability_table()

    assert set(rel["trust_tier"]) == {"uncalibrated"}, "widths alone must not define trust tiers"
    assert not {"transfer-supported", "within-section only", "low"} & set(rel["trust_tier"])
    # the widths themselves are still reported — only the invented ranking is gone
    assert np.isfinite(rel["conf_width"].to_numpy()).all()


def test_reliability_tier_uses_predictability_when_present():
    """Chemistry must not override weak measured held-out predictability."""
    mz = np.array([255.2329, 306.0765, 89.0244], dtype=float)   # FA16:0, GSH, lactate [M-H]-
    tr = [paired(nmz=3, seed=s, mz=mz) for s in range(2)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(tr)
    m.train_predictability_ = np.array([0.45, 0.05, 0.05], dtype=np.float32)

    rel = m.reliability_table().set_index("annotation")

    assert rel.loc["FA16:0", "trust_tier"] == "within-section only"   # r >= 0.30, transfer not measured
    assert rel.loc["Glutathione", "trust_tier"] == "low"           # no unsupported class rescue
    assert rel.loc["Lactate", "trust_tier"] == "low"       # polar ion stays low

    # transfer evidence decides: a strong within-section score with weak transfer is not transfer-supported
    m.transfer_predictability_ = np.array([0.05, 0.25, np.nan], dtype=np.float32)
    rel = m.reliability_table().set_index("annotation")
    assert rel.loc["FA16:0", "trust_tier"] == "within-section only"
    assert rel.loc["Glutathione", "trust_tier"] == "transfer-supported"
    assert rel.loc["Lactate", "trust_tier"] == "low"
    rel = m.reliability_table(transfer_min=0.30).set_index("annotation")
    assert rel.loc["Glutathione", "trust_tier"] == "low"


def test_reliability_tier_uncalibrated_per_ion_when_predictability_is_nan():
    """Ions whose held-out correlation could not be computed are uncalibrated, not width-ranked."""
    tr = [paired(nmz=4, seed=s) for s in range(2)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(tr)
    m.conf_width_ = np.linspace(0.1, 4.0, len(m.mz_)).astype(np.float32)
    m.train_predictability_ = np.array([0.5, np.nan, np.nan, 0.2], dtype=np.float32)

    tiers = list(m.reliability_table()["trust_tier"])

    assert tiers[0] == "within-section only" and tiers[3] == "low"
    assert tiers[1] == tiers[2] == "uncalibrated"


@needs_shipped_model
def test_shipped_model_reports_tiers_from_held_out_predictability():
    """The shipped bundle carries held-out per-ion predictability (within-section CV and leave-one-section-out),
    so tiers come from measured accuracy, with transfer evidence first; ions never detected stay 'uncalibrated'."""
    m = MetaboCast.load(PKL)
    assert getattr(m, "train_predictability_", None) is not None
    rel = m.reliability_table()
    assert len(rel) == len(m.mz_)
    assert set(rel["trust_tier"]) <= {"transfer-supported", "within-section only", "low", "uncalibrated"}
    assert {"transfer-supported", "within-section only", "low"} <= set(rel["trust_tier"])
    ts = rel["trust_tier"] == "transfer-supported"
    assert (rel.loc[ts, "transfer_predictability_spearman"] >= 0.20).all()
    assert np.isfinite(rel["transfer_predictability_spearman"].to_numpy()).mean() > 0.9
    assert np.isfinite(rel["conf_width"].to_numpy()).all(), "shipped widths reported for every ion"


# --------------------------------------------------------------------------- #
# m/z axes must match by value, not just by length
# --------------------------------------------------------------------------- #
def test_fit_rejects_same_length_different_valued_mz_axes():
    a = paired(nmz=10, seed=0)
    b = paired(nmz=10, seed=1, mz=np.linspace(100.0, 900.0, 10) + 0.5)   # same length, shifted grid
    with pytest.raises(ValueError, match="m/z axis mismatch"):
        MetaboCast(n_hvg=40, n_pcs=8).fit([a, b])


def test_fit_rejects_different_length_mz_axes():
    a = paired(nmz=10, seed=0)
    b = paired(nmz=8, seed=1)
    with pytest.raises(ValueError, match="m/z axis mismatch"):
        MetaboCast(n_hvg=40, n_pcs=8).fit([a, b])


def test_fit_accepts_identical_axes_across_dtypes_and_keeps_the_axis():
    """Backward compatibility: the same axis stored as float32 and float64 is one axis.

    A float32 m/z near 1000 Da is off by up to ~6e-5 Da, which must not be read as a
    different grid — real cohorts mix the two dtypes across .h5ad files.
    """
    mz = np.linspace(100.0, 1000.0, 40)
    a = paired(nmz=40, seed=0, mz=mz)
    b = paired(nmz=40, seed=1, mz=mz.astype(np.float32))
    assert not np.array_equal(np.asarray(b.uns["mz_features"], float), mz), "dtype noise expected"

    m = MetaboCast(n_hvg=40, n_pcs=8).fit([a, b])

    assert np.allclose(m.mz_, mz) and m.mz_.shape == (40,)
    assert group_sections_by_mz_axis([a, b]) == [[0, 1]], "dtype noise must not split a cohort"


def test_fit_mz_atol_is_configurable():
    mz = np.linspace(100.0, 900.0, 10)
    a = paired(nmz=10, seed=0, mz=mz)
    b = paired(nmz=10, seed=1, mz=mz + 0.02)          # 20 mDa: a different grid at the default 1 mDa
    with pytest.raises(ValueError, match="m/z axis mismatch"):
        MetaboCast(n_hvg=40, n_pcs=8).fit([a, b])
    m = MetaboCast(n_hvg=40, n_pcs=8).fit([a, b], mz_atol=0.05)
    assert np.allclose(m.mz_, mz)


def test_common_mz_axis_error_names_the_offending_section_and_channel():
    mz = np.linspace(100.0, 900.0, 6)
    bad = mz.copy()
    bad[4] += 0.02
    with pytest.raises(ValueError) as e:
        common_mz_axis([paired(nmz=6, mz=mz), paired(nmz=6, mz=mz), paired(nmz=6, mz=bad)])
    msg = str(e.value)
    assert "section 2" in msg and "index 4" in msg


def test_group_sections_by_mz_axis_splits_on_values_not_length():
    """Cohort selection (pipeline/train_and_evaluate.py): same ion count, different grid -> different groups."""
    axis_a = np.linspace(100.0, 900.0, 10)
    axis_b = axis_a + 0.5                       # same length, genuinely different ions
    adatas = [paired(nmz=10, seed=0, mz=axis_a),
              paired(nmz=10, seed=1, mz=axis_b),
              paired(nmz=10, seed=2, mz=axis_a),
              paired(nmz=10, seed=3, mz=axis_a)]

    groups = group_sections_by_mz_axis(adatas)

    assert groups == [[0, 2, 3], [1]], "equal ion counts alone must not pool sections"
    # the dominant group is poolable; the held-out one is not part of it
    assert np.allclose(common_mz_axis([adatas[i] for i in groups[0]]), axis_a)


def test_group_sections_by_mz_axis_orders_by_size_then_first_appearance():
    axis_a = np.linspace(100.0, 900.0, 5)
    axis_b = np.linspace(150.0, 950.0, 5)
    adatas = [paired(nmz=5, seed=0, mz=axis_b), paired(nmz=5, seed=1, mz=axis_a),
              paired(nmz=5, seed=2, mz=axis_b), paired(nmz=5, seed=3, mz=axis_a)]
    assert group_sections_by_mz_axis(adatas) == [[0, 2], [1, 3]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))


def test_calibrate_records_transfer_evidence_per_section():
    """calibrate() scores the fixed model on sections it never saw: that is transfer evidence, averaged over
    sections, and it must not overwrite the within-section score."""
    tr = [paired(seed=s) for s in range(2)]
    m = MetaboCast(n_hvg=40, n_pcs=8).fit(tr)
    m.train_predictability_ = np.full(len(m.mz_), 0.5, np.float32)
    cal = [paired(seed=s) for s in (7, 8)]
    m.calibrate(cal, unit="spot")
    assert np.array_equal(m.train_predictability_, np.full(len(m.mz_), 0.5, np.float32))
    from scipy.stats import spearmanr
    j = 0
    per = []
    for a in cal:
        true = np.log1p(np.asarray(a.uns["msi"], float))[:, j]
        pred = m.predict_metabolome(a.copy())[:, j]
        per.append(spearmanr(true, pred)[0])
    assert np.isclose(m.transfer_predictability_[j], np.mean(per), atol=1e-6)


# --------------------------------------------------------------------------- #
# a cross-validated penalty at the edge of the grid is reported
# --------------------------------------------------------------------------- #
def test_penalty_at_the_largest_grid_value_warns():
    """With a one-value grid the selection is necessarily at the edge, so fit() must warn."""
    tr = [paired(seed=s) for s in range(3)]
    with pytest.warns(RuntimeWarning, match="largest value of the grid"):
        m = MetaboCast(n_hvg=40, n_pcs=8, estimator="ridge", alpha_grid=[10.0]).fit(tr)
    assert m.alpha_ == 10.0


def test_penalty_inside_the_grid_does_not_warn(monkeypatch):
    """A selection below the largest grid value must not warn."""
    import warnings
    import metabocast.metabocast as mm
    real = mm.select_alpha

    def first_value(units, grid, pooled=False):
        _, scores = real(units, grid, pooled=pooled)
        return float(np.asarray(grid, float)[0]), scores
    monkeypatch.setattr(mm, "select_alpha", first_value)
    tr = [paired(seed=s) for s in range(3)]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = MetaboCast(n_hvg=40, n_pcs=8, estimator="ridge", alpha_grid=[1.0, 10.0, 100.0]).fit(tr)
    assert m.alpha_ == 1.0
    assert not any("largest value of the grid" in str(w.message) for w in rec)
