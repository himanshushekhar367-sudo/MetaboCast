import numpy as np
import pytest
from metabocast import MetaboCast, MorphologyMetabolitePredictor, common_mz_axis
from metabocast.validation import bh_adjust, conformal_quantile, single_gene_predict, subject_splits
from test_regression_fixes import paired


def test_bh_matches_reference():
    from scipy.stats import false_discovery_control
    p = np.random.default_rng(2).uniform(size=30)
    np.testing.assert_allclose(bh_adjust(p), false_discovery_control(p))
    np.testing.assert_allclose(bh_adjust([.01, .02, .03]), [.03, .03, .03])
    np.testing.assert_allclose(bh_adjust([np.nan, .02, .01]), [np.nan, .02, .02])


@pytest.mark.parametrize("p", [[-.1], [1.1], [np.inf]])
def test_bh_rejects_invalid(p):
    with pytest.raises(ValueError):
        bh_adjust(p)


def test_negative_single_gene_signal_retains_sign_and_scale():
    x = np.arange(20.)[:, None]
    y = np.column_stack([100 - 3 * x[:, 0], 5 + 2 * x[:, 0]])
    np.testing.assert_allclose(single_gene_predict(x[:15], y[:15], x[15:]), y[15:])


def test_constant_baseline_predicts_training_mean():
    x = np.ones((12, 2))
    y = np.arange(24).reshape(12, 2)
    np.testing.assert_allclose(single_gene_predict(x, y, x[:2]), np.tile(y.mean(0), (2, 1)))


def test_finite_sample_quantile_and_insufficient_units():
    s = np.arange(1, 10.)[:, None]
    assert conformal_quantile(s, .9)[0] == 9
    assert np.isinf(conformal_quantile(s[:1], .9)).all()
    with pytest.raises(ValueError):
        conformal_quantile(s, 1)


@pytest.mark.parametrize("axis", [[100, np.nan], [100, 100], [100, np.inf], [-1, 100], []])
def test_invalid_mz_rejected(axis):
    a = paired(nmz=len(axis), mz=axis)
    with pytest.raises(ValueError):
        common_mz_axis([a])


def test_refit_clears_uncertainty_and_resets_extra_statistics():
    sections = [paired(seed=s) for s in range(3)]
    for a in sections:
        a.obsm['extra'] = np.column_stack([np.arange(a.n_obs), np.ones(a.n_obs)])
    m = MetaboCast(n_hvg=20, n_pcs=5, extra_key='extra').fit(sections)
    m.estimate_conformal(sections)
    m.fit(sections)
    fresh = MetaboCast(n_hvg=20, n_pcs=5, extra_key='extra').fit(sections)
    assert m.conf_width_ is None and m.train_predictability_ is None and m.transfer_predictability_ is None
    np.testing.assert_allclose(m.extra_train_mean_, fresh.extra_train_mean_)
    np.testing.assert_allclose(m.extra_train_sd_, fresh.extra_train_sd_)


def test_calibration_keeps_predictor_and_marks_unit():
    train, held = paired(seed=1), paired(seed=10)
    m = MetaboCast(n_hvg=20, n_pcs=5).fit([train])
    before = m.predict_metabolome(held.copy())
    coefficients = m.reg_.coef_.copy()
    m.calibrate([held], unit='spot')
    np.testing.assert_array_equal(m.reg_.coef_, coefficients)
    np.testing.assert_array_equal(m.predict_metabolome(held.copy()), before)
    assert m.uncertainty_['method'] == 'heldout_fixed_model'
    assert m.uncertainty_['n_units'] == held.n_obs
    m.calibrate([held], unit='section')
    assert np.isinf(m.conf_width_).all()


def test_prediction_clears_stale_calibration_metadata():
    a = paired()
    m = MetaboCast(n_hvg=20, n_pcs=5).fit([a])
    a.uns['metabocast_conf_width'] = np.ones(12)
    a.uns['metabocast_reliability'] = 'stale'
    m.predict_metabolome(a)
    assert 'metabocast_conf_width' not in a.uns
    assert 'metabocast_reliability' not in a.uns


def test_missing_extra_and_section_labels_fail_clearly():
    a = paired()
    with pytest.raises(ValueError, match='extra modality'):
        MetaboCast(extra_key='missing').fit([a])
    m = MetaboCast(n_hvg=20, n_pcs=5).fit([a])
    with pytest.raises(ValueError, match='section_key'):
        m.predict_metabolome(a, section_key='missing')


def test_morphology_axis_guard_and_batch_invariance():
    a, b = paired(seed=1), paired(seed=2)
    rng = np.random.default_rng(4)
    for x in (a, b):
        x.obsm['histology'] = rng.normal(size=(x.n_obs, 4))
    m = MorphologyMetabolitePredictor().fit([a])
    whole = m.predict_from_histology(b)
    one = m.predict_from_histology(b[:1].copy())
    np.testing.assert_allclose(one, whole[:1], atol=1e-6)
    b.uns['mz_features'] += 1
    with pytest.raises(ValueError, match='m/z axis mismatch'):
        m.fit([a, b])


def test_ambiguous_annotation_retains_alternatives():
    names, classes = MetaboCast._annotate_mz([135.031])
    assert 'Hypoxanthine' in names[0] and 'Lactate' in names[0]
    assert names[0].startswith('Hypoxanthine')      # nearest candidate first (-0.2 against +1.1 mDa)
    assert classes[0] == 'ambiguous'


def test_isomers_sharing_a_formula_carry_one_group_name():
    names, classes = MetaboCast._annotate_mz([179.055, 215.032])
    assert list(names) == ['Hexose', 'Hexose']        # glucose, fructose, galactose and inositol share C6H12O6
    assert list(classes) == ['sugar', 'sugar']


def test_package_annotation_uses_the_manuscript_reference():
    from metabocast.annotation import annotate, candidate_labels
    names, classes = MetaboCast._annotate_mz([187.0064, 302.0651, 100.0])
    assert list(names) == ['Xanthine', 'Adenosine', '']
    assert list(classes) == ['nucleobase', 'nucleoside', '']
    row = annotate([135.031]).iloc[0]
    assert row.n_candidates == 2 and row.adduct == '[M-H]-' and row.error_mDa < 0   # observed - theoretical
    assert candidate_labels([167.0207]) == ['Urate [M-H]-; Asparagine [M+Cl]-']


def test_subject_splits_never_separate_replicates():
    subjects = np.array(['a', 'a', 'b', 'c'])
    splits = list(subject_splits(subjects))
    assert len(splits) == 3
    for tr, te in splits:
        assert not set(subjects[tr]) & set(subjects[te])
