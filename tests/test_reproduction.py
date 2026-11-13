import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_regression_fixes import paired
from metabocast import MetaboCast
from reproduce.magpie_lung_validation import classify, coregister, build

ROOT = Path(__file__).resolve().parents[1]


def test_high_mass_is_not_automatically_a_lipid():
    assert classify([950.123])[0] == 'unannotated'


def test_registration_does_not_use_expression_when_orientation_supplied():
    coords = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    vis = SimpleNamespace(obsm={'spatial': coords})
    sm, score, null, orient = coregister(vis, coords, np.ones((4, 2)), orientation=0)
    assert sm.shape == (4, 2) and orient == 0
    assert np.isnan(score) and np.isnan(null)
    with pytest.raises(ValueError, match='independently established'):
        coregister(vis, coords, np.ones((4, 2)))


def test_missing_registration_manifest_fails_before_creating_outputs(tmp_path):
    cohort = tmp_path / 'cohort'
    (cohort / 'sample' / 'msi').mkdir(parents=True)
    output = tmp_path / 'built'
    with pytest.raises(ValueError, match='missing independent orientation'):
        build(str(cohort), str(output))
    assert not output.exists()


@pytest.mark.parametrize('script', ['reproduce/within_benchmark.py', 'reproduce/loso_benchmark.py',
    'reproduce/build_shipped_model.py', 'reproduce/per_ion_reliability.py', 'reproduce/celltype_ablation.py',
    'reproduce/crosscancer/summarise_crosscancer.py', 'reproduce/crosscancer/derive_figure_sources.py',
    'reproduce/magpie_lung_validation.py', 'reproduce/figures/make_figures.py',
    'reproduce/tables/build_supplementary_tables.py', 'pipeline/train_and_evaluate.py',
    'examples/train_on_paired_data.py', 'examples/predict_on_your_data.py'])
def test_cli_help_needs_no_data(script):
    result = subprocess.run([sys.executable, str(ROOT / script), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_training_cli_keeps_calibration_out_of_fit(tmp_path):
    train, cal = tmp_path / 'train', tmp_path / 'cal'
    train.mkdir()
    cal.mkdir()
    paired(nspots=12, ngenes=10).write_h5ad(train / 'train.h5ad')
    paired(nspots=12, ngenes=10, seed=5).write_h5ad(cal / 'held.h5ad')
    path = tmp_path / 'model.pkl'
    command = [sys.executable, str(ROOT / 'examples/train_on_paired_data.py'),
               '--data', str(train), '--calibration-data', str(cal), '--out', str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    model = MetaboCast.load(path)
    assert model.pca_.n_samples_ == 12
    assert model.uncertainty_['n_units'] == 1
    assert np.isinf(model.conf_width_).all()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and 'output model exists' in result.stderr


def test_query_preparation_preserves_supplied_log_layer():
    from metabocast import prepare_query
    a = paired(nspots=12, ngenes=10)
    layer = np.asarray(a.layers['log1p']).copy()
    a.X = np.full_like(a.X, 100)
    q = prepare_query(a, verbose=False)
    np.testing.assert_array_equal(np.asarray(q.layers['log1p']), layer)


def test_prediction_cli_exports_uncertainty_provenance(tmp_path):
    import pandas as pd
    model = MetaboCast(n_hvg=10, n_pcs=5).fit([paired(nspots=12, ngenes=10)])
    model_path = tmp_path / 'model.pkl'
    model.save(model_path)
    query = tmp_path / 'query.h5ad'
    paired(nspots=12, ngenes=10, seed=10).write_h5ad(query)
    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, str(ROOT / 'examples/predict_on_your_data.py'),
                             '--model', str(model_path), '--query', str(query), '--out', str(out)],
                             capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    table = pd.read_csv(out / 'predicted_metabolites.csv')
    assert set(table.trust_tier) == {'uncalibrated'}
    assert set(table.uncertainty_method) == {'none'}
