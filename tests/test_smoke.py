"""Smoke tests: the shipped model must load and predict via the advertised 3-line workflow,
and a from-scratch train->save->load cycle must round-trip. Run: pytest -q  (or python tests/test_smoke.py)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anndata as ad
from metabocast import MetaboCast, MorphologyMetabolitePredictor, add_histology_features, add_celltype_features

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(REPO, "metabocast_model.pkl")


def _synthetic_query(genes, n=200, seed=0):
    rng = np.random.RandomState(seed)
    use = list(genes[:max(1, int(len(genes) * 0.8))])          # 80% panel overlap
    X = rng.gamma(1.0, 1.0, size=(n, len(use))).astype(np.float32)
    q = ad.AnnData(X=np.log1p(X)); q.var_names = use
    q.layers["log1p"] = q.X
    q.obsm["spatial"] = rng.rand(n, 2) * 100
    return q


def test_shipped_model_loads_and_predicts():
    assert os.path.exists(PKL), "shipped model missing"
    model = MetaboCast.load(PKL)                              # the bundle loads as a MetaboCast, not a dict
    assert isinstance(model, MetaboCast)
    assert getattr(model, "gene_mean_", None) is not None, "gene_mean_ must be present (absent genes are mean-imputed)"
    q = _synthetic_query(model.genes_)
    pred = model.predict_metabolome(q)
    assert pred.shape == (q.n_obs, len(model.mz_))
    assert np.isfinite(pred).all()
    assert (pred >= 0).all(), "predictions must be non-negative log1p intensities"
    assert 0.0 <= model.last_gene_overlap_ <= 1.0 and model.last_gene_overlap_ == model.last_gene_overlap_, "gene-panel overlap must be set"
    # the shipped uncertainty widths are attached to the query
    assert "metabocast_conf_width" in q.uns
    assert np.asarray(q.uns["metabocast_conf_width"]).shape[0] == len(model.mz_)


def test_train_save_load_roundtrip(tmp_path=None):
    import tempfile
    def paired(nspots, ngenes=300, nmz=40, seed=0):
        r = np.random.RandomState(seed)
        a = ad.AnnData(X=np.log1p(r.gamma(1, 1, (nspots, ngenes)).astype(np.float32)))
        a.var_names = [f"G{i}" for i in range(ngenes)]; a.layers["log1p"] = a.X
        a.obsm["spatial"] = r.rand(nspots, 2) * 50
        a.uns["msi"] = r.gamma(1, 1, (nspots, nmz)).astype(np.float32)
        a.uns["mz_features"] = np.linspace(100, 900, nmz)
        return a
    tr = [paired(150, seed=s) for s in range(3)]
    m = MetaboCast(n_hvg=100, use_kegg=False).fit(tr)
    m.estimate_conformal(tr)
    q = paired(120, seed=99)
    p1 = m.predict_metabolome(q.copy())
    d = tempfile.mkdtemp(); path = os.path.join(d, "m.pkl")
    m.save(path)
    m2 = MetaboCast.load(path)
    assert isinstance(m2, MetaboCast)
    p2 = m2.predict_metabolome(q.copy())
    assert np.allclose(p1, p2), "save->load must round-trip numerically"
    assert (p1 >= 0).all()


def test_kegg_human_map_autoloads():
    """use_kegg=True must auto-load the shipped human KEGG map (no path) and train/predict."""
    import anndata as ad, numpy as np
    def paired(nspots, seed):
        rr = np.random.RandomState(seed)
        a = ad.AnnData(X=np.log1p(rr.gamma(1, 1, (nspots, 300)).astype(np.float32)))
        a.var_names = [f"G{i}" for i in range(300)]; a.layers["log1p"] = a.X
        a.obsm["spatial"] = rr.rand(nspots, 2) * 50
        a.uns["msi"] = rr.gamma(1, 1, (nspots, 30)).astype(np.float32)
        a.uns["mz_features"] = np.linspace(100, 900, 30)
        return a
    tr = [paired(120, s) for s in range(2)]
    m = MetaboCast(n_hvg=100, use_kegg=True).fit(tr)
    assert len(m.kegg) > 50, "shipped human KEGG map did not auto-load with use_kegg=True"
    assert m.kegg_gmt_ is not None and str(m.kegg_gmt_).endswith(".gmt")
    p = m.predict_metabolome(paired(80, 99))
    assert p.shape[1] == len(m.mz_) and (p >= 0).all()


def test_histology_features_from_embedded_visium_image():
    rng = np.random.RandomState(3)
    a = ad.AnnData(X=np.ones((4, 3), np.float32))
    a.var_names = ["G1", "G2", "G3"]
    a.obs["orig.ident"] = ["lib1"] * a.n_obs
    a.obsm["spatial"] = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=float)
    img = rng.rand(80, 80, 3).astype(np.float32)
    a.uns["spatial"] = {
        "lib1": {
            "images": {"hires": img},
            "scalefactors": {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": 8.0},
        }
    }
    add_histology_features(a)
    assert "histology" in a.obsm
    assert a.obsm["histology"].shape == (a.n_obs, 13)
    assert np.isfinite(a.obsm["histology"]).all()
    assert a.uns["histology_source"]["library_id"] == "lib1"


def test_morphology_predictor_train_predict_roundtrip():
    rng = np.random.RandomState(4)

    def paired(nspots, seed):
        rr = np.random.RandomState(seed)
        a = ad.AnnData(X=np.ones((nspots, 2), np.float32))
        a.var_names = ["G1", "G2"]
        a.obsm["histology"] = rr.normal(size=(nspots, 13)).astype(np.float32)
        a.uns["msi"] = rr.gamma(1.0, 1.0, size=(nspots, 9)).astype(np.float32)
        a.uns["mz_features"] = np.linspace(100, 300, 9)
        return a

    train = [paired(30, 10), paired(35, 11)]
    query = ad.AnnData(X=np.ones((12, 2), np.float32))
    query.var_names = ["G1", "G2"]
    query.obsm["histology"] = rng.normal(size=(12, 13)).astype(np.float32)
    m = MorphologyMetabolitePredictor().fit(train)
    pred = m.predict_from_histology(query)
    assert pred.shape == (query.n_obs, 9)
    assert "metabocast_he_pred" in query.obsm
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()


def test_celltype_reliability_focus_filter():
    def paired(nspots, seed):
        rr = np.random.RandomState(seed)
        markers = ["CD3D", "MS4A1", "LYZ", "EPCAM", "PECAM1", "COL1A1", "PTPRC"]
        genes = markers + [f"G{i}" for i in range(60)]
        a = ad.AnnData(X=np.log1p(rr.gamma(1, 1, (nspots, len(genes))).astype(np.float32)))
        a.var_names = genes
        a.layers["log1p"] = a.X
        a.obsm["spatial"] = rr.rand(nspots, 2) * 50
        a.uns["msi"] = rr.gamma(1.0, 1.0, size=(nspots, 3)).astype(np.float32)
        a.uns["mz_features"] = np.array([255.2329, 306.0765, 89.0244], dtype=float)  # FA16:0, GSH, lactate [M-H]-
        add_celltype_features(a)
        return a

    tr = [paired(80, 20), paired(90, 21)]
    q = paired(50, 22)
    m = MetaboCast(n_hvg=40, n_pcs=10, extra_key="celltype").fit(tr)
    m.predict_metabolome(q)
    rel = m.reliability_table(q, focus="lipid_redox")
    assert len(rel) == 2
    assert set(rel["annotation"]) == {"FA16:0", "Glutathione"}
    assert "mean_celltype_entropy" in rel.columns
    assert "cell_state_context" in rel.columns
    assert "metabocast_cell_state_summary" in q.uns


if __name__ == "__main__":
    test_shipped_model_loads_and_predicts()
    test_train_save_load_roundtrip()
    test_kegg_human_map_autoloads()
    test_histology_features_from_embedded_visium_image()
    test_morphology_predictor_train_predict_roundtrip()
    test_celltype_reliability_focus_filter()
    print("ALL SMOKE TESTS PASSED")
