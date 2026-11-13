"""
Per-spot metabolic PATHWAY ACTIVITY scoring — a first-class MetaboCast output.

This reproduces what scMetabolism gives you (a spots x metabolic-pathways activity matrix,
from KEGG/Reactome gene sets, scored per spot with the standard mean-z or AUCell methods)
and adds two things scMetabolism does not do:

  1. SPATIAL smoothing  — activity is diffused over the tissue k-NN graph, so pathway maps
     are spatially coherent instead of per-spot-independent (the right prior for spatial data).
  2. METABOLITE grounding (optional) — because the same MetaboCast model predicts measured
     metabolite abundances, you can compare a pathway's transcript activity against the
     predicted abundance of its metabolites (see `consistency_with_prediction`).

Usage:
    from metabocast import PathwayScorer
    ps = PathwayScorer("data/genesets/scMetab_KEGG.gmt", method="aucell")
    scores = ps.score(adata, spatial_smooth=1)      # DataFrame: spots x pathways
    # also stored in adata.obsm['pathway_scores'] with names in adata.uns['pathway_names']
"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye


def load_gmt(path):
    d = {}
    for ln in open(path):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 3:
            d[p[0]] = [g for g in p[2:] if g]
    return d


def _dense(x):
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)


def _norm_adj(coords, k=6):
    n = len(coords)
    _, idx = cKDTree(coords).query(coords, k=min(k + 1, n))
    r = np.repeat(np.arange(n), idx.shape[1] - 1); c = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1 / np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)) @ A


class PathwayScorer:
    """scMetabolism-style metabolic pathway activity, with optional spatial smoothing."""

    def __init__(self, gmt, method="mean_z", min_genes=3, k=6, aucell_top=0.05):
        """gmt: path to a .gmt (e.g. scMetab_KEGG.gmt) or a {name: [genes]} dict.
        method: 'mean_z' (fast; mean of z-scored member genes) or 'aucell' (rank recovery AUC)."""
        self.sets = load_gmt(gmt) if isinstance(gmt, str) else dict(gmt)
        self.method = method; self.min_genes = min_genes; self.k = k; self.aucell_top = aucell_top

    def _matrix(self, adata, layer):
        if layer and layer in adata.layers:
            X = _dense(adata[:, :].layers[layer])
        else:
            X = _dense(adata.X)
        return X, list(adata.var_names)

    def score(self, adata, layer="log1p", spatial_smooth=0, key_added="pathway_scores"):
        """Return a spots x pathways DataFrame of metabolic pathway activity.
        spatial_smooth: number of graph-diffusion hops to smooth activity (0 = none)."""
        X, names = self._matrix(adata, layer)
        pos = {g: i for i, g in enumerate(names)}
        keep = {p: [pos[g] for g in gs if g in pos] for p, gs in self.sets.items()}
        keep = {p: ix for p, ix in keep.items() if len(ix) >= self.min_genes}
        if not keep:
            raise ValueError("No pathway had >= min_genes present — check gene symbols / species.")

        if self.method == "mean_z":
            mu, sd = X.mean(0), X.std(0) + 1e-8
            Z = (X - mu) / sd
            cols = [Z[:, ix].mean(1) for ix in keep.values()]
        elif self.method == "aucell":
            # per-spot gene ranks (1 = highest expression), then recovery-AUC per gene set
            order = np.argsort(-X, axis=1)
            R = np.empty_like(order, dtype=np.int32)
            R[np.arange(X.shape[0])[:, None], order] = np.arange(1, X.shape[1] + 1, dtype=np.int32)
            T = max(1, int(np.ceil(self.aucell_top * X.shape[1])))
            cols = []
            for ix in keep.values():
                Rg = R[:, ix]                                   # spots x |set|
                cols.append(np.maximum(0, T - Rg + 1).sum(1) / (len(ix) * T))
        else:
            raise ValueError("method must be 'mean_z' or 'aucell'")

        S = np.vstack(cols).T.astype(np.float32)               # spots x pathways
        pw = list(keep.keys())
        if spatial_smooth and "spatial" in adata.obsm:
            A = _norm_adj(np.asarray(adata.obsm["spatial"], float), self.k)
            for _ in range(int(spatial_smooth)):
                S = np.asarray(A @ S, dtype=np.float32)
        adata.obsm[key_added] = S
        adata.uns[key_added + "_names"] = pw
        return pd.DataFrame(S, index=adata.obs_names, columns=pw)

    @staticmethod
    def consistency_with_prediction(adata, pathway_df, model, mz_to_pathway):
        """OPTIONAL improvement over scMetabolism: for pathways whose metabolites are annotated
        (mz_to_pathway: {pathway_name: [m/z, ...]}), correlate the transcript-based pathway score
        with the mean PREDICTED abundance of that pathway's metabolites, per pathway.
        Returns a DataFrame: pathway, spearman(transcript_activity, predicted_metabolite_level)."""
        from scipy.stats import spearmanr
        pred = adata.obsm.get("metabocast_pred")
        if pred is None:
            pred = model.predict_metabolome(adata)
        mz = np.asarray(model.mz_, float)
        rows = []
        for p in pathway_df.columns:
            mzs = mz_to_pathway.get(p, [])
            cols = [int(np.argmin(np.abs(mz - m))) for m in mzs]
            if not cols:
                continue
            metab_level = pred[:, cols].mean(1)
            rho = spearmanr(pathway_df[p].values, metab_level)[0]
            rows.append((p, len(cols), rho))
        return pd.DataFrame(rows, columns=["pathway", "n_metabolites", "transcript_vs_predicted_rho"])


# ============================================================================
# Metabolite-grounded metabolic pathway activity — the part scMetabolism can't do
# ============================================================================
# curated neutral monoisotopic masses (Da) of the metabolites that map to KEGG pathways below; one name per formula
# (Hexose covers glucose, fructose, galactose and inositol)
_MASS_DB = {
    "Lactate": 90.0317, "Pyruvate": 88.0160, "Alanine": 89.0477, "Serine": 105.0426, "Glycine": 75.0320,
    "Succinate": 118.0266, "Fumarate": 116.0110, "Malate": 134.0215, "Aspartate": 133.0375, "Glutamate": 147.0532,
    "Glutamine": 146.0691, "a-Ketoglutarate": 146.0215, "Citrate/Isocitrate": 192.0270, "Hexose": 180.0634,
    "Hexose-P": 260.0297, "Taurine": 125.0147, "Creatine": 131.0695, "Creatinine": 113.0589, "Hypoxanthine": 136.0385,
    "Inosine": 268.0808, "AMP": 347.0631, "ADP": 427.0294, "ATP": 506.9957, "GSH": 307.0838, "GSSG": 612.1519,
    "Ascorbate": 176.0321, "Urate": 168.0283, "FA16:0": 256.2402, "FA18:0": 284.2715, "FA18:1": 282.2559,
    "FA18:2": 280.2402, "FA20:4": 304.2402, "FA22:6": 328.2402, "Sphingosine": 299.2824,
    "UDP-GlcNAc": 607.0817, "Carnitine": 161.1052, "Acetylcarnitine": 203.1158, "Kynurenine": 208.0848, "Adenosine": 267.0968,
}
_NEG_ADDUCTS = [("[M-H]-", -1.0073), ("[M+Cl]-", 34.9694), ("[M+HCOO]-", 44.9982)]
_POS_ADDUCTS = [("[M+H]+", 1.0073), ("[M+Na]+", 22.9892), ("[M+K]+", 38.9632)]
# curated metabolite -> KEGG metabolic pathway(s) (standard biochemistry)
_MET2PATH = {
    "Lactate": ["Glycolysis / Gluconeogenesis", "Pyruvate metabolism"],
    "Pyruvate": ["Glycolysis / Gluconeogenesis", "Pyruvate metabolism", "Citrate cycle (TCA cycle)"],
    "Hexose": ["Glycolysis / Gluconeogenesis", "Galactose metabolism", "Fructose and mannose metabolism"],
    "Hexose-P": ["Glycolysis / Gluconeogenesis", "Pentose phosphate pathway"],
    "Succinate": ["Citrate cycle (TCA cycle)"], "Fumarate": ["Citrate cycle (TCA cycle)"],
    "Malate": ["Citrate cycle (TCA cycle)", "Pyruvate metabolism"], "Citrate/Isocitrate": ["Citrate cycle (TCA cycle)"],
    "a-Ketoglutarate": ["Citrate cycle (TCA cycle)", "Alanine, aspartate and glutamate metabolism"],
    "Alanine": ["Alanine, aspartate and glutamate metabolism"], "Aspartate": ["Alanine, aspartate and glutamate metabolism"],
    "Glutamate": ["Alanine, aspartate and glutamate metabolism", "Arginine and proline metabolism"],
    "Glutamine": ["Alanine, aspartate and glutamate metabolism"], "Glycine": ["Glycine, serine and threonine metabolism"],
    "Serine": ["Glycine, serine and threonine metabolism"],
    "Creatine": ["Glycine, serine and threonine metabolism", "Arginine and proline metabolism"],
    "Creatinine": ["Arginine and proline metabolism"], "Taurine": ["Taurine and hypotaurine metabolism"],
    "Hypoxanthine": ["Purine metabolism"], "Inosine": ["Purine metabolism"], "AMP": ["Purine metabolism"],
    "ADP": ["Purine metabolism"], "ATP": ["Purine metabolism"], "Adenosine": ["Purine metabolism"], "Urate": ["Purine metabolism"],
    "GSH": ["Glutathione metabolism"], "GSSG": ["Glutathione metabolism"], "Ascorbate": ["Ascorbate and aldarate metabolism"],
    "FA16:0": ["Fatty acid biosynthesis", "Fatty acid degradation"], "FA18:0": ["Fatty acid biosynthesis", "Fatty acid elongation"],
    "FA18:1": ["Biosynthesis of unsaturated fatty acids", "Fatty acid degradation"],
    "FA18:2": ["Linoleic acid metabolism", "Biosynthesis of unsaturated fatty acids"],
    "FA20:4": ["Arachidonic acid metabolism", "Biosynthesis of unsaturated fatty acids"],
    "FA22:6": ["Biosynthesis of unsaturated fatty acids", "alpha-Linolenic acid metabolism"],
    "Sphingosine": ["Sphingolipid metabolism"],
    "UDP-GlcNAc": ["Amino sugar and nucleotide sugar metabolism"], "Carnitine": ["Fatty acid degradation"],
    "Acetylcarnitine": ["Fatty acid degradation"], "Kynurenine": ["Tryptophan metabolism"],
}


class MetabolicActivity:
    """Metabolite-grounded metabolic pathway activity.

    Unlike scMetabolism / AUCell / ssGSEA (which score pathways from ENZYME transcripts only),
    this scores a pathway high only where its enzymes AND its metabolites agree — the flux
    signature — and reports the enzyme-metabolite spatial COHERENCE per pathway as a confidence
    (which pathway scores to trust). Metabolites come from a fitted MetaboCast model (predicted)
    or from measured `adata.uns['msi']`; each m/z is assigned the nearest candidate of a curated mass DB + adducts.
    """

    def __init__(self, gmt, mass_db=None, met2path=None, adducts="neg", tol=0.01, min_genes=3):
        self.sets = load_gmt(gmt) if isinstance(gmt, str) else dict(gmt)
        self.mass_db = mass_db or _MASS_DB
        self.met2path = met2path or _MET2PATH
        self.adducts = _NEG_ADDUCTS if adducts == "neg" else (_POS_ADDUCTS if adducts == "pos" else adducts)
        self.tol = tol; self.min_genes = min_genes

    def annotate(self, mz):
        """m/z -> name of the nearest candidate within tol (curated mass DB + adducts; ties by name).
        Returns {index: name}."""
        mz = np.asarray(mz, float); out = {}
        for i, v in enumerate(mz):
            hits = [(abs(v - (ms + dl)), nm) for nm, ms in self.mass_db.items() for _, dl in self.adducts
                    if abs(v - (ms + dl)) <= self.tol]
            if hits:
                out[i] = min(hits)[1]
        return out

    def score(self, adata, model=None, metab_key="msi", mz_key="mz_features", layer="log1p", spatial_smooth=0):
        """Return (activity_df: spots x pathways, confidence_df, metabolite_source).
        confidence_df ranks pathways by enzyme_metabolite_coherence and flags metabolite_supported."""
        from scipy.stats import spearmanr, rankdata
        X = _dense(adata[:, :].layers[layer]) if (layer and layer in adata.layers) else _dense(adata.X)
        if X.max() > 50:
            X = np.log1p(X / (X.sum(1, keepdims=True) + 1) * 1e4)
        Zg = (X - X.mean(0)) / (X.std(0) + 1e-8)
        gpos = {g: k for k, g in enumerate(adata.var_names)}

        if model is not None:
            Pm = model.predict_metabolome(adata); mz = np.asarray(model.mz_, float); src = "predicted"
        elif metab_key in adata.uns:
            Pm = np.log1p(np.asarray(adata.uns[metab_key], float)); mz = np.asarray(adata.uns[mz_key], float); src = "measured"
        else:
            Pm = None; mz = None; src = "none"
        pmet = {}
        if mz is not None:
            Zm = (Pm - Pm.mean(0)) / (Pm.std(0) + 1e-8)
            for i, nm in self.annotate(mz).items():
                for p in self.met2path.get(nm, []):
                    pmet.setdefault(p, []).append(i)

        acts, names, rows = [], [], []
        for p, gs in self.sets.items():
            gi = [gpos[g] for g in gs if g in gpos]
            if len(gi) < self.min_genes:
                continue
            E = Zg[:, gi].mean(1); eR = rankdata(E) / len(E)
            mi = sorted(set(pmet.get(p, [])))
            if mi:
                M = Zm[:, mi].mean(1); mR = rankdata(M) / len(M)
                A = (eR * mR).astype(np.float32); coh = float(spearmanr(E, M)[0]); supported = True
            else:
                A = eR.astype(np.float32); coh = float("nan"); supported = False
            acts.append(A); names.append(p); rows.append((p, len(gi), len(mi), coh, supported))
        S = np.vstack(acts).T.astype(np.float32)
        if spatial_smooth and "spatial" in adata.obsm:
            Adj = _norm_adj(np.asarray(adata.obsm["spatial"], float))
            for _ in range(int(spatial_smooth)):
                S = np.asarray(Adj @ S, dtype=np.float32)
        adata.obsm["metabolic_activity"] = S; adata.uns["metabolic_activity_names"] = names
        conf = pd.DataFrame(rows, columns=["pathway", "n_genes", "n_metabolites",
                                           "enzyme_metabolite_coherence", "metabolite_supported"])
        conf = conf.sort_values("enzyme_metabolite_coherence", ascending=False, na_position="last").reset_index(drop=True)
        return pd.DataFrame(S, index=adata.obs_names, columns=names), conf, src
