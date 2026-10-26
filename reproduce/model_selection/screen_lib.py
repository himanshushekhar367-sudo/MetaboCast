"""Shared helpers for the MetaboCast model-selection screen (leakage-safe; labels of a test fold/section are never used
for fitting, feature construction or hyper-parameter choice)."""
import numpy as np, h5py, scipy.sparse as sp
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye
from sklearn.decomposition import PCA, NMF
from sklearn.ensemble import ExtraTreesRegressor

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))
import paths as _P                       # reproduce/paths.py: DESIUM_DIR, METABOCAST_DATA_ROOT, ...
DESIUM_DIR = _P.DESIUM
SMA_DIR = _os.environ.get("SMA_DIR", _os.path.join(_P.ROOT, "SMA", "sma_h5ad"))
GMT = _os.path.join(_P.REPO, "metabocast", "data", "genesets", "scMetab_KEGG.gmt")
DESIUM = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
ALPHAS = np.array([1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 200, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6])


def load_gmt(path=GMT):
    d = {}
    for ln in open(path):
        p = ln.rstrip("\n").split("\t")
        d[p[0]] = [g for g in p[2:] if g]
    return d


def _names(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def h5_strings(o):
    """Strings from an h5ad string column: a plain dataset, a nullable-string-array group
    (written by anndata with pandas >= 3) or a categorical group."""
    if isinstance(o, h5py.Dataset):
        return _names(o[:])
    enc = o.attrs.get("encoding-type", "")
    if enc == "categorical":
        cats = np.asarray(_names(o["categories"][:]), dtype=object)
        return list(cats[o["codes"][:]])
    return _names(o["values"][:])


def var_names(f):
    idx = f["var"].attrs.get("_index", "_index")
    return h5_strings(f["var"][idx])


def read_matrix(f, key, cols=None):
    """Dense float32 matrix from an h5ad dataset or csr/csc group; optional column subset."""
    o = f[key]
    if isinstance(o, h5py.Dataset):
        if cols is None:
            return o[:].astype(np.float32)
        order = np.argsort(cols)
        out = np.empty((o.shape[0], len(cols)), np.float32)
        for a in range(0, o.shape[0], 512):
            out[a:a + 512][:, order] = o[a:a + 512, :][:, np.asarray(cols)[order]]
        return out
    enc = o.attrs["encoding-type"]
    shape = tuple(o.attrs["shape"])
    M = (sp.csr_matrix if "csr" in enc else sp.csc_matrix)((o["data"][:], o["indices"][:], o["indptr"][:]), shape=shape)
    if cols is not None:
        M = M.tocsc()[:, cols]
    return np.asarray(M.toarray(), np.float32)


def gene_stats(path, key="layers/log1p"):
    """Per-gene sum and sum of squares without densifying the full matrix."""
    with h5py.File(path, "r") as f:
        o = f[key]
        if isinstance(o, h5py.Dataset):
            s1 = np.zeros(o.shape[1]); s2 = np.zeros(o.shape[1])
            for a in range(0, o.shape[0], 512):
                b = o[a:a + 512, :].astype(np.float64); s1 += b.sum(0); s2 += (b * b).sum(0)
            return s1, s2, o.shape[0]
        enc = o.attrs["encoding-type"]; shape = tuple(o.attrs["shape"])
        M = (sp.csr_matrix if "csr" in enc else sp.csc_matrix)((o["data"][:], o["indices"][:], o["indptr"][:]), shape=shape).astype(np.float64)
        return np.asarray(M.sum(0)).ravel(), np.asarray(M.multiply(M).sum(0)).ravel(), shape[0]


def norm_adj(c, k=6):
    """D^-1/2 A kNN adjacency with self loops (identical to the released predictor)."""
    n = len(c)
    _, idx = cKDTree(c).query(c, k=min(k + 1, n))
    r = np.repeat(np.arange(n), idx.shape[1] - 1); cc = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(r)), (r, cc)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1 / np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)) @ A


def diffuse(E, A, hops=2):
    out = [E]; cur = E
    for _ in range(hops):
        cur = A @ cur; out.append(cur)
    return np.hstack(out).astype(np.float32)


def pathway_scores(X, names, gmt):
    pos = {g: i for i, g in enumerate(names)}; cols = []
    for gs in gmt.values():
        idx = [pos[g] for g in gs if g in pos]
        if len(idx) >= 3:
            v = X[:, idx]; cols.append(((v - v.mean(0)) / (v.std(0) + 1e-8)).mean(1))
    return np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(X), 0), np.float32)


def block_scale(blocks, tr):
    """Scale each feature block so its mean training-column variance is 1 (for concatenated feature sets)."""
    out = []
    for B in blocks:
        v = B[tr].var(0).mean()
        out.append(B / np.sqrt(v + 1e-12))
    return np.hstack(out).astype(np.float32)


class RidgePath:
    """Exact multi-output ridge with intercept (as sklearn Ridge) for many alphas from one SVD."""
    def __init__(self, F, Y):
        F = np.asarray(F, np.float64); Y = np.asarray(Y, np.float64)
        self.fm = F.mean(0); self.ym = Y.mean(0)
        U, s, Vt = np.linalg.svd(F - self.fm, full_matrices=False)
        self.s, self.Vt = s, Vt
        self.UtY = U.T @ (Y - self.ym)

    def predict(self, Fte, alpha):
        Z = (np.asarray(Fte, np.float64) - self.fm) @ self.Vt.T
        a = np.atleast_1d(alpha)
        if a.size == 1:
            return ((Z * (self.s / (self.s ** 2 + a[0]))) @ self.UtY + self.ym).astype(np.float32)
        # per-target alpha
        out = np.empty((Z.shape[0], self.UtY.shape[1]), np.float32)
        for av in np.unique(a):
            j = np.where(a == av)[0]
            out[:, j] = (Z * (self.s / (self.s ** 2 + av))) @ self.UtY[:, j] + self.ym[j]
        return out

    def predict_path(self, Fte, alphas):
        Z = (np.asarray(Fte, np.float64) - self.fm) @ self.Vt.T
        for a in alphas:
            yield (Z * (self.s / (self.s ** 2 + a))) @ self.UtY + self.ym


def col_pearson(P, Y):
    Pc = P - P.mean(0); Yc = Y - Y.mean(0)
    den = np.sqrt((Pc * Pc).sum(0) * (Yc * Yc).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Pc * Yc).sum(0) / den
    r[den <= 1e-12] = np.nan
    return r


def col_spearman(P, Y):
    return col_pearson(rankdata(P, axis=0), rankdata(Y, axis=0))


def select_alpha(scores):
    """scores: (n_alpha, n_ions) inner-CV correlations. Returns (global alpha index, per-ion alpha indices)."""
    with np.errstate(all="ignore"):
        g = np.nanmedian(scores, axis=1)
    gi = int(np.nanargmax(g))
    filled = np.where(np.isnan(scores), -np.inf, scores)
    pi = filled.argmax(0)
    allnan = np.all(np.isnan(scores), axis=0)
    pi[allnan] = gi
    return gi, pi


def et_on_ypcs(Ftr, Ytr, Fte, ncomp=50, n_estimators=150, leaf=5, seed=0):
    mu = Ytr.mean(0); sd = Ytr.std(0) + 1e-8
    Z = (Ytr - mu) / sd
    pca = PCA(min(ncomp, Z.shape[1], Z.shape[0] - 1), svd_solver="randomized", random_state=0).fit(Z)
    T = pca.transform(Z)
    et = ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=leaf, max_features=0.33,
                             n_jobs=1, random_state=seed).fit(Ftr, T)
    return (pca.inverse_transform(et.predict(Fte)) * sd + mu).astype(np.float32)


def nmf_rep(Xtr, Xall, k=100, seed=0):
    m = NMF(k, init="nndsvda", max_iter=200, random_state=seed).fit(np.clip(Xtr, 0, None))
    return m.transform(np.clip(Xall, 0, None)).astype(np.float32)


def rank_gauss(X):
    from scipy.special import ndtri
    R = rankdata(X, axis=0)
    return ndtri((R - 0.5) / X.shape[0]).astype(np.float32)


class SectionGram:
    """Per-section sufficient statistics for fast exact ridge on any union of sections."""
    def __init__(self, F, Y):
        F = np.asarray(F, np.float64); Y = np.asarray(Y, np.float64)
        self.n = F.shape[0]; self.fs = F.sum(0); self.ys = Y.sum(0)
        self.FF = F.T @ F; self.FY = F.T @ Y


class GramRidge:
    """Exact multi-output ridge with intercept from summed section statistics (equivalent to sklearn Ridge)."""
    def __init__(self, grams, cols=None):
        n = sum(g.n for g in grams); fs = sum(g.fs for g in grams); ys = sum(g.ys for g in grams)
        FF = sum(g.FF for g in grams); FY = sum(g.FY for g in grams)
        if cols is not None:
            FY = FY[:, cols]; ys = ys[cols]
        self.fm = fs / n; self.ym = ys / n
        G = FF - n * np.outer(self.fm, self.fm)
        C = FY - n * np.outer(self.fm, self.ym)
        lam, Q = np.linalg.eigh(G); self.lam = np.clip(lam, 0, None); self.Q = Q
        self.B = (Q.T @ C).astype(np.float32)

    def _Z(self, Fte):
        return ((np.asarray(Fte, np.float64) - self.fm) @ self.Q).astype(np.float32)

    def predict(self, Fte, alpha):
        Z = self._Z(Fte); a = np.atleast_1d(alpha)
        if a.size == 1:
            return (Z * (1.0 / (self.lam + a[0])).astype(np.float32)) @ self.B + self.ym.astype(np.float32)
        out = np.empty((Z.shape[0], self.B.shape[1]), np.float32)
        for av in np.unique(a):
            j = np.where(a == av)[0]
            out[:, j] = (Z * (1.0 / (self.lam + av)).astype(np.float32)) @ self.B[:, j] + self.ym[j].astype(np.float32)
        return out

    def predict_path(self, Fte, alphas):
        Z = self._Z(Fte)
        for a in alphas:
            yield (Z * (1.0 / (self.lam + a)).astype(np.float32)) @ self.B + self.ym.astype(np.float32)
