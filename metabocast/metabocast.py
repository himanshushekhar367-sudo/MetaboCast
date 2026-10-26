"""
MetaboCast — prediction of spatial metabolite (MSI ion) maps from spatial transcriptomics, with per-ion reliability.

Minimal, scanpy-compatible API:
    from metabocast import MetaboCast
    model = MetaboCast(use_kegg=True).fit(train_adatas)   # each adata: .layers['log1p'] + .uns['msi'] + .uns['mz_features'] + .obsm['spatial']
    pred  = model.predict_metabolome(query_adata)          # -> (n_spots x n_metab); also stored in query_adata.obsm['metabocast_pred']

Design: HVG -> PCA -> diffusion features [P, A P, A^2 P] over the spot graph (+ optional KEGG pathway scores)
-> multi-output regression. The default training objective rank-transforms each ion within each
training section to Gaussian scores (target_transform="rank"; rank harmonisation removes monotone
between-section intensity shifts) and chooses the ridge penalty by inner cross-validation over training
sections (alpha="auto"; with a single training section, spatial blocks whose held-out predictions are
pooled into one map before scoring). The default estimator averages the
ridge with an ExtraTrees regressor fitted to the same features and rank targets (estimator="ensemble");
estimator="ridge" keeps the linear model alone. Predictions are mapped back to log1p intensity through each
ion's pooled training distribution, a monotone map that preserves the predicted spatial ordering.
MetaboCast(alpha=200, target_transform=None, estimator="ridge") instead fits one ridge with a fixed penalty to
per-ion standardised log1p intensities.
Genes are matched by symbol; the model predicts the training metabolite (m/z) axis. Query genes absent
from the training panel are imputed at the training per-gene mean, so they contribute exactly zero after
PCA centring. Outputs are log1p MSI intensities (non-negative). Optional residual-based widths are
attached to the query as .uns['metabocast_conf_width']; .uns['metabocast_uncertainty'] records their
provenance.
"""
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, eye


def _package_version():
    from . import __version__
    return __version__


def _default_kegg_gmt():
    """Locate the human KEGG metabolic gene sets shipped with the repo (85 metabolic
    pathways, ~1.7k human gene symbols). Returns a path or None. Works for editable
    installs (repo layout) and for the packaged data dir; env override MC_KEGG_GMT wins."""
    env = os.environ.get("MC_KEGG_GMT")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "data", "genesets", "scMetab_KEGG.gmt"),          # packaged inside the module
        os.path.join(here, "..", "data", "genesets", "scMetab_KEGG.gmt"),     # repo root (editable install)
        os.path.join(os.getcwd(), "data", "genesets", "scMetab_KEGG.gmt"),    # run from repo root
    ):
        if os.path.exists(cand):
            return os.path.normpath(cand)
    return None


def _norm_adj(coords, groups=None, k=6):
    """Degree-normalized (D^{-1/2} A) symmetric kNN adjacency + self loops, built within each group.

    NOTE: this is the left/degree normalization used by the released predictor; it is
    intentionally not the symmetric D^{-1/2} A D^{-1/2}. Changing it would alter the diffusion operator
    and break reproducibility with the reported results.
    """
    n = len(coords); groups = np.zeros(n) if groups is None else np.asarray(groups)
    rows, cols = [], []
    for g in np.unique(groups):
        gi = np.where(groups == g)[0]
        if len(gi) < 2: continue
        _, idx = cKDTree(coords[gi]).query(coords[gi], k=min(k + 1, len(gi)))
        for a in range(len(gi)):
            for b in idx[a, 1:]:
                rows.append(gi[a]); cols.append(gi[b])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1 / np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)) @ A


def _dense(x):
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)


#: Ridge penalties searched when alpha="auto" (features are unscaled PCA/diffusion features, so useful
#: penalties span many orders of magnitude).
ALPHA_GRID = np.array([1e-1, 1, 10, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7])


def _rank_gauss(Y):
    """Per-column rank-Gaussian (van der Waerden) scores. Tied values, including undetected zeros,
    share their average rank, so the transform is invariant to any monotone intensity rescaling."""
    from scipy.stats import rankdata
    from scipy.special import ndtri
    Y = np.asarray(Y, np.float64)
    R = rankdata(Y, axis=0)
    return ndtri((R - 0.5) / Y.shape[0]).astype(np.float32)


class _Gram:
    """Sufficient statistics of one training unit (a section or a spatial block) for exact ridge."""
    def __init__(self, F, T):
        F = np.asarray(F, np.float64); T = np.asarray(T, np.float64)
        self.n = F.shape[0]; self.fs = F.sum(0); self.ts = T.sum(0)
        self.FF = F.T @ F; self.FT = F.T @ T


def _gram_solver(grams, cols=None):
    """Eigen-decomposed ridge system for the union of `grams` (identical to sklearn Ridge with an
    intercept); returns a function mapping (F, alpha) -> predictions for the selected target columns."""
    n = sum(g.n for g in grams); fs = sum(g.fs for g in grams); ts = sum(g.ts for g in grams)
    FT = sum(g.FT for g in grams)
    if cols is not None:
        FT = FT[:, cols]; ts = ts[cols]
    fm, tm = fs / n, ts / n
    G = sum(g.FF for g in grams) - n * np.outer(fm, fm)
    lam, Q = np.linalg.eigh(G); lam = np.clip(lam, 0, None)
    B = Q.T @ (FT - n * np.outer(fm, tm))

    def predict(F, alpha):
        Z = (np.asarray(F, np.float64) - fm) @ Q
        return (Z / (lam + alpha)) @ B + tm
    return predict


class _CompactForest:
    """Array form of a fitted multi-output scikit-learn forest regressor, used for the ExtraTrees half of the
    ensemble so that a saved model stays small. Only what prediction needs is kept: split feature and
    threshold per node, child pointers and float32 leaf values (internal-node values are dropped).
    Thresholds are rounded DOWN to float32, so `x <= t32` on float32 inputs is exactly sklearn's
    `x <= t64`; predictions match sklearn to float32 precision."""

    def __init__(self, forest):
        feat, thr, left, right, leafrow, vals = [], [], [], [], [], []
        offs, voffs = [0], [0]
        for est in forest.estimators_:
            t = est.tree_
            th = t.threshold.astype(np.float32)
            up = th.astype(np.float64) > t.threshold
            th[up] = np.nextafter(th[up], np.float32(-np.inf))
            leaf = t.children_left == -1
            row = np.full(t.node_count, -1, np.int32); row[leaf] = np.arange(int(leaf.sum()), dtype=np.int32)
            feat.append(np.maximum(t.feature, 0).astype(np.int32)); thr.append(th)
            left.append(t.children_left.astype(np.int32)); right.append(t.children_right.astype(np.int32))
            leafrow.append(row); vals.append(t.value[leaf, :, 0].astype(np.float32))
            offs.append(offs[-1] + t.node_count); voffs.append(voffs[-1] + int(leaf.sum()))
        self.feature = np.concatenate(feat); self.threshold = np.concatenate(thr)
        self.left = np.concatenate(left); self.right = np.concatenate(right)
        self.leafrow = np.concatenate(leafrow); self.values = np.concatenate(vals)
        self.offs = np.asarray(offs, np.int64); self.voffs = np.asarray(voffs, np.int64)
        self.n_trees = len(forest.estimators_); self.n_outputs = self.values.shape[1]
        self.n_features = int(forest.n_features_in_)

    def predict(self, X):
        X = np.asarray(X, np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape}")
        out = np.zeros((X.shape[0], self.n_outputs), np.float64)
        rows = np.arange(X.shape[0])
        for i in range(self.n_trees):
            a, b = self.offs[i], self.offs[i + 1]
            feat, thr, L, R = self.feature[a:b], self.threshold[a:b], self.left[a:b], self.right[a:b]
            node = np.zeros(X.shape[0], np.int64)
            act = rows if L[0] != -1 else rows[:0]
            while act.size:
                nd = node[act]
                nxt = np.where(X[act, feat[nd]] <= thr[nd], L[nd], R[nd])
                node[act] = nxt
                act = act[L[nxt] != -1]
            out += self.values[self.voffs[i]:self.voffs[i + 1]][self.leafrow[a:b][node]]
        return out / self.n_trees


def _equal_weights(r_tr, e_tr, floor=1e-6):
    """Per-ion weights that give the ridge and tree predictions equal spread on the training spots.

    Each half is centred on its training mean and scaled to the geometric mean of the two training-set
    standard deviations, so neither half dominates the average merely because it is shrunk less (a
    heavily penalised ridge has a much smaller spread than the trees). Returns (w_r, w_e, m_r, m_e);
    the combination is 0.5 (m_r + m_e) + w_r (r - m_r) + w_e (e - m_e). An ion whose half is constant
    on the training spots is predicted by the other half alone."""
    r_tr = np.asarray(r_tr, np.float64); e_tr = np.asarray(e_tr, np.float64)
    m_r, m_e = r_tr.mean(0), e_tr.mean(0)
    sr, se = r_tr.std(0), e_tr.std(0)
    ok_r, ok_e = sr > floor, se > floor
    both = ok_r & ok_e
    g = np.sqrt(np.where(both, sr * se, 1.0))
    w_r = np.full(sr.shape, 0.5); w_e = np.full(se.shape, 0.5)          # neither half varies
    w_r[both] = 0.5 * g[both] / sr[both]; w_e[both] = 0.5 * g[both] / se[both]
    w_r[ok_r & ~ok_e], w_e[ok_r & ~ok_e] = 1.0, 0.0                       # trees constant: ridge alone
    w_r[ok_e & ~ok_r], w_e[ok_e & ~ok_r] = 0.0, 1.0                       # ridge constant: trees alone
    return (w_r.astype(np.float32), w_e.astype(np.float32), m_r.astype(np.float32), m_e.astype(np.float32))


def _col_pearson(P, Y):
    P = P - P.mean(0); Y = Y - Y.mean(0)
    den = np.sqrt((P * P).sum(0) * (Y * Y).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (P * Y).sum(0) / den
    r[~(den > 1e-12)] = np.nan
    return r


def select_alpha(units, grid=ALPHA_GRID, pooled=False):
    """Choose the ridge penalty by leave-one-unit-out cross-validation.

    `units` is a list of (features, targets, detected-ion mask) tuples, one per training section (or
    spatial block). Each unit is predicted from a model fitted on the others.

    pooled=False (sections): the score of a penalty is the mean over units of the median per-ion Pearson
    correlation on that unit's detected ions, matching an evaluation that scores each held-out section
    on its own.
    pooled=True (spatial blocks of one section): the held-out predictions of all units are pooled into
    one map and the score is the median per-ion Pearson correlation of that map, matching a within-
    section evaluation in which the held-out blocks together make up the section. Scoring each block
    on its own instead ignores whether the model gets the differences between blocks right, and can
    favour penalties so large that the predictions collapse to the training mean. The detected-ion
    mask of the first unit is used.
    Returns (alpha, scores): scores has one column per unit, or a single column when pooled."""
    grams = [_Gram(F, T) for F, T, _ in units]
    if pooled:
        det = units[0][2]
        j = np.where(det)[0] if det is not None and np.any(det) else np.arange(units[0][1].shape[1])
        A = len(grid); J = len(j)
        sp, spp, spt = np.zeros((A, J)), np.zeros((A, J)), np.zeros((A, J))
        st, stt, n = np.zeros(J), np.zeros(J), 0
        for k, (F, T, _) in enumerate(units):
            T = np.asarray(T, np.float64)[:, j]
            solve = _gram_solver([g for i, g in enumerate(grams) if i != k], cols=j)
            st += T.sum(0); stt += (T * T).sum(0); n += len(T)
            for a, alpha in enumerate(grid):
                P = solve(F, alpha)
                sp[a] += P.sum(0); spp[a] += (P * P).sum(0); spt[a] += (P * T).sum(0)
        with np.errstate(all="ignore"):
            cov = spt / n - (sp / n) * (st / n)
            r = cov / np.sqrt((spp / n - (sp / n) ** 2) * (stt / n - (st / n) ** 2))
            r[~np.isfinite(r)] = np.nan
            scores = np.nanmedian(r, axis=1)[:, None]
    else:
        scores = np.full((len(grid), len(units)), np.nan)
        for k, (F, T, det) in enumerate(units):
            j = np.where(det)[0] if det is not None and np.any(det) else np.arange(T.shape[1])
            solve = _gram_solver([g for i, g in enumerate(grams) if i != k], cols=j)
            for a, alpha in enumerate(grid):
                with np.errstate(all="ignore"):
                    scores[a, k] = np.nanmedian(_col_pearson(solve(F, alpha), np.asarray(T, np.float64)[:, j]))
    with np.errstate(all="ignore"):
        mean = np.nanmean(scores, axis=1)
    if not np.isfinite(mean).any():
        return None, scores
    return float(grid[int(np.nanargmax(mean))]), scores


#: Default m/z agreement tolerance, in Da. Two sections are on the same axis when every channel
#: agrees to within this. 1 mDa is orders of magnitude above float32 storage noise (a float32
#: m/z near 1000 Da is off by up to ~6e-5 Da, so the same axis saved in different dtypes still
#: matches) and orders of magnitude below any real channel spacing or the +-0.01 Da used for
#: putative annotation, so a genuinely different grid is still caught.
MZ_ATOL = 1e-3


def _mz_axis(adata, mz_key="mz_features"):
    """One section's metabolite (m/z) axis as a flat float64 array."""
    axis = np.asarray(adata.uns[mz_key], float)
    if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() or (axis <= 0).any():
        raise ValueError("m/z axis must be a nonempty, finite, positive vector")
    if len(np.unique(axis)) != axis.size:
        raise ValueError("m/z axis must not contain duplicate channels")
    return axis.copy()


def common_mz_axis(adatas, mz_key="mz_features", atol=MZ_ATOL):
    """Return the m/z axis shared by every section, or raise ValueError.

    Sections are pooled column-wise into a single multi-output regression, so column j of
    every section's target matrix must be the SAME ion. Matching axis *lengths* are not
    sufficient evidence of that: two cohorts can both report 2,086 channels on different
    m/z grids, in which case pooling silently fits each ion's weights to another ion's
    intensities. Values are therefore compared elementwise, to `atol` Da (see MZ_ATOL).
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("m/z tolerance must be finite and nonnegative")
    if len(adatas) == 0:
        raise ValueError("need at least one section to determine the m/z axis")
    ref = _mz_axis(adatas[0], mz_key)
    for i, a in enumerate(adatas[1:], start=1):
        cur = _mz_axis(a, mz_key)
        if cur.shape != ref.shape:
            raise ValueError(
                f"m/z axis mismatch: section 0 has {ref.size} channels but section {i} has "
                f"{cur.size}. Only sections on one common m/z axis can be pooled — re-bin them "
                f"onto a shared axis or train the cohorts separately.")
        bad = ~np.isclose(cur, ref, rtol=0.0, atol=atol, equal_nan=True)
        if bad.any():
            j = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"m/z axis mismatch: section {i} differs from section 0 in {int(bad.sum())} of "
                f"{ref.size} channels (first at index {j}: {cur[j]:.6f} vs {ref[j]:.6f}, "
                f"|delta| = {abs(cur[j] - ref[j]):.3g} Da > atol = {atol:g}). Pooling these "
                f"sections would fit each ion to a different ion's intensities — re-bin onto a "
                f"shared axis, raise mz_atol if the grids are genuinely identical, or train the "
                f"cohorts separately.")
    return ref


def group_sections_by_mz_axis(adatas, mz_key="mz_features", atol=MZ_ATOL):
    """Group section indices by their actual m/z axis, largest group first.

    Returns a list of index lists; size ties keep first-appearance order. Sections are
    bucketed by comparing axis *values* under the same tolerance `common_mz_axis` uses,
    not by axis length, so same-length cohorts on different m/z grids fall into separate
    groups instead of being pooled (and one axis saved in different dtypes stays one
    group). Use this to pick a poolable set; `common_mz_axis`, called by fit, remains the
    authoritative check.
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("m/z tolerance must be finite and nonnegative")
    reps = []                                    # [(axis, [section indices]), ...] in first-seen order
    for i, a in enumerate(adatas):
        cur = _mz_axis(a, mz_key)
        for axis, members in reps:
            if cur.shape == axis.shape and np.allclose(cur, axis, rtol=0.0, atol=atol, equal_nan=True):
                members.append(i)
                break
        else:
            reps.append((cur, [i]))
    ranked = sorted(range(len(reps)), key=lambda p: (-len(reps[p][1]), p))
    return [reps[p][1] for p in ranked]


class MetaboCast:
    def __init__(self, n_hvg=3000, n_pcs=100, alpha="auto", k=6, use_kegg=False, kegg_gmt=None, extra_key=None,
                 target_transform="rank", alpha_grid=None, estimator="ensemble", n_trees=100, tree_min_leaf=10,
                 n_target_pcs=50, random_state=0, gene_scaling=None, n_jobs=1):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if not isinstance(n_pcs, (int, np.integer)) or n_pcs < 1:
            raise ValueError("n_pcs must be a positive integer")
        if n_hvg is not None and (not isinstance(n_hvg, (int, np.integer)) or n_hvg < 0):
            raise ValueError("n_hvg must be a nonnegative integer or None")
        if isinstance(alpha, str):
            if alpha != "auto":
                raise ValueError('alpha must be "auto" or a finite nonnegative number')
        elif not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if target_transform not in ("rank", None):
            raise ValueError('target_transform must be "rank" or None')
        if gene_scaling not in (None, "section"):
            raise ValueError('gene_scaling must be None or "section" (z-score genes within each section)')
        if estimator not in ("ridge", "ensemble"):
            raise ValueError('estimator must be "ridge" or "ensemble" (ridge + ExtraTrees on the same targets)')
        self.n_hvg, self.n_pcs, self.alpha, self.k = n_hvg, n_pcs, alpha, k
        self.target_transform = target_transform
        self.alpha_grid = None if alpha_grid is None else np.asarray(alpha_grid, float)
        self.estimator, self.n_trees, self.tree_min_leaf = estimator, int(n_trees), int(tree_min_leaf)
        self.n_target_pcs, self.random_state, self.n_jobs = int(n_target_pcs), random_state, n_jobs
        self.gene_scaling = gene_scaling
        self.use_kegg = use_kegg
        self.extra_key = extra_key   # optional per-spot modality in adata.obsm (e.g. antibody-derived protein); concatenated to features
        # Human metabolic map: when use_kegg=True and no explicit .gmt is given, fall back to the
        # shipped human KEGG metabolic gene sets so the flag is zero-config (not silently inert).
        gmt = kegg_gmt or (_default_kegg_gmt() if use_kegg else None)
        self.kegg_gmt_ = gmt
        self.kegg = self._load_gmt(gmt) if (use_kegg and gmt) else {}
        if use_kegg and not self.kegg:
            import warnings
            warnings.warn("use_kegg=True but no KEGG gene sets could be loaded "
                          "(set kegg_gmt=... or MC_KEGG_GMT); pathway features are disabled.")
        self.last_gene_overlap_ = float("nan")   # set on every predict_metabolome call
        self.conf_width_ = None                  # per-metabolite uncertainty half-width (if estimated)
        self.extra_train_mean_ = None
        self.extra_train_sd_ = None

    @staticmethod
    def _load_gmt(path):
        d = {}
        for ln in open(path):
            p = ln.rstrip("\n").split("\t"); d[p[0]] = [g for g in p[2:] if g]
        return d

    def _pathway_scores(self, X, var_names):
        pos = {g: i for i, g in enumerate(var_names)}; cols = []
        for gs in self.kegg.values():
            idx = [pos[g] for g in gs if g in pos]
            if len(idx) >= 3:
                z = (X[:, idx] - X[:, idx].mean(0)) / (X[:, idx].std(0) + 1e-8); cols.append(z.mean(1))
        return np.array(cols).T.astype(np.float32) if cols else np.zeros((X.shape[0], 0), np.float32)

    def _features(self, X_full, var_names, coords, groups, extra=None):
        # X_full columns are aligned to self.genes_; select HVG columns by position
        hvg_cols = np.asarray([self._gpos[g] for g in self.hvg_genes_])
        Xh = X_full[:, hvg_cols]
        if getattr(self, "gene_scaling", None) == "section":
            Xh = self._section_scale(Xh, groups)
        P = (Xh @ self.Vt_.T - self.pmean_ @ self.Vt_.T).astype(np.float32)
        PW = (self._pathway_scores(X_full, var_names)
              if getattr(self, "use_kegg", False) and getattr(self, "kegg", None) else None)
        return self._diffusion_features(P, coords, groups, extra=extra, PW=PW)

    def _diffusion_features(self, P, coords, groups, extra=None, PW=None):
        """[P, A P, A^2 P] (+ pathway scores and an extra modality, each with one diffused copy)."""
        A = _norm_adj(coords, groups, self.k); P1 = A @ P; F = np.hstack([P, P1, A @ P1])
        if PW is not None:
            F = np.hstack([F, PW, A @ PW])
        if extra is not None:                                   # optional extra modality (e.g. protein): value + neighbour-diffused
            E = np.asarray(extra, np.float32); F = np.hstack([F, E, A @ E])
        return F

    def _query_pca_scores(self, adata, chunk=2048):
        """PCA scores of a query computed in row chunks from its highly variable genes only, so a large query
        (tens of thousands of spots) is never densified across the whole gene panel. Genes absent from the
        query take the training mean, exactly as in the full path; the result equals the first step of
        _features."""
        pos = {g: i for i, g in enumerate(list(adata.var_names))}
        hv = list(self.hvg_genes_)
        qcol = np.array([pos.get(g, -1) for g in hv])
        have = qcol >= 0
        gmean = (np.asarray(self.gene_mean_, np.float32)[[self._gpos[g] for g in hv]]
                 if getattr(self, "gene_mean_", None) is not None else np.zeros(len(hv), np.float32))
        M = adata.layers["log1p"] if "log1p" in adata.layers else adata.X
        VtT = np.asarray(self.Vt_, np.float32).T; off = (self.pmean_ @ self.Vt_.T).astype(np.float32)
        P = np.empty((adata.n_obs, VtT.shape[1]), np.float32)
        for s in range(0, adata.n_obs, chunk):
            blk = _dense(M[s:s + chunk][:, qcol[have]])
            if not np.isfinite(blk).all():
                raise ValueError("query gene expression must be finite")
            Xh = np.empty((blk.shape[0], len(hv)), np.float32)
            Xh[:, have] = blk; Xh[:, ~have] = gmean[~have]
            P[s:s + chunk] = (Xh @ VtT - off).astype(np.float32)
        return P

    def fit(self, adatas, metab_key="msi", mz_key="mz_features", mz_atol=MZ_ATOL):
        # every pooled section must sit on ONE m/z axis, verified on values (not just length),
        # because column j of the stacked target matrix has to be the same ion in every section.
        # Checked up front so a mismatched cohort fails before the expensive passes below.
        axis = common_mz_axis(adatas, mz_key, atol=mz_atol)
        extra_size = None
        for a in adatas:
            target = np.asarray(a.uns[metab_key], float)
            if target.shape != (a.n_obs, len(axis)) or not np.isfinite(target).all() or (target < 0).any():
                raise ValueError("MSI must be finite, nonnegative and shaped (n_spots, n_ions)")
            self._check_query(a)
            if self.extra_key:
                size = np.asarray(a.obsm[self.extra_key]).shape[1]
                if extra_size is not None and size != extra_size:
                    raise ValueError("extra modality feature dimensions must match across sections")
                extra_size = size
        if not set.intersection(*[set(a.var_names) for a in adatas]):
            raise ValueError("training sections must share at least one gene")
        if sum(a.n_obs for a in adatas) < 2:
            raise ValueError("training requires at least two spots")
        self.mz_ = axis
        self.conf_width_ = None
        self.train_predictability_ = None
        self.transfer_predictability_ = None
        self.uncertainty_ = None
        self.extra_train_mean_ = None
        self.extra_train_sd_ = None
        self._extra_train_n_ = 0
        self.extra_n_features_ = extra_size
        # memory-lean: never hold all dense gene matrices at once (robust on laptops & big cohorts)
        self.genes_ = sorted(set.intersection(*[set(a.var_names) for a in adatas]))
        self._gpos = {g: i for i, g in enumerate(self.genes_)}
        G = len(self.genes_)
        n_hvg = min(self.n_hvg if self.n_hvg else G, G)
        def gm(a):
            v = a[:, self.genes_].layers["log1p"] if "log1p" in a.layers else a[:, self.genes_].X
            x = _dense(v)
            if not np.isfinite(x).all():
                raise ValueError("gene expression must be finite")
            return x
        # pass 1 — pooled per-gene mean/variance to choose HVGs (free each matrix immediately)
        s1 = np.zeros(G); s2 = np.zeros(G); N = 0
        for a in adatas:
            X = gm(a); s1 += X.sum(0); s2 += (X * X).sum(0); N += X.shape[0]; del X
        var = s2 / N - (s1 / N) ** 2
        self.gene_mean_ = (s1 / N).astype(np.float32)   # training per-gene mean (full gene set); imputes query-absent genes
        self.hvg_ = np.argsort(var)[-n_hvg:]
        self.hvg_genes_ = [self.genes_[i] for i in self.hvg_]
        # pass 2 — PCA on HVG columns only
        if getattr(self, "gene_scaling", None) == "section":
            Xh_all = np.vstack([self._section_scale(gm(a)[:, self.hvg_], None) for a in adatas])
        else:
            Xh_all = np.vstack([gm(a)[:, self.hvg_] for a in adatas])
        self.pca_ = PCA(min(self.n_pcs, Xh_all.shape[1], Xh_all.shape[0]),
                        svd_solver="randomized", random_state=0).fit(Xh_all)
        self.Vt_ = self.pca_.components_.astype(np.float32); self.pmean_ = self.pca_.mean_.astype(np.float32)
        del Xh_all
        # pass 3 — diffusion features (per-section graph) + metabolite targets
        F = []; Y = []; DET = []; COORDS = []
        for a in adatas:
            X = gm(a)
            ex = np.asarray(a.obsm[self.extra_key]) if (self.extra_key and self.extra_key in a.obsm) else None
            if ex is not None:
                ex = np.asarray(ex, np.float32)
            F.append(self._features(X, self.genes_, np.asarray(a.obsm["spatial"], float), None, extra=ex)); del X
            raw = np.asarray(a.uns[metab_key], float)
            Y.append(np.log1p(raw)); DET.append((raw > 0).mean(0) > 0.2)
            COORDS.append(np.asarray(a.obsm["spatial"], float))
            if ex is not None:
                if self.extra_train_mean_ is None:
                    self.extra_train_mean_ = np.zeros(ex.shape[1], np.float64)
                    self.extra_train_sd_ = np.zeros(ex.shape[1], np.float64)
                    self._extra_train_n_ = 0
                self.extra_train_mean_ += ex.sum(0)
                self.extra_train_sd_ += (ex * ex).sum(0)
                self._extra_train_n_ += ex.shape[0]
        Fs, Ys = F, [y.astype(np.float32) for y in Y]
        F = np.vstack(Fs); Y = np.vstack(Ys).astype(np.float32)
        if self.extra_train_mean_ is not None:
            n_extra = max(int(self._extra_train_n_), 1)
            mu_extra = self.extra_train_mean_ / n_extra
            var_extra = self.extra_train_sd_ / n_extra - mu_extra ** 2
            self.extra_train_mean_ = mu_extra.astype(np.float32)
            self.extra_train_sd_ = (np.sqrt(np.maximum(var_extra, 0.0)) + 1e-8).astype(np.float32)
        self.mu_ = Y.mean(0).astype(np.float32); self.sd_ = (Y.std(0) + 1e-8).astype(np.float32)
        if getattr(self, "target_transform", None) == "rank":
            # rank-Gaussian targets within each training section (rank harmonisation); keep each ion's
            # pooled training quantiles to map predictions back to log1p intensity
            Ts = [_rank_gauss(y) for y in Ys]
            self.target_q_levels_ = np.linspace(0.0, 1.0, 201)
            self.target_quantiles_ = np.quantile(Y, self.target_q_levels_, axis=0).astype(np.float32)
        else:
            # target_transform=None: per-metabolite standardisation of log1p intensities
            Ts = [(y - self.mu_) / self.sd_ for y in Ys]
        self.alpha_, self.alpha_cv_scores_, self.alpha_selection_ = self._choose_alpha(Fs, Ts, DET, COORDS)
        T = np.vstack(Ts)
        self.reg_ = Ridge(alpha=self.alpha_).fit(F, T)
        self.trees_ = None
        if getattr(self, "estimator", "ridge") == "ensemble":
            # ExtraTrees on the leading principal components of the (standardised) training targets,
            # averaged with the ridge prediction on the same target scale; stored in compact array form
            from sklearn.ensemble import ExtraTreesRegressor
            self.t_mu_ = T.mean(0).astype(np.float32); self.t_sd_ = (T.std(0) + 1e-8).astype(np.float32)
            Tz = (T - self.t_mu_) / self.t_sd_
            self.tpca_ = PCA(min(self.n_target_pcs, Tz.shape[1], Tz.shape[0] - 1), svd_solver="randomized",
                             random_state=0).fit(Tz)
            et = ExtraTreesRegressor(n_estimators=self.n_trees, min_samples_leaf=self.tree_min_leaf,
                                     max_features=0.33, n_jobs=getattr(self, "n_jobs", 1),
                                     random_state=self.random_state).fit(F, self.tpca_.transform(Tz).astype(np.float32))
            self.trees_ = _CompactForest(et)
            del et
            # equal-spread weights for the two halves, from their predictions on the training spots
            self.ens_w_r_, self.ens_w_e_, self.ens_m_r_, self.ens_m_e_ = _equal_weights(
                self.reg_.predict(F), self._tree_scores(F))
        self.last_gene_overlap_ = 1.0   # trained on its own gene panel
        return self

    @staticmethod
    def _section_scale(Xh, groups):
        """z-score each gene within each section (or within each `groups` label of a multi-section query);
        genes that are constant within a section, including query-absent genes, become exactly zero."""
        Xh = np.asarray(Xh, np.float32)
        out = np.empty_like(Xh)
        labels = np.zeros(len(Xh)) if groups is None else np.asarray(groups)
        for g in np.unique(labels):
            i = labels == g
            mu = Xh[i].mean(0); sd = Xh[i].std(0)
            out[i] = np.where(sd > 1e-6, (Xh[i] - mu) / np.maximum(sd, 1e-6), 0.0)
        return out

    def _tree_scores(self, F):
        """ExtraTrees half on the training-target scale (inverse of the target PCA and standardisation)."""
        tz = self.tpca_.inverse_transform(self.trees_.predict(F).astype(np.float32))
        return (tz * self.t_sd_ + self.t_mu_).astype(np.float32)

    def _raw_predict(self, F):
        """Model output on the training-target scale: the ridge, or the equal-spread average of the ridge
        and ExtraTrees halves (see _equal_weights)."""
        pred = self.reg_.predict(F).astype(np.float32)
        if getattr(self, "trees_", None) is not None:
            e = self._tree_scores(F)
            pred = (0.5 * (self.ens_m_r_ + self.ens_m_e_) + self.ens_w_r_ * (pred - self.ens_m_r_)
                    + self.ens_w_e_ * (e - self.ens_m_e_)).astype(np.float32)
        return pred

    def _choose_alpha(self, Fs, Ts, DET, COORDS):
        """Fixed alpha, or alpha="auto": leave-one-section-out CV over the training sections (each held-out
        section scored on its own), or, when a single section is given, leave-one-spatial-block-out CV over
        five k-means blocks with the held-out blocks pooled into one map before scoring (see select_alpha)."""
        if not isinstance(self.alpha, str):
            return float(self.alpha), None, "fixed"
        grid = ALPHA_GRID if getattr(self, "alpha_grid", None) is None else np.asarray(self.alpha_grid, float)
        units = []
        if len(Fs) >= 2:
            units = [(f, t, d) for f, t, d in zip(Fs, Ts, DET) if len(f) >= 10]
            how = "leave-one-section-out"
        elif len(Fs[0]) >= 50:
            from sklearn.cluster import KMeans
            blk = KMeans(5, n_init=3, random_state=0).fit_predict(COORDS[0])
            units = [(Fs[0][blk == b], Ts[0][blk == b], DET[0]) for b in range(5) if (blk == b).sum() >= 10]
            how = "leave-one-spatial-block-out"
        if len(units) < 2:
            return 200.0, None, "fallback (too few spots for inner CV)"
        pooled = how == "leave-one-spatial-block-out"
        if pooled:
            how += " (pooled)"
        alpha, scores = select_alpha(units, grid, pooled=pooled)
        if alpha is None:
            return 200.0, scores, "fallback (no finite CV score)"
        if alpha >= float(np.max(grid)):
            import warnings
            warnings.warn(f"the cross-validated ridge penalty ({alpha:g}) is the largest value of the grid; a wider "
                          "alpha_grid shows whether larger penalties fit better (reproduce/penalty_grid_check.py)",
                          RuntimeWarning, stacklevel=3)
        return alpha, scores, how

    def _from_rank(self, Z):
        """Map rank-scale (Gaussian score) predictions back to log1p intensity through each ion's pooled
        training quantiles. The map is monotone per ion, so predicted spatial ordering is unchanged; a
        1e-4 tie-breaker keeps it strictly increasing where the training distribution is flat (zeros)."""
        from scipy.special import ndtr
        P = ndtr(np.asarray(Z, np.float64))
        lv = self.target_q_levels_; Q = self.target_quantiles_
        out = np.empty(P.shape, np.float32)
        for j in range(P.shape[1]):
            out[:, j] = np.interp(P[:, j], lv, Q[:, j]) + 1e-4 * P[:, j]
        return out

    def estimate_conformal(self, adatas, metab_key="msi", q=0.90, mz_key="mz_features",
                           mz_atol=MZ_ATOL):
        """Empirical refit residual widths, NOT guaranteed split-conformal intervals.

        Refits on all-but-last section, predicts the held-out section, and stores the per-metabolite
        `q`-quantile of absolute residuals in self.conf_width_ (log1p-intensity units). Requires >=2
        sections. Call after fit(); leaves the full-data model in self.reg_ untouched.

        The calibration sections must sit on this model's own m/z axis: conf_width_ is indexed by
        self.mz_, so widths measured on another axis would be attached to the wrong ions. That is
        checked before any refitting, and a mismatch raises ValueError leaving conf_width_ and
        transfer_predictability_ untouched. The held-out section's per-ion Spearman correlation is stored as
        transfer_predictability_ (accuracy on a section the refit never saw).

        Hyper-parameters are read with getattr defaults, so models pickled without optional attributes
        (kegg_gmt_, extra_key, use_kegg) stay usable.
        Use calibrate() on independent data to calibrate the actual fitted predictor.
        """
        if not 0 < q < 1:
            raise ValueError("q must be strictly between 0 and 1")
        if not hasattr(self, "reg_"):
            raise ValueError("fit the model before estimating uncertainty")
        if len(adatas) < 2:
            raise ValueError("empirical refit uncertainty needs at least two sections")
        # the calibration cohort must agree with itself AND with the fitted model's ion axis
        cal_axis = common_mz_axis(adatas, mz_key, atol=mz_atol)
        own = getattr(self, "mz_", None)
        if own is not None and len(np.asarray(own, float)):
            own = np.asarray(own, float).ravel()
            if cal_axis.shape != own.shape or not np.allclose(cal_axis, own, rtol=0.0,
                                                              atol=mz_atol, equal_nan=True):
                raise ValueError(
                    f"calibration m/z axis does not match the fitted model: model has {own.size} "
                    f"ions ({own.min():.4f}-{own.max():.4f} Da), calibration sections have "
                    f"{cal_axis.size} ({cal_axis.min():.4f}-{cal_axis.max():.4f} Da). conf_width_ "
                    f"is indexed by the model's own axis, so calibrating here would attach each "
                    f"width to the wrong ion — calibrate on sections from the model's own cohort. "
                    f"conf_width_ and transfer_predictability_ are left unchanged.")
        # attribute lookup with defaults: a pickled model may lack kegg_gmt_/extra_key/use_kegg
        cal = MetaboCast(getattr(self, "n_hvg", 3000), getattr(self, "n_pcs", 100),
                         getattr(self, "alpha", 200.0), getattr(self, "k", 6),
                         use_kegg=False, kegg_gmt=None,
                         extra_key=getattr(self, "extra_key", None),
                         target_transform=getattr(self, "target_transform", None),
                         alpha_grid=getattr(self, "alpha_grid", None),
                         estimator=getattr(self, "estimator", "ridge"),
                         n_trees=getattr(self, "n_trees", 100), tree_min_leaf=getattr(self, "tree_min_leaf", 10),
                         n_target_pcs=getattr(self, "n_target_pcs", 50), random_state=getattr(self, "random_state", 0),
                         gene_scaling=getattr(self, "gene_scaling", None), n_jobs=getattr(self, "n_jobs", 1))
        cal.kegg = dict(getattr(self, "kegg", None) or {})   # reuse the exact same gene sets
        cal.use_kegg = bool(getattr(self, "use_kegg", False))
        cal.kegg_gmt_ = getattr(self, "kegg_gmt_", None)
        cal.fit(adatas[:-1], metab_key=metab_key, mz_key=mz_key, mz_atol=mz_atol)
        held = adatas[-1]
        pred = cal.predict_metabolome(held.copy())
        true = np.log1p(np.asarray(held.uns[metab_key], float))
        self.conf_width_ = np.quantile(np.abs(true - pred), q, axis=0).astype(np.float32)
        self.uncertainty_ = {"method": "empirical_auxiliary_refit", "coverage_requested": q,
                             "unit": "spot", "n_units": held.n_obs,
                             "coverage_guaranteed": False}
        # per-ion accuracy on the held-out section -> transfer evidence for the tiers of reliability_table()
        try:
            from scipy.stats import spearmanr
            tp = np.full(true.shape[1], np.nan, np.float32)
            for k in range(true.shape[1]):
                if true[:, k].std() > 1e-9 and pred[:, k].std() > 1e-9:
                    tp[k] = spearmanr(true[:, k], pred[:, k])[0]
            self.transfer_predictability_ = tp
        except Exception:
            self.transfer_predictability_ = None
        return self

    def calibrate(self, adatas, q=0.90, unit="section", metab_key="msi",
                  mz_key="mz_features", mz_atol=MZ_ATOL):
        """Calibrate the unchanged fitted model on independent held-out data.

        Never include these sections (or their donors) in fitting/tuning/registration
        selection. Section mode uses the maximum absolute residual per ion within
        each section; sections must represent independent, exchangeable subjects.
        Spot mode pools residuals and requires exchangeable spots, generally NOT
        justified for spatial tissue. Small calibration sets can yield infinite
        widths. No distribution-free coverage under arbitrary cohort shift is claimed.
        Each ion's Spearman correlation on every calibration section, averaged over sections, is stored as
        transfer_predictability_ (these sections are unseen by the fit), which reliability_table() uses for
        the transfer-supported tier.
        """
        from scipy.stats import spearmanr
        from .validation import conformal_quantile
        if not hasattr(self, "reg_"):
            raise ValueError("fit the model before calibration")
        if unit not in {"section", "spot"} or not 0 < q < 1:
            raise ValueError("unit must be 'section' or 'spot' and 0 < q < 1")
        axis = common_mz_axis(adatas, mz_key, atol=mz_atol)
        if axis.shape != self.mz_.shape or not np.allclose(axis, self.mz_, rtol=0, atol=mz_atol):
            raise ValueError("calibration m/z axis does not match the fitted model")
        scores, per_section = [], []
        old_overlap = self.last_gene_overlap_
        try:
            for a in adatas:
                raw = np.asarray(a.uns[metab_key], float)
                if raw.shape != (a.n_obs, len(axis)) or not np.isfinite(raw).all() or (raw < 0).any():
                    raise ValueError("calibration MSI must be finite, nonnegative and aligned")
                true = np.log1p(raw)
                pred = self.predict_metabolome(a.copy())
                residual = np.abs(true - pred)
                scores.append(residual.max(0, keepdims=True) if unit == "section" else residual)
                r = np.full(len(axis), np.nan)
                for j in range(len(axis)):
                    if true[:, j].std() > 1e-9 and pred[:, j].std() > 1e-9:
                        r[j] = spearmanr(true[:, j], pred[:, j])[0]
                per_section.append(r)
        finally:
            self.last_gene_overlap_ = old_overlap
        scores = np.vstack(scores)
        widths = conformal_quantile(scores, q).astype(np.float32)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN ions stay NaN (uncalibrated)
            tp = np.nanmean(np.vstack(per_section), 0).astype(np.float32)
        self.conf_width_, self.transfer_predictability_ = widths, tp
        self.uncertainty_ = {"method": "heldout_fixed_model", "coverage_requested": q,
                             "unit": unit, "n_units": len(scores),
                             "coverage_guaranteed": False,
                             "assumption": "independent exchangeable calibration and query units"}
        return self

    def _check_query(self, adata):
        if not adata.n_obs or not adata.var_names.is_unique:
            raise ValueError("need nonempty spots and unique gene names")
        coords = np.asarray(adata.obsm["spatial"], float)
        if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2 or not np.isfinite(coords).all():
            raise ValueError("spatial coordinates must be a finite spots-by-dimensions matrix")
        key = getattr(self, "extra_key", None)
        if key:
            if key not in adata.obsm:
                raise ValueError(f"required extra modality adata.obsm[{key!r}] is missing")
            extra = np.asarray(adata.obsm[key])
            if extra.ndim != 2 or not extra.shape[1] or not np.isfinite(extra).all():
                raise ValueError("extra modality must be a nonempty finite matrix")

    def predict_metabolome(self, adata, metab_uns_key="metabocast_pred", section_key=None):
        if not hasattr(self, "reg_"):
            raise ValueError("fit or load a model before prediction")
        self._check_query(adata)
        # align genes to the training set; impute genes absent from the query at the TRAINING per-gene mean
        # so absent genes contribute exactly zero after PCA centring (X - pmean_ -> 0 on those columns)
        # and do not bias the KEGG pathway z-scores
        pos = {g: i for i, g in enumerate(list(adata.var_names))}
        present_genes = [g for g in self.genes_ if g in pos]
        if not present_genes:
            raise ValueError("query has no genes in common with the training panel")
        self.last_gene_overlap_ = len(present_genes) / max(len(self.genes_), 1)
        hv = getattr(self, "hvg_genes_", None)
        self.last_hvg_overlap_ = (float(np.mean([g in pos for g in hv])) if hv is not None and len(hv)
                                  else float("nan"))
        groups = None
        if section_key is not None:
            if section_key not in adata.obs or adata.obs[section_key].isna().any():
                raise ValueError("section_key must name an observation column without missing values")
            groups = np.asarray(adata.obs[section_key].values)
        key = getattr(self, "extra_key", None)
        ex = np.asarray(adata.obsm[key]) if key else None
        if ex is not None and getattr(self, "extra_n_features_", None) not in (None, ex.shape[1]):
            raise ValueError("query extra modality must match the fitted feature dimension")
        lean = (getattr(self, "_lean_predict", True) and not (getattr(self, "use_kegg", False) and getattr(self, "kegg", None))
                and getattr(self, "gene_scaling", None) != "section" and getattr(self, "hvg_genes_", None) is not None)
        if lean:          # default path: PCA scores from the HVG columns in row chunks (memory-lean for large queries)
            F = self._diffusion_features(self._query_pca_scores(adata), np.asarray(adata.obsm["spatial"], float), groups, extra=ex)
        else:
            F = self._full_query_features(adata, present_genes, pos, groups, ex)
        return self._finish_prediction(adata, F, metab_uns_key)

    def _full_query_features(self, adata, present_genes, pos, groups, ex):
        """Features from the whole gene panel (needed for the KEGG prior and per-section gene scaling)."""
        v = adata[:, present_genes]
        Xsub = _dense(v.layers["log1p"] if "log1p" in v.layers else v.X)
        if not np.isfinite(Xsub).all():
            raise ValueError("query gene expression must be finite")
        if getattr(self, "gene_mean_", None) is not None:
            X = np.tile(self.gene_mean_, (adata.n_obs, 1)).astype(np.float32)
        else:                                                # models pickled before mean-imputation: fall back to zero-fill
            X = np.zeros((adata.n_obs, len(self.genes_)), np.float32)
        present = [k for k, g in enumerate(self.genes_) if g in pos]
        X[:, present] = Xsub
        # section-aware neighbour graph: keep neighbours within each section for multi-section queries
        return self._features(X, self.genes_, np.asarray(adata.obsm["spatial"], float), groups, extra=ex)

    def _finish_prediction(self, adata, F, metab_uns_key):
        pred = self._raw_predict(F)
        adata.obsm.pop(metab_uns_key + "_rank", None)
        if getattr(self, "target_transform", None) == "rank":
            adata.obsm[metab_uns_key + "_rank"] = pred   # predicted within-section rank (Gaussian) scores
            pred = self._from_rank(pred)
        elif getattr(self, "mu_", None) is not None and getattr(self, "sd_", None) is not None:
            pred = pred * self.sd_ + self.mu_          # invert per-metabolite standardization -> log1p intensity
        pred = np.maximum(pred, 0.0).astype(np.float32)  # log1p MSI intensities are non-negative
        adata.obsm[metab_uns_key] = pred
        for stale in ("metabocast_conf_width", "metabocast_conf_width_adj", "metabocast_uncertainty",
                      "metabocast_reliability", "metabocast_cell_state_summary"):
            adata.uns.pop(stale, None)
        if getattr(self, "conf_width_", None) is not None:
            cw = np.asarray(self.conf_width_, np.float32)
            adata.uns["metabocast_conf_width"] = cw
            adata.uns["metabocast_uncertainty"] = dict(getattr(self, "uncertainty_", None) or
                                                       {"method": "unspecified", "coverage_guaranteed": False})
            # Overlap-adaptive widening is a heuristic, not Mondrian conformal calibration.
            # When the query shares fewer genes
            # with the training panel the prediction is more OOD, so widen the interval proportionally.
            # factor = 1 + beta*(1 - overlap); beta default 1.0. Reduces to the base width at full overlap.
            ov = float(getattr(self, "last_gene_overlap_", 1.0))
            if np.isfinite(ov):
                beta = float(getattr(self, "conf_overlap_beta_", 1.0))
                if not np.isfinite(beta) or beta < 0:
                    raise ValueError("conf_overlap_beta_ must be finite and nonnegative")
                adata.uns["metabocast_conf_width_adj"] = (cw * (1.0 + beta * (1.0 - ov))).astype(np.float32)
        return pred

    # --------------------------------------------------------------------- #
    #  Per-ion reliability export + putative annotation (default user output)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _annotate_mz(mz, tol=0.01):
        """Putative m/z -> (names, chemical class) from the curated reference of metabocast.annotation (the one the
        manuscript uses), with the negative-mode adducts [M-H]-, [M+Cl]- and [M+HCOO]-. Exact-mass candidates only:
        every candidate within `tol` is kept, nearest first; isomers that share a formula carry one group name; the
        class is 'ambiguous' when the candidates span more than one class and '' without a candidate."""
        from .annotation import annotate
        a = annotate(np.asarray(mz, float), tol=tol)
        names = a["annotation"].fillna("").astype(str).to_numpy(dtype=object)
        klass = a["class"].where(a["n_candidates"] > 0, "").fillna("").astype(str).to_numpy(dtype=object)
        return names, klass

    @staticmethod
    def _cell_state_summary(adata, key):
        if adata is None or key is None or key not in adata.obsm:
            return {}
        C = np.asarray(adata.obsm[key], np.float32)
        if C.ndim != 2 or C.shape[1] == 0:
            return {}
        names = list(adata.uns.get(key + "_names", [f"{key}_{i}" for i in range(C.shape[1])]))
        # Marker-signature scores can be negative; shift to pseudo-proportions for entropy/purity only.
        P = C - np.nanmin(C, axis=1, keepdims=True) + 1e-6
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-8)
        purity = P.max(1)
        entropy = -(P * np.log(P)).sum(1) / np.log(max(P.shape[1], 2))
        dom_i = int(np.argmax(P.mean(0)))
        return {
            "celltype_key": key,
            "dominant_celltype": names[dom_i] if dom_i < len(names) else str(dom_i),
            "mean_celltype_purity": float(np.nanmean(purity)),
            "mean_celltype_entropy": float(np.nanmean(entropy)),
        }

    @staticmethod
    def _focus_mask(df, focus):
        if focus is None:
            return np.ones(len(df), dtype=bool)
        if focus != "lipid_redox":
            raise ValueError("focus must be None or 'lipid_redox'")
        lipid = {"fatty acid", "phospholipid", "lysophospholipid", "sphingolipid", "glycerolipid", "sterol",
                 "lipid headgroup", "acylcarnitine"}
        nucleotide_redox = {"nucleotide", "nucleoside", "nucleobase", "peptide/redox", "cofactor"}
        redox_names = {"Glutathione", "GSSG", "Ascorbate", "Urate", "Glutamate"}
        klass = df["chemical_class"].fillna("").astype(str)      # pandas >= 3 keeps NaN under astype(str)
        first = df["annotation"].fillna("").astype(str).str.split(";").str[0].str.strip()
        return (klass.isin(lipid | nucleotide_redox).to_numpy(dtype=bool)
                | first.isin(redox_names).to_numpy(dtype=bool))

    def reliability_table(self, adata=None, annotate=True, focus=None, celltype_key=None, within_min=0.30, transfer_min=0.20):
        """Per-ion reliability report, the recommended user-facing output. Returns a DataFrame with, for
        every predicted m/z channel: predicted mean and s.d. (when `adata` with predictions is given), the
        uncertainty width, held-out predictability within sections (train_predictability_spearman) and on
        unseen sections (transfer_predictability_spearman) where available, putative annotation and
        chemical class, and a reliability tier (column trust_tier). Also stored in
        `adata.uns['metabocast_reliability']` when `adata` is provided.

        Tiers separate accuracy on unseen sections from accuracy inside the training sections, because
        within-section scores are raised by spatial autocorrelation and need not carry over to new tissue:
          'transfer-supported'   held-out-section Spearman >= transfer_min (default 0.20)
          'within-section only'  within-section Spearman >= within_min (default 0.30), transfer below
                                 transfer_min or not measured
          'low'                  measured, but below both thresholds
          'uncalibrated'         no held-out predictability for the ion. The shipped model carries both scores;
                                 for a retrained model, calibrate() and estimate_conformal() add the score on
                                 unseen sections, and train_predictability_ can be set from spatial-block
                                 cross-validation (reproduce/within_benchmark.py). Ions are never ranked by
                                 uncertainty width alone."""
        import pandas as pd
        mz = np.asarray(getattr(self, "mz_", []), float)
        n = len(mz)
        cw = np.asarray(self.conf_width_, float) if getattr(self, "conf_width_", None) is not None else np.full(n, np.nan)
        tp = np.asarray(self.train_predictability_, float) if getattr(self, "train_predictability_", None) is not None else np.full(n, np.nan)
        pmean = pstd = np.full(n, np.nan)
        if adata is not None and "metabocast_pred" in adata.obsm:
            P = np.asarray(adata.obsm["metabocast_pred"], float); pmean = P.mean(0); pstd = P.std(0)
        names, klass = (self._annotate_mz(mz) if annotate else (np.array([""] * n, object), np.array([""] * n, object)))
        # Trust tier comes from held-out per-ion predictability ONLY. Without it the model is
        # uncalibrated: uncertainty widths rank ions by residual spread on an arbitrary intensity
        # scale, so splitting them into terciles would invent a reliability ordering the data
        # never supported and would always label a third of the ions "high". Report
        # "uncalibrated" instead and leave conf_width in the table for the user to weigh.
        tr_p = (np.asarray(self.transfer_predictability_, float)
                if getattr(self, "transfer_predictability_", None) is not None else np.full(n, np.nan))

        def tier(i):
            w, t = tp[i], tr_p[i]
            if np.isfinite(t) and t >= transfer_min:
                return "transfer-supported"
            if np.isfinite(w) and w >= within_min:
                return "within-section only"
            if np.isfinite(w) or np.isfinite(t):
                return "low"
            return "uncalibrated"
        df = pd.DataFrame({
            "mz": np.round(mz, 4),
            "pred_mean": np.round(pmean, 4),
            "pred_std": np.round(pstd, 4),
            "conf_width": np.round(cw, 4),
            "train_predictability_spearman": np.round(tp, 4),
            "transfer_predictability_spearman": np.round(tr_p, 4),
            "annotation": names,
            "chemical_class": klass,
            "trust_tier": [tier(i) for i in range(n)],
        })
        uncertainty = getattr(self, "uncertainty_", None) or {}
        df['uncertainty_method'] = uncertainty.get('method', 'unspecified' if np.isfinite(cw).any() else 'none')
        df['calibration_unit'] = uncertainty.get('unit', 'unknown')
        if adata is not None:
            ckey = celltype_key
            extra_key = getattr(self, 'extra_key', None)
            if ckey is None and extra_key and extra_key in adata.obsm:
                ckey = extra_key
            if ckey is None and "celltype" in adata.obsm:
                ckey = "celltype"
            cell_summary = self._cell_state_summary(adata, ckey)
            if cell_summary:
                if (ckey == getattr(self, "extra_key", None) and getattr(self, "extra_train_mean_", None) is not None
                        and getattr(self, "extra_train_sd_", None) is not None):
                    C = np.asarray(adata.obsm[ckey], np.float32)
                    z = np.abs((C - self.extra_train_mean_) / self.extra_train_sd_)
                    cell_summary["celltype_ood_fraction_z_gt_3"] = float(np.mean(z > 3.0))
                    cell_summary["celltype_mean_abs_z"] = float(np.mean(z))
                for key, value in cell_summary.items():
                    df[key] = value
                if (cell_summary.get("mean_celltype_entropy", 0.0) >= 0.75
                        or cell_summary.get("celltype_ood_fraction_z_gt_3", 0.0) >= 0.10):
                    df["cell_state_context"] = "mixed_or_ood"
                else:
                    df["cell_state_context"] = "no_flag_from_heuristic"
                adata.uns["metabocast_cell_state_summary"] = cell_summary
            df = df.loc[self._focus_mask(df, focus)].reset_index(drop=True)
            adata.uns["metabocast_reliability"] = df
        else:
            df = df.loc[self._focus_mask(df, focus)].reset_index(drop=True)
        return df

    def save(self, path, bundle=True):
        """Pickle the model to `path`. By default writes a self-describing bundle dict
        {model, mz, genes, conf_width, version, ...}; reload with MetaboCast.load()."""
        import pickle
        obj = self
        if bundle:
            obj = dict(model=self,
                       mz=np.asarray(getattr(self, "mz_", []), float),
                       genes=list(getattr(self, "genes_", [])),
                       conf_width=(None if getattr(self, "conf_width_", None) is None
                                   else np.asarray(self.conf_width_, np.float32)),
                       n_hvg=self.n_hvg,
                       target_transform=getattr(self, "target_transform", None),
                       estimator=getattr(self, "estimator", "ridge"),
                       alpha=getattr(self, "alpha_", getattr(self, "alpha", None)),
                       version=f"metabocast-{_package_version()}")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def _reset_query_state(m):
        """Clear per-query state a pickle carries over from whoever saved it.

        last_gene_overlap_ and last_hvg_overlap_ describe the LAST predict_metabolome call. A value
        stored at save time would otherwise be read as if it described the caller's query, and would
        widen the overlap-adaptive width for a query that has not been predicted yet. predict_metabolome
        sets both on every call, so NaN is the honest value until then.
        """
        m.last_gene_overlap_ = float("nan")
        m.last_hvg_overlap_ = float("nan")
        return m

    @classmethod
    def load(cls, path):
        """Load a model saved with .save() OR the shipped bundle dict. Always returns a MetaboCast
        instance (unwrapping the {model, ...} bundle and exposing the extra keys as .bundle_meta_).
        Stale per-query state from the saving session is reset (see _reset_query_state)."""
        import pickle
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with open(path, "rb") as f:
                obj = pickle.load(f)
        # scikit-learn warns once per pickled estimator when the installed version differs from the one that
        # saved the model; report that once, and pass every other warning through unchanged
        saved = sorted({str(getattr(w.message, "original_sklearn_version", "")) for w in caught
                        if type(w.message).__name__ == "InconsistentVersionWarning"} - {""})
        for w in caught:
            if type(w.message).__name__ != "InconsistentVersionWarning":
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        if saved:
            import sklearn
            warnings.warn(f"{os.path.basename(str(path))} was saved with scikit-learn {', '.join(saved)} and is being "
                          f"loaded with scikit-learn {sklearn.__version__}. MetaboCast uses only the fitted arrays of "
                          "these estimators; install the versions in requirements-lock.txt to reproduce the saved "
                          "environment exactly.", UserWarning, stacklevel=2)
        if isinstance(obj, cls):
            return cls._reset_query_state(obj)
        if isinstance(obj, dict) and isinstance(obj.get("model"), cls):
            m = obj["model"]
            m.bundle_meta_ = {k: v for k, v in obj.items() if k != "model"}
            # surface a bundled uncertainty width onto the model so predict_metabolome attaches it
            if getattr(m, "conf_width_", None) is None and obj.get("conf_width") is not None:
                m.conf_width_ = np.asarray(obj["conf_width"], np.float32)
            return cls._reset_query_state(m)
        return obj   # last resort: hand back whatever was pickled


if __name__ == "__main__":
    # portable self-test: pass a folder containing MetaboCast-format .h5ad, e.g.
    #   python metabocast.py /path/to/sections
    import anndata as ad, glob, sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    h5 = [p for p in sorted(glob.glob(os.path.join(root, "**", "*.h5ad"), recursive=True))
          if "training_ready" not in p]
    if len(h5) < 2:
        print("Point me at a folder with >=2 MetaboCast-format .h5ad files."); sys.exit(0)
    tr = [ad.read_h5ad(f) for f in h5[:-1]]; te = ad.read_h5ad(h5[-1])
    m = MetaboCast(use_kegg=False).fit(tr)
    p = m.predict_metabolome(te)
    print(f"demo: trained on {len(tr)} sections; predicted {p.shape} metabolites for {os.path.basename(h5[-1])}")
