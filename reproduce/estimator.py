"""The MetaboCast estimator applied to precomputed feature matrices.

The reproduction scripts build their own features (fold-specific PCA, transductive graphs, ablated graphs
...), so they cannot call MetaboCast.fit directly. Estimator performs exactly the steps of MetaboCast.fit /
predict_metabolome on a given feature matrix (verified against the package to float32 precision):
rank-Gaussian targets within each training section, ridge penalty chosen by leave-one-unit-out CV over
training sections (each scored on its own) or over the spatial blocks of one section (held-out blocks
pooled before scoring), ExtraTrees on 50 target principal components averaged with the ridge on the rank
scale, and the monotone back-transform to log1p intensity through the pooled training quantiles.
"""
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metabocast.metabocast import _rank_gauss, _CompactForest, _equal_weights, select_alpha, ALPHA_GRID   # noqa: E402



class Estimator:
    """MetaboCast estimator on precomputed features (defaults identical to MetaboCast())."""

    def __init__(self, estimator="ensemble", n_trees=100, min_leaf=10, n_target_pcs=50, seed=0):
        self.estimator, self.n_trees, self.min_leaf = estimator, n_trees, min_leaf
        self.n_target_pcs, self.seed = n_target_pcs, seed

    def fit(self, F, y, groups, sections=None):
        """groups: inner-CV unit per training spot (training sections, or spatial blocks of one section).
        sections: section label per training spot for the rank transform (default: one section)."""
        F = np.asarray(F, np.float32); y = np.asarray(y, np.float32); groups = np.asarray(groups)
        if sections is None:
            T = _rank_gauss(y)
        else:
            sections = np.asarray(sections); T = np.empty_like(y)
            for s in np.unique(sections):
                T[sections == s] = _rank_gauss(y[sections == s])
        det = np.ones(y.shape[1], bool)
        units = [(F[groups == g], T[groups == g], det) for g in np.unique(groups) if (groups == g).sum() >= 10]
        one_section = sections is None or len(np.unique(sections)) < 2   # groups are spatial blocks: pooled scoring
        alpha = select_alpha(units, ALPHA_GRID, pooled=one_section)[0] if len(units) >= 2 else None
        self.alpha = 200.0 if alpha is None else alpha
        self.reg = Ridge(alpha=self.alpha).fit(F, T)
        self.cf = None
        if self.estimator == "ensemble":
            self.t_mu = T.mean(0).astype(np.float32); self.t_sd = (T.std(0) + 1e-8).astype(np.float32)
            Tz = (T - self.t_mu) / self.t_sd
            self.tpca = PCA(min(self.n_target_pcs, Tz.shape[1], Tz.shape[0] - 1), svd_solver="randomized",
                            random_state=0).fit(Tz)
            et = ExtraTreesRegressor(n_estimators=self.n_trees, min_samples_leaf=self.min_leaf, max_features=0.33,
                                     n_jobs=1, random_state=self.seed).fit(F, self.tpca.transform(Tz).astype(np.float32))
            self.cf = _CompactForest(et)
            self.w = _equal_weights(self.reg.predict(F), self._trees(F))
        self.lv = np.linspace(0.0, 1.0, 201)
        self.Q = np.quantile(y, self.lv, axis=0).astype(np.float32)
        return self

    def rank_scores(self, F, ridge_only=False):
        F = np.asarray(F, np.float32)
        z = self.reg.predict(F).astype(np.float32)
        if self.cf is not None and not ridge_only:      # equal-spread average, as MetaboCast._raw_predict
            w_r, w_e, m_r, m_e = self.w
            z = (0.5 * (m_r + m_e) + w_r * (z - m_r) + w_e * (self._trees(F) - m_e)).astype(np.float32)
        return z

    def _trees(self, F):
        tz = self.tpca.inverse_transform(self.cf.predict(F).astype(np.float32))
        return (tz * self.t_sd + self.t_mu).astype(np.float32)

    def to_log1p(self, Z):
        from scipy.special import ndtr
        P = ndtr(np.asarray(Z, np.float64))
        return np.column_stack([np.interp(P[:, j], self.lv, self.Q[:, j]) + 1e-4 * P[:, j] for j in range(P.shape[1])])

    def predict(self, F, ridge_only=False):
        return np.maximum(self.to_log1p(self.rank_scores(F, ridge_only)), 0.0)


def make(**kw):
    return Estimator(**kw)


def fit_predict(Ftr, ytr, Fte, groups, sections=None, **kw):
    """Fit the estimator on (Ftr, ytr) and predict Fte (log1p scale)."""
    return make(**kw).fit(Ftr, ytr, groups, sections).predict(Fte)
