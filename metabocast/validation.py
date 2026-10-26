"""Numerical helpers shared by the package and reproducible evaluations."""
import numpy as np


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjustment, retaining NaNs for untested hypotheses."""
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    if np.isinf(p).any() or ((p[finite] < 0) | (p[finite] > 1)).any():
        raise ValueError("P values must be in [0, 1] or NaN")
    q = np.full(p.shape, np.nan)
    values = p[finite]
    n = values.size
    if n:
        order = np.argsort(values, kind="stable")
        adjusted = values[order] * n / np.arange(1, n + 1)
        out = np.empty(n)
        out[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
        q[finite] = out
    return q


def conformal_quantile(scores, coverage):
    """Finite-sample order statistic; insufficient calibration yields infinite width."""
    scores = np.asarray(scores, dtype=float)
    if not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between 0 and 1")
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError("scores must be a nonempty units-by-ions matrix")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("scores must be finite and nonnegative")
    rank = int(np.ceil((len(scores) + 1) * coverage))
    if rank > len(scores):
        return np.full(scores.shape[1], np.inf)
    return np.partition(scores, rank - 1, axis=0)[rank - 1]


def single_gene_predict(x_train, y_train, x_test):
    """Select by absolute training correlation, then fit signed univariate OLS per ion."""
    x, y, test = (np.asarray(v, float) for v in (x_train, y_train, x_test))
    if x.ndim != 2 or y.ndim != 2 or test.ndim != 2 or len(x) != len(y):
        raise ValueError("expected aligned two-dimensional training and query matrices")
    if len(x) < 2 or x.shape[1] == 0 or test.shape[1] != x.shape[1]:
        raise ValueError("need >=2 training observations and matching nonempty gene axes")
    if not all(np.isfinite(v).all() for v in (x, y, test)):
        raise ValueError("baseline inputs must be finite")
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    ss = (xc * xc).sum(0)
    cov = xc.T @ yc
    denom = np.sqrt(ss[:, None] * (yc * yc).sum(0)[None, :])
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    best = np.abs(corr).argmax(0)
    slope = np.divide(cov[best, np.arange(y.shape[1])], ss[best],
                      out=np.zeros(y.shape[1]), where=ss[best] > 0)
    return ym + (test[:, best] - xm[best]) * slope


def subject_splits(subjects):
    """Leave one explicitly supplied subject out; never infer donors from filenames."""
    subjects = np.asarray(subjects, dtype=str)
    if subjects.ndim != 1 or len(set(subjects)) < 2 or np.any(subjects == ""):
        raise ValueError("need at least two nonempty subject IDs")
    for subject in dict.fromkeys(subjects):
        yield np.flatnonzero(subjects != subject), np.flatnonzero(subjects == subject)
