"""Within-section benchmark (Figs 2-4; benchmark, per_ion and sma_within tables; SuppFig_SMA_within).

Protocol: five k-means spatial blocks per section; HVGs (3,000) and PCA (100) fitted on the training blocks;
the spatial graph uses the whole section (transductive); the benchmark panel is the 1,000 most variable ions
detected in >20% of spots. Every detected ion is also scored, for the full-metabolome analyses.

MetaboCast is the released estimator applied to the fold's features (it mirrors MetaboCast.fit and
_raw_predict line for line): rank-Gaussian training targets, ridge penalty chosen by leave-one-block-out
CV on the training blocks only (held-out blocks pooled before scoring, as MetaboCast does for a single
section), ExtraTrees (100 trees, min leaf 10, 50 target PCs) averaged with the ridge on the rank scale
after scaling both halves to equal spread on the training spots, and the monotone back-transform to log1p
intensity through the training quantiles. Every regression baseline uses the same estimator, so
comparisons isolate the features:
  no diffusion      PCA features without the diffused copies (the diffusion ablation)
  KEGG pathway features + supervised regressor
                    KEGG pathway scores (+ one neighbour-diffused copy) instead of PCA/diffusion features
  ridge only        MetaboCast without the ExtraTrees half
Joint NMF (gene and metabolite matrices factorised together) and the signed single-gene regression are
fitted on the training blocks of the benchmark panel.

Scoring scale. Out-of-fold predictions from the five fold models are pooled before the per-ion Spearman is
computed. The primary score pools the intensity (log1p) predictions. Because each fold model maps back to
intensity through its own training spots, the script also reports the score with predictions pooled on each
model's own training-target scale (rank-Gaussian scores, the mean-scaled NMF reconstruction and the
standardised single-gene prediction), stored with the suffix " [target scale]".

Usage:
    python reproduce/within_benchmark.py --data <DESIUM_DIR> --out results/within_desium.json
    python reproduce/within_benchmark.py --data <SMA_DIR> --sections SMA_DHB_V11L12-038_A1 ... --out results/within_sma.json
Writes <out>.json (per-section medians and summaries), <out>_per_ion.npz (per-ion Spearman for every
detected ion, NaN where not scored), <out>_maps.npz (held-out maps of selected ions) and
<out>_conformal.npz (within-section interval widths and coverage).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metabocast.metabocast import _norm_adj, _rank_gauss, _CompactForest, _equal_weights, select_alpha, ALPHA_GRID   # noqa: E402
from metabocast.validation import single_gene_predict                                              # noqa: E402

DESIUM = ['BC_515_Section_1', 'BC_515_Section_2', 'BC_525', 'BC_823', 'LC_091', 'LC_170', 'LC_276']
METHODS = ['MetaboCast', 'MetaboCast + KEGG', 'MetaboCast (ridge only)',
           'KEGG pathway features + supervised regressor', 'no diffusion', 'joint-NMF', 'single-gene OLS']
PANEL_ONLY = ('joint-NMF', 'single-gene OLS')


def col_spearman(P, Y):
    R1 = rankdata(P, axis=0); R2 = rankdata(Y, axis=0)
    R1 -= R1.mean(0); R2 -= R2.mean(0)
    den = np.sqrt((R1 * R1).sum(0) * (R2 * R2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (R1 * R2).sum(0) / den
    r[~(den > 1e-12)] = np.nan
    return r


class SectionEstimator:
    """The released estimator on precomputed features of one section (same steps and settings as MetaboCast.fit)."""

    def __init__(self, estimator='ensemble', n_trees=100, min_leaf=10, n_target_pcs=50, seed=0):
        self.estimator, self.n_trees, self.min_leaf, self.n_target_pcs, self.seed = estimator, n_trees, min_leaf, n_target_pcs, seed

    def fit(self, F, y, blocks):
        T = _rank_gauss(y)
        det = np.ones(y.shape[1], bool)
        units = [(F[blocks == b], T[blocks == b], det) for b in np.unique(blocks) if (blocks == b).sum() >= 10]
        alpha = select_alpha(units, ALPHA_GRID, pooled=True)[0] if len(units) >= 2 else None   # one section: pooled blocks
        self.alpha = 200.0 if alpha is None else alpha
        self.reg = Ridge(alpha=self.alpha).fit(F, T)
        self.cf = None
        if self.estimator == 'ensemble':
            self.t_mu = T.mean(0).astype(np.float32); self.t_sd = (T.std(0) + 1e-8).astype(np.float32)
            Tz = (T - self.t_mu) / self.t_sd
            self.tpca = PCA(min(self.n_target_pcs, Tz.shape[1], Tz.shape[0] - 1), svd_solver='randomized',
                            random_state=0).fit(Tz)
            et = ExtraTreesRegressor(n_estimators=self.n_trees, min_samples_leaf=self.min_leaf, max_features=0.33,
                                     n_jobs=1, random_state=self.seed).fit(F, self.tpca.transform(Tz).astype(np.float32))
            self.cf = _CompactForest(et)
            self.w = _equal_weights(self.reg.predict(F), self._trees(F))
        self.lv = np.linspace(0.0, 1.0, 201)
        self.Q = np.quantile(y, self.lv, axis=0).astype(np.float32)
        return self

    def rank_scores(self, F, ridge_only=False):
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


def read_section(path):
    """Only what the benchmark needs, read with h5py (log1p expression as float32, in row chunks)."""
    import h5py
    import scipy.sparse as sp
    try:
        from anndata.io import read_elem
    except ImportError:                                   # anndata < 0.11
        from anndata.experimental import read_elem
    with h5py.File(path, 'r') as f:
        names = [str(v) for v in read_elem(f['var']).index]
        o = f['layers/log1p'] if 'log1p' in f['layers'] else f['X']
        if isinstance(o, h5py.Dataset):
            x = np.empty(o.shape, np.float32)
            for i in range(0, o.shape[0], 256):
                x[i:i + 256] = o[i:i + 256]
        else:
            enc = o.attrs['encoding-type']; shape = tuple(o.attrs['shape'])
            M = (sp.csr_matrix if 'csr' in enc else sp.csc_matrix)((o['data'][:], o['indices'][:], o['indptr'][:]), shape=shape)
            x = M.astype(np.float32).toarray()
        raw = np.asarray(f['uns/msi'][:], np.float32)
        mz = np.asarray(read_elem(f['uns/mz_features']), float)
        coords = np.asarray(f['obsm/spatial'][:], float)
    return names, x, raw, mz, coords


def run_section(path, kegg, extra_ions):
    names, x, raw, mz, coords = read_section(path)
    y_all = np.log1p(raw)
    det = (y_all > 0).mean(0) > .2
    dets = np.where(det)[0]
    top = dets[np.argsort(y_all[:, det].var(0))[-1000:]]
    in_panel = np.isin(dets, top)
    y = y_all[:, dets]                                   # every detected ion
    pos = {g: i for i, g in enumerate(names)}           # human KEGG symbols: mouse panels get no pathway block
    pw = []
    for genes in kegg.values():
        idx = [pos[g] for g in genes if g in pos]
        if len(idx) >= 3:
            v = x[:, idx]
            pw.append(((v - v.mean(0)) / (v.std(0) + 1e-8)).mean(1))
    pw = np.column_stack(pw).astype(np.float32) if pw else np.zeros((len(x), 0), np.float32)
    adj = _norm_adj(coords)
    folds = KMeans(5, n_init=3, random_state=0).fit_predict(coords)
    preds = {m: np.full(y.shape, np.nan, np.float32) for m in METHODS}
    tpreds = {m: np.full(y.shape, np.nan, np.float32) for m in METHODS}     # each model's training-target scale
    lo = np.full(y.shape, np.nan, np.float32); hi = np.full(y.shape, np.nan, np.float32)
    wsum = np.zeros(y.shape[1]); alphas = {}
    for fold in range(5):
        tr, te = folds != fold, folds == fold
        hvg = np.argsort(x[tr].var(0))[-3000:]
        xh = x[:, hvg]
        p = PCA(min(100, tr.sum(), len(hvg)), svd_solver='randomized', random_state=0).fit(xh[tr]).transform(xh).astype(np.float32)
        t = np.hstack([p, adj @ p, adj @ (adj @ p)])
        pwt = np.hstack([pw, adj @ pw])
        blocks = folds[tr]
        jobs = [('MetaboCast', t), ('no diffusion', p)]
        if pw.shape[1]:
            jobs += [('MetaboCast + KEGG', np.hstack([t, pwt])), ('KEGG pathway features + supervised regressor', pwt)]
        for method, f in jobs:
            m = SectionEstimator().fit(f[tr], y[tr], blocks)
            z = m.rank_scores(f[te])
            tpreds[method][te] = z; preds[method][te] = np.maximum(m.to_log1p(z), 0.0)
            alphas.setdefault(method, []).append(m.alpha)
            if method == 'MetaboCast':
                zr = m.rank_scores(f[te], ridge_only=True)
                tpreds['MetaboCast (ridge only)'][te] = zr
                preds['MetaboCast (ridge only)'][te] = np.maximum(m.to_log1p(zr), 0.0)
                # within-section uncertainty: fit on 75% of the training spots,
                # calibrate the 90% half-width on the other 25%, evaluate coverage on the test block
                rs = np.random.RandomState(fold).permutation(np.where(tr)[0]); nc = len(rs) // 4
                cal, fit = rs[:nc], rs[nc:]
                m2 = SectionEstimator().fit(f[fit], y[fit], folds[fit])
                q = np.quantile(np.abs(y[cal] - m2.predict(f[cal])), 0.9, axis=0)
                pt = m2.predict(f[te]); lo[te] = pt - q; hi[te] = pt + q; wsum += 2 * q
        if not pw.shape[1]:                               # no KEGG genes: the KEGG variant equals MetaboCast
            preds['MetaboCast + KEGG'][te] = preds['MetaboCast'][te]
            tpreds['MetaboCast + KEGG'][te] = tpreds['MetaboCast'][te]
        # joint NMF and single-gene baselines on the benchmark panel only
        yp = y[:, in_panel]
        gs = xh / (xh[tr].mean() + 1e-8); ms = yp / (yp[tr].mean() + 1e-8)
        cat = np.hstack([gs[tr], ms[tr]]).clip(0, None)
        nmf = NMF(n_components=min(20, *cat.shape), init='nndsvda', max_iter=250, random_state=0).fit(cat)
        hg, hm = nmf.components_[:, :gs.shape[1]], nmf.components_[:, gs.shape[1]:]
        rec = gs[te] @ np.linalg.pinv(hg) @ hm
        tpreds['joint-NMF'][np.ix_(te, in_panel)] = rec
        preds['joint-NMF'][np.ix_(te, in_panel)] = rec * (yp[tr].mean() + 1e-8)
        sg = single_gene_predict(xh[tr], yp[tr], xh[te])
        preds['single-gene OLS'][np.ix_(te, in_panel)] = sg
        tpreds['single-gene OLS'][np.ix_(te, in_panel)] = (sg - yp[tr].mean(0)) / (yp[tr].std(0) + 1e-8)
    per_ion, scores, scores_t = {}, {}, {}
    ok_y = y.std(0) > 1e-9
    for suffix, P_, S_ in (('', preds, scores), (' [target scale]', tpreds, scores_t)):
        for m, p in P_.items():
            if not np.isfinite(p).any():
                continue                                  # method not applicable (e.g. no KEGG genes)
            cols = in_panel if m in PANEL_ONLY else np.ones(y.shape[1], bool)
            r = np.full(y.shape[1], np.nan, np.float32)
            good = cols & ok_y & np.isfinite(p).all(0)
            good &= np.nan_to_num(p, nan=0).std(0) > 1e-9
            r[good] = col_spearman(p[:, good], y[:, good])
            per_ion[m + suffix] = r
            S_[m] = float(np.nanmedian(r[in_panel]))
    cover = ((y >= lo) & (y <= hi)).mean(0)
    width = wsum / 5
    relwidth = width / (y.std(0) + 1e-8)
    maps = {}
    for nm, target in extra_ions.items():
        j = int(np.argmin(np.abs(mz[dets] - target)))
        if abs(mz[dets][j] - target) <= 0.01:
            maps[nm] = dict(measured=y[:, j], predicted=preds['MetaboCast'][:, j],
                            predicted_rank=tpreds['MetaboCast'][:, j],
                            r=float(per_ion['MetaboCast'][j]),
                            r_target=float(per_ion['MetaboCast [target scale]'][j]),
                            mz=float(mz[dets][j]), lo=lo[:, j], hi=hi[:, j])
    full = per_ion['MetaboCast']
    summary = dict(n_spots=int(len(x)), n_detected=int(len(dets)), n_panel=int(in_panel.sum()),
                   alpha_per_fold=alphas, full_median=float(np.nanmedian(full)), full_mean=float(np.nanmean(full)),
                   full_frac_gt_0_3=float(np.mean(full[np.isfinite(full)] > 0.3)),
                   full_frac_gt_0_5=float(np.mean(full[np.isfinite(full)] > 0.5)),
                   coverage90=float(np.nanmean(cover)), scores_target_scale=scores_t)
    return scores, per_ion, dets, mz, summary, maps, (coords, width, relwidth, cover)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--sections', nargs='+', default=DESIUM)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kegg', type=Path, default=ROOT / 'data/genesets/scMetab_KEGG.gmt')
    ap.add_argument('--maps', nargs='*', default=['EPA:301.2173', 'Xanthine:151.0261', 'Fumarate:115.0037',
                                                  'Tryptophan:203.0826', 'GSH:306.0765', 'DHA:327.2329'],
                    help='name:m/z of held-out maps to save')
    args = ap.parse_args()
    kegg = {}
    for line in args.kegg.read_text().splitlines():
        p = line.split('\t'); kegg[p[0]] = [g for g in p[2:] if g]
    extra = {k: float(v) for k, v in (s.split(':') for s in args.maps)}
    res, per_ion, maps, conf = {}, {}, {}, {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for s in args.sections:
            t0 = time.time()
            scores, pi, dets, mz, summary, mp, cf = run_section(args.data / f'{s}.h5ad', kegg, extra)
            res[s] = dict(scores=scores, **summary, seconds=round(time.time() - t0))
            for m, r in pi.items():
                full = np.full(len(mz), np.nan, np.float32); full[dets] = r
                per_ion[f'{m}__{s}'] = full
            per_ion[f'mz__{s}'] = mz
            for nm, d in mp.items():
                for k, v in d.items():
                    maps[f'{s}__{nm}__{k}'] = np.asarray(v)
            coords, width, relwidth, cover = cf
            conf[f'{s}__coords'] = coords; conf[f'{s}__dets'] = dets
            conf[f'{s}__width'] = width; conf[f'{s}__relwidth'] = relwidth; conf[f'{s}__coverage'] = cover
            print(s, {k: round(v, 4) for k, v in scores.items()}, f'{time.time() - t0:.0f}s', flush=True)
            args.out.write_text(json.dumps(res, indent=1))
            np.savez_compressed(str(args.out).replace('.json', '_per_ion.npz'), **per_ion)
            np.savez_compressed(str(args.out).replace('.json', '_maps.npz'), **maps)
            np.savez_compressed(str(args.out).replace('.json', '_conformal.npz'), **conf)
    print('saved', args.out)


if __name__ == '__main__':
    main()
