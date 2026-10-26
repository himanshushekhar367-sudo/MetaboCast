"""Design check for the released estimator: (a) which ions select the ridge penalty (all detected ions vs the
1,000 most variable detected ions, the benchmark-panel definition) and (b) how the ridge and ExtraTrees
halves are combined (plain mean on the rank scale vs equal weights after scaling each half by its per-ion SD
on the training spots). One ExtraTrees fit per fold is shared by all variants."""
import sys, json, time
import numpy as np, anndata as ad
import os as _os
_REP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.dirname(_REP)); sys.path.insert(0, _REP)   # repository root and reproduce/
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from metabocast import MetaboCast
from metabocast.metabocast import _norm_adj, _rank_gauss, _CompactForest, select_alpha, ALPHA_GRID
from loso_benchmark import slim, col_spearman

mode, D, names, out = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4]


def panel_mask(y, det, n=1000):
    j = np.where(det)[0]
    m = np.zeros(y.shape[1], bool); m[j[np.argsort(y[:, j].var(0))[-n:]]] = True
    return m


def et_fit(F, T):
    mu, sd = T.mean(0), T.std(0) + 1e-8; Tz = (T - mu) / sd
    tp = PCA(min(50, T.shape[1], T.shape[0] - 1), svd_solver='randomized', random_state=0).fit(Tz)
    cf = _CompactForest(ExtraTreesRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.33, n_jobs=1,
                                            random_state=0).fit(F, tp.transform(Tz).astype(np.float32)))
    return lambda X: tp.inverse_transform(cf.predict(X).astype(np.float32)) * sd + mu


def variants(Ftr, T, units_all, units_panel, Fte):
    out = {}
    e_pred = et_fit(Ftr, T)
    e_tr, e_te = e_pred(Ftr), e_pred(Fte)
    for tag, units in (('allions', units_all), ('panel1000', units_panel)):
        a = select_alpha(units, ALPHA_GRID)[0] or 200.0
        reg = Ridge(alpha=a).fit(Ftr, T)
        r_tr, r_te = reg.predict(Ftr), reg.predict(Fte)
        sr, se = r_tr.std(0) + 1e-8, e_tr.std(0) + 1e-8
        out[f'{tag}|alpha'] = a
        out[f'{tag}|ridge'] = r_te
        out[f'{tag}|mean'] = 0.5 * (r_te + e_te)
        out[f'{tag}|eqSD'] = 0.5 * (r_te / sr + e_te / se)
    out['trees'] = e_te
    return out


res = {}
with threadpool_limits(limits=1):
    if mode == 'loso':
        secs = {s: slim(ad.read_h5ad(f'{D}/{s}.h5ad')) for s in names}
        for held in names:
            t0 = time.time()
            tr = [secs[s] for s in names if s != held]
            m = MetaboCast(estimator='ridge', alpha=200.0).fit(tr)      # features only (HVG, PCA, graph)
            def feats(a):
                pos = {g: i for i, g in enumerate(a.var_names)}
                X = np.tile(m.gene_mean_, (a.n_obs, 1)).astype(np.float32)
                pres = [k for k, g in enumerate(m.genes_) if g in pos]
                X[:, pres] = np.asarray(a[:, [m.genes_[k] for k in pres]].layers['log1p'].todense(), np.float32)
                return m._features(X, m.genes_, np.asarray(a.obsm['spatial'], float), None)
            Fs = [feats(a) for a in tr]; Ys = [np.log1p(np.asarray(a.uns['msi'], float)) for a in tr]
            dets = [(np.asarray(a.uns['msi']) > 0).mean(0) > 0.2 for a in tr]
            Ts = [_rank_gauss(y) for y in Ys]
            if len(tr) >= 2:
                ua = [(f, t, d) for f, t, d in zip(Fs, Ts, dets)]
                up = [(f, t, panel_mask(y, d)) for f, t, y, d in zip(Fs, Ts, Ys, dets)]
            else:                                # one training section: five spatial blocks (as MetaboCast.fit)
                blk = KMeans(5, n_init=3, random_state=0).fit_predict(np.asarray(tr[0].obsm['spatial'], float))
                f, t, y, d = Fs[0], Ts[0], Ys[0], dets[0]; pm = panel_mask(y, d)
                ua = [(f[blk == b], t[blk == b], d) for b in range(5)]
                up = [(f[blk == b], t[blk == b], pm) for b in range(5)]
            Ftr, T = np.vstack(Fs), np.vstack(Ts)
            q = secs[held]; Fq = feats(q)
            V = variants(Ftr, T, ua, up, Fq)
            raw = np.asarray(q.uns['msi'], float); y = np.log1p(raw); det = (raw > 0).mean(0) > 0.2
            res[held] = {}
            for k, p in V.items():
                if k.endswith('alpha'):
                    res[held][k] = p; continue
                ok = det & (y.std(0) > 1e-9) & (p.std(0) > 1e-9)
                rr = np.full(y.shape[1], np.nan); rr[ok] = col_spearman(p[:, ok], y[:, ok])
                res[held][k] = float(np.nanmedian(rr[det]))
            print(held, {k: round(v, 4) for k, v in res[held].items()}, f'{time.time()-t0:.0f}s', flush=True)
            json.dump(res, open(out, 'w'), indent=1)
    else:   # within-section, benchmark protocol
        for s in names:
            t0 = time.time()
            a = ad.read_h5ad(f'{D}/{s}.h5ad')
            x = a.layers['log1p']; x = np.asarray(x.toarray() if hasattr(x, 'toarray') else x, np.float32)
            raw = np.asarray(a.uns['msi'], np.float32); yall = np.log1p(raw); co = np.asarray(a.obsm['spatial'], float)
            det = (yall > 0).mean(0) > .2; dets = np.where(det)[0]
            top = dets[np.argsort(yall[:, det].var(0))[-1000:]]; inp = np.isin(dets, top); y = yall[:, dets]
            adj = _norm_adj(co); folds = KMeans(5, n_init=3, random_state=0).fit_predict(co)
            P = {}
            for fold in range(5):
                tr, te = folds != fold, folds == fold
                hvg = np.argsort(x[tr].var(0))[-3000:]; xh = x[:, hvg]
                p = PCA(min(100, tr.sum()), svd_solver='randomized', random_state=0).fit(xh[tr]).transform(xh).astype(np.float32)
                f = np.hstack([p, adj @ p, adj @ (adj @ p)])
                T = _rank_gauss(y[tr]); b = folds[tr]
                allm = np.ones(y.shape[1], bool); pm = panel_mask(y[tr], allm)
                ua = [(f[tr][b == k], T[b == k], allm) for k in np.unique(b)]
                up = [(f[tr][b == k], T[b == k], pm) for k in np.unique(b)]
                V = variants(f[tr], T, ua, up, f[te])
                for k, v in V.items():
                    if k.endswith('alpha'):
                        P.setdefault(k, []).append(v); continue
                    P.setdefault(k, np.full(y.shape, np.nan, np.float32))[te] = v
            res[s] = {}
            for k, p in P.items():
                if k.endswith('alpha'):
                    res[s][k] = p; continue
                r = col_spearman(p, y); res[s][k] = float(np.nanmedian(r[inp]))
            print(s, {k: (round(v, 4) if not isinstance(v, list) else v) for k, v in res[s].items()}, f'{time.time()-t0:.0f}s', flush=True)
            json.dump(res, open(out, 'w'), indent=1)
keys = [k for k in next(iter(res.values())) if not k.endswith('alpha')]
print('MEAN', {k: round(float(np.mean([res[s][k] for s in res])), 4) for k in keys})
