"""Screen: how to choose the ridge penalty when a single section is used for training (within-section
benchmark protocol; model_selection table, penalty-rule rows).

Strategies for the ridge penalty of each outer fold (trees fitted once per fold and shared):
  perblock  inner blocks scored one at a time: mean over blocks of the within-block median Pearson
  pooled    inner held-out predictions pooled over blocks, one per-ion Pearson, median over ions (released rule)
  a200      fixed alpha = 200
  a3e4      fixed alpha = 3e4
  fixed200_logtargets  reference: ridge with alpha = 200 on per-ion standardised log1p targets, no trees
Scores: pooled log1p Spearman and target-scale Spearman, median over the top-1,000 panel.
"""
import json, sys, time
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import os as _os
_REP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.dirname(_REP)); sys.path.insert(0, _REP)   # repository root and reproduce/
from metabocast.metabocast import (_norm_adj, _rank_gauss, _CompactForest, _equal_weights, select_alpha,
                                   ALPHA_GRID, _Gram, _gram_solver)
from within_benchmark import read_section, col_spearman


def select_pooled(units, grid=ALPHA_GRID):
    grams = [_Gram(F, T) for F, T in units]
    J = units[0][1].shape[1]; A = len(grid)
    sp, spp, spt = np.zeros((A, J)), np.zeros((A, J)), np.zeros((A, J))
    st, stt, n = np.zeros(J), np.zeros(J), 0
    for k, (F, T) in enumerate(units):
        T = np.asarray(T, np.float64)
        solve = _gram_solver([g for i, g in enumerate(grams) if i != k])
        st += T.sum(0); stt += (T * T).sum(0); n += len(T)
        for a, al in enumerate(grid):
            P = solve(F, al)
            sp[a] += P.sum(0); spp[a] += (P * P).sum(0); spt[a] += (P * T).sum(0)
    cov = spt / n - (sp / n) * (st / n)
    vp = spp / n - (sp / n) ** 2; vt = stt / n - (st / n) ** 2
    with np.errstate(all='ignore'):
        r = cov / np.sqrt(vp * vt)
    sc = np.nanmedian(r, axis=1)
    return float(grid[int(np.nanargmax(sc))]), sc


def to_log1p(Z, lv, Q):
    from scipy.special import ndtr
    P = ndtr(np.asarray(Z, np.float64))
    return np.column_stack([np.interp(P[:, j], lv, Q[:, j]) + 1e-4 * P[:, j] for j in range(P.shape[1])])


def run(path):
    names, x, raw, mz, coords = read_section(path)
    y_all = np.log1p(raw)
    det = (y_all > 0).mean(0) > .2
    dets = np.where(det)[0]
    top = dets[np.argsort(y_all[:, det].var(0))[-1000:]]
    in_panel = np.isin(dets, top)
    y = y_all[:, dets]
    adj = _norm_adj(coords)
    folds = KMeans(5, n_init=3, random_state=0).fit_predict(coords)
    S = ['perblock', 'pooled', 'a200', 'a3e4']
    P = {f'{s}|{h}': np.full(y.shape, np.nan, np.float32) for s in S for h in ('ens', 'ridge')}
    Tt = {k: np.full(y.shape, np.nan, np.float32) for k in P}
    P['fixed200_logtargets'] = np.full(y.shape, np.nan, np.float32); Tt['fixed200_logtargets'] = np.full(y.shape, np.nan, np.float32)
    alph = {s: [] for s in S}
    for fold in range(5):
        tr, te = folds != fold, folds == fold
        hvg = np.argsort(x[tr].var(0))[-3000:]
        xh = x[:, hvg]
        p = PCA(min(100, tr.sum(), len(hvg)), svd_solver='randomized', random_state=0).fit(xh[tr]).transform(xh).astype(np.float32)
        f = np.hstack([p, adj @ p, adj @ (adj @ p)])
        T = _rank_gauss(y[tr])
        blocks = folds[tr]
        units = [(f[tr][blocks == b], T[blocks == b]) for b in np.unique(blocks)]
        a_pb = select_alpha([(F_, T_, np.ones(T.shape[1], bool)) for F_, T_ in units], ALPHA_GRID)[0]
        a_po = select_pooled(units)[0]
        t_mu = T.mean(0); t_sd = T.std(0) + 1e-8; Tz = (T - t_mu) / t_sd
        tpca = PCA(min(50, Tz.shape[1], Tz.shape[0] - 1), svd_solver='randomized', random_state=0).fit(Tz)
        et = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.33, n_jobs=1, random_state=0).fit(
            f[tr], tpca.transform(Tz).astype(np.float32))
        cf = _CompactForest(et)
        ef = lambda F: (tpca.inverse_transform(cf.predict(F).astype(np.float32)) * t_sd + t_mu).astype(np.float32)
        e_tr, e_te = ef(f[tr]), ef(f[te])
        lv = np.linspace(0, 1, 201); Q = np.quantile(y[tr], lv, axis=0)
        for s, a in zip(S, [a_pb, a_po, 200.0, 3e4]):
            alph[s].append(a)
            reg = Ridge(alpha=a).fit(f[tr], T)
            r_tr, r_te = reg.predict(f[tr]).astype(np.float32), reg.predict(f[te]).astype(np.float32)
            w_r, w_e, m_r, m_e = _equal_weights(r_tr, e_tr)
            z = 0.5 * (m_r + m_e) + w_r * (r_te - m_r) + w_e * (e_te - m_e)
            for h, zz in (('ens', z), ('ridge', r_te)):
                Tt[f'{s}|{h}'][te] = zz; P[f'{s}|{h}'][te] = np.maximum(to_log1p(zz, lv, Q), 0)
        leg = Ridge(alpha=200).fit(f[tr], (y[tr] - y[tr].mean(0)) / (y[tr].std(0) + 1e-8))
        zl = leg.predict(f[te]); Tt['fixed200_logtargets'][te] = zl
        P['fixed200_logtargets'][te] = zl * (y[tr].std(0) + 1e-8) + y[tr].mean(0)
    ok = y.std(0) > 1e-9
    out = {}
    for k in P:
        for nm, M in (('log', P[k]), ('tgt', Tt[k])):
            good = ok & (np.nan_to_num(M).std(0) > 1e-9)
            r = np.full(y.shape[1], np.nan); r[good] = col_spearman(M[:, good], y[:, good])
            out[f'{k}|{nm}'] = float(np.nanmedian(r[in_panel]))
    return out, alph


if __name__ == '__main__':
    data, secs, outp = Path(sys.argv[1]), sys.argv[2].split(','), sys.argv[3]
    res = json.load(open(outp)) if Path(outp).exists() else {}
    with threadpool_limits(limits=1):
        for s in secs:
            if s in res: continue
            t0 = time.time()
            o, al = run(data / f'{s}.h5ad')
            res[s] = dict(scores=o, alphas=al)
            print(s, f'{time.time()-t0:.0f}s', {k: round(v, 3) for k, v in o.items() if k.endswith('|log')}, al, flush=True)
            json.dump(res, open(outp, 'w'), indent=1)
