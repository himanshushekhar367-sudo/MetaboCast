"""Within-section controls for the benchmark of within_benchmark.py (same sections, folds and features).

1. Coordinates only. A k-nearest-neighbour regressor on spot coordinates, with k (1-320) and the weighting
   (uniform or inverse distance) chosen by leave-one-block-out cross-validation inside the training blocks of
   each fold, predicts the held-out block from the measured metabolites of nearby training spots. It measures
   how much of the within-section score spatial position alone explains.
2. Training-only ion selection. The benchmark panel (1,000 most variable ions detected in >20% of spots) is chosen
   on the whole section. Here each fold chooses its panel from the training blocks only, and the held-out block
   is scored on that panel (per-block Spearman), next to the same per-block score on the whole-section panel.
   Ions chosen by every fold are also scored with the pooled out-of-fold predictions used in the benchmark.

MetaboCast is refitted exactly as in within_benchmark.py, so its pooled score reproduces the benchmark.

Usage:
    python reproduce/within_controls.py --data <DESIUM_DIR> --out reproduce/results/within_controls.json [--sections ...]
Writes <out>.json and <out>_per_ion.npz (per-ion pooled Spearman of both methods for every detected ion).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
from metabocast.metabocast import _norm_adj                                   # noqa: E402
from within_benchmark import SectionEstimator, col_spearman, read_section, DESIUM   # noqa: E402

K_GRID = (1, 2, 3, 5, 10, 20, 40, 80, 160, 320)


def knn_predict(tr_xy, tr_y, te_xy, k, weighted):
    k = min(k, len(tr_xy))
    d, idx = cKDTree(tr_xy).query(te_xy, k=k)
    d = d.reshape(len(te_xy), k); idx = idx.reshape(len(te_xy), k)
    w = 1.0 / np.maximum(d, 1e-6) if weighted else np.ones_like(d)
    w = w / w.sum(1, keepdims=True)
    return np.einsum("nk,nkj->nj", w, tr_y[idx])


def tune_knn(xy, y, blocks):
    """Leave-one-block-out over the training blocks; pooled per-ion Spearman, median over ions."""
    best = (-np.inf, None)
    ub = np.unique(blocks)
    for weighted in (False, True):
        for k in K_GRID:
            pred = np.full(y.shape, np.nan, np.float32)
            for b in ub:
                te = blocks == b
                pred[te] = knn_predict(xy[~te], y[~te], xy[te], k, weighted)
            ok = (y.std(0) > 1e-9) & (np.nan_to_num(pred).std(0) > 1e-9)
            s = float(np.nanmedian(col_spearman(pred[:, ok], y[:, ok])))
            if s > best[0]:
                best = (s, (k, weighted))
    return best[1]


def block_scores(pred, y, te, cols):
    ok = cols & (y[te].std(0) > 1e-9) & (pred[te].std(0) > 1e-9)
    return float(np.nanmedian(col_spearman(pred[te][:, ok], y[te][:, ok])))


def run_section(path):
    names, x, raw, mz, coords = read_section(path)
    y_all = np.log1p(raw)
    det = (y_all > 0).mean(0) > .2
    dets = np.where(det)[0]
    y = y_all[:, dets]
    panel_full = np.zeros(len(dets), bool); panel_full[np.argsort(y.var(0))[-1000:]] = True
    adj = _norm_adj(coords)
    folds = KMeans(5, n_init=3, random_state=0).fit_predict(coords)
    pred_mc = np.full(y.shape, np.nan, np.float32); pred_xy = np.full(y.shape, np.nan, np.float32)
    per_fold, chosen, panels = [], [], []
    for fold in range(5):
        tr, te = folds != fold, folds == fold
        hvg = np.argsort(x[tr].var(0))[-3000:]
        xh = x[:, hvg]
        p = PCA(min(100, tr.sum(), len(hvg)), svd_solver='randomized', random_state=0).fit(xh[tr]).transform(xh).astype(np.float32)
        f = np.hstack([p, adj @ p, adj @ (adj @ p)])
        m = SectionEstimator().fit(f[tr], y[tr], folds[tr])
        pred_mc[te] = np.maximum(m.to_log1p(m.rank_scores(f[te])), 0.0)
        k, weighted = tune_knn(coords[tr], y[tr], folds[tr])
        chosen.append(dict(k=int(k), weighted=bool(weighted)))
        pred_xy[te] = knn_predict(coords[tr], y[tr], coords[te], k, weighted)
        # training-only panel: detection and variance on the training blocks only
        yt = y_all[tr]; det_t = (yt > 0).mean(0) > .2
        cand = np.where(det_t)[0]
        top_t = cand[np.argsort(yt[:, cand].var(0))[-1000:]]
        panel_t = np.isin(dets, top_t)                                        # ions outside dets cannot be scored
        panels.append(panel_t)
        per_fold.append(dict(fold=fold, n_panel_train_only=int(panel_t.sum()),
                             overlap_with_full_panel=int((panel_t & panel_full).sum()),
                             metabocast_full_panel=block_scores(pred_mc, y, te, panel_full),
                             metabocast_train_panel=block_scores(pred_mc, y, te, panel_t),
                             coords_full_panel=block_scores(pred_xy, y, te, panel_full),
                             coords_train_panel=block_scores(pred_xy, y, te, panel_t)))
    every = np.logical_and.reduce(panels)
    out = {}
    for name, pr in (("MetaboCast", pred_mc), ("coordinates only", pred_xy)):
        ok = (y.std(0) > 1e-9) & (pr.std(0) > 1e-9)
        r = np.full(y.shape[1], np.nan, np.float32); r[ok] = col_spearman(pr[:, ok], y[:, ok])
        out[name] = r
    summary = dict(n_spots=int(len(x)), n_detected=int(len(dets)), knn_choice=chosen, per_fold=per_fold,
                   n_panel_every_fold=int(every.sum()),
                   pooled={name: dict(full_panel=float(np.nanmedian(r[panel_full])),
                                      every_fold_panel=float(np.nanmedian(r[every])),
                                      all_detected=float(np.nanmedian(r))) for name, r in out.items()})
    return summary, out, dets, mz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--sections', nargs='+', default=DESIUM)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res, arrays = {}, {}
    with threadpool_limits(limits=1):
        for s in args.sections:
            t0 = time.time()
            summary, out, dets, mz = run_section(args.data / f'{s}.h5ad')
            res[s] = dict(summary, seconds=round(time.time() - t0))
            for name, r in out.items():
                full = np.full(len(mz), np.nan, np.float32); full[dets] = r
                arrays[f'{name}__{s}'] = full
            arrays[f'mz__{s}'] = mz
            print(s, json.dumps(summary['pooled']), f"{time.time() - t0:.0f}s", flush=True)
            args.out.write_text(json.dumps(res, indent=1))
            np.savez_compressed(str(args.out).replace('.json', '_per_ion.npz'), **arrays)
    pf = [f for v in res.values() for f in v['per_fold']]
    agg = {k: float(np.mean([f[k] for f in pf])) for k in ('metabocast_full_panel', 'metabocast_train_panel',
                                                           'coords_full_panel', 'coords_train_panel')}
    pooled = {name: {k: float(np.mean([v['pooled'][name][k] for v in res.values()]))
                     for k in ('full_panel', 'every_fold_panel', 'all_detected')}
              for name in ('MetaboCast', 'coordinates only')}
    res['_summary'] = dict(per_block_mean_over_folds=agg, pooled_mean_over_sections=pooled)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res['_summary'], indent=1))


if __name__ == '__main__':
    main()
