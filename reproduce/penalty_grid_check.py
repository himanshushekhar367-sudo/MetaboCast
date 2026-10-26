#!/usr/bin/env python3
"""Ridge penalty beyond the default grid (penalty_grid table).

The cross-validated ridge penalty of the shipped model, of four of the seven leave-one-section-out DESIUM models and
of the HCC/GBM application model is 1e7, the largest value of metabocast.ALPHA_GRID. This script repeats each
selection on the default grid extended to 1e10 and reports:

  shipped      the inner leave-one-section-out score of every penalty for the shipped-model configuration
               (MetaboCast() on all seven DESIUM sections) and the penalty chosen on the extended grid;
  transfer     for every held-out DESIUM section, the penalty chosen on the extended grid and the held-out accuracy
               (detection-matched median Spearman, as in loso_benchmark.py) with that penalty and with the penalty
               chosen on the default grid;
  application  the same score curve for the HCC/GBM configuration (genes shared with HCC and GBM, PCA on all of them)
               and, when the extended choice differs from the default one, the per-ion Spearman correlation between
               HCC-2 predictions made with the two penalties and the change in the HCC-2 normal-to-PVTT contrasts of
               the transfer-supported channels.

The score of a penalty is the one used to choose it: the mean over held-out training sections of the median per-ion
Pearson correlation on detected ions (metabocast.select_alpha).

Usage:
    python reproduce/penalty_grid_check.py --data <DESIUM dir> --root <data root> \
        --out reproduce/results/penalty_grid.json
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reproduce"))
from metabocast import MetaboCast                      # noqa: E402
from metabocast.metabocast import ALPHA_GRID           # noqa: E402
from loso_benchmark import slim, col_spearman          # noqa: E402

DESIUM = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
PATIENTS = ["HCC-1", "HCC-2", "HCC-3", "HCC-4", "HCC-5"]
CYT_H5 = "CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_filtered_feature_bc_matrix.h5"
VIS_H5 = "Parent_Visium_Human_Glioblastoma_filtered_feature_bc_matrix.h5"
EXTRA = np.array([3e7, 1e8, 3e8, 1e9, 3e9, 1e10])
GRID = np.concatenate([ALPHA_GRID, EXTRA])
ND = len(ALPHA_GRID)


def curve(m):
    s = np.asarray(m.alpha_cv_scores_, float)
    mean = np.nanmean(s, axis=1)
    return dict(alpha=[float(a) for a in GRID], mean=[round(float(x), 6) for x in mean],
                per_unit=np.round(s, 6).tolist(), chosen_extended=float(GRID[int(np.nanargmax(mean))]),
                chosen_default=float(ALPHA_GRID[int(np.nanargmax(mean[:ND]))]))


def heldout(m, a):
    pred = m.predict_metabolome(a.copy())
    raw = np.asarray(a.uns["msi"], float)
    y = np.log1p(raw)
    det = (raw > 0).mean(0) > 0.2
    ok = det & (y.std(0) > 1e-9) & (pred.std(0) > 1e-9)
    r = np.full(y.shape[1], np.nan)
    r[ok] = col_spearman(pred[:, ok], y[:, ok])
    return r, det


def gene_symbols_from_h5(path):
    import h5py
    with h5py.File(path, "r") as f:
        ft = f["matrix/features/feature_type"][:]
        nm = f["matrix/features/name"][:]
    return {n.decode() for n, t in zip(nm, ft) if t.decode() == "Gene Expression"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="folder with the seven DESIUM .h5ad sections")
    ap.add_argument("--root", type=Path, required=True, help="data root with the HCC and GBM inputs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    out = dict(grid=[float(x) for x in GRID], default_grid_max=float(ALPHA_GRID[-1]))
    secs = {s: slim(ad.read_h5ad(a.data / f"{s}.h5ad")) for s in DESIUM}
    with threadpool_limits(limits=a.threads):
        # ---------------------------------------------------------------- shipped configuration
        t0 = time.time()
        m = MetaboCast(alpha_grid=GRID).fit([secs[s] for s in DESIUM])
        out["shipped"] = dict(curve(m), seconds=round(time.time() - t0))
        print("shipped", {k: out["shipped"][k] for k in ("chosen_default", "chosen_extended", "seconds")}, flush=True)
        # ---------------------------------------------------------------- leave-one-section-out transfer
        rows = {}
        for held in DESIUM:
            t0 = time.time()
            tr = [secs[s] for s in DESIUM if s != held]
            m = MetaboCast(alpha_grid=GRID).fit(tr)
            c = curve(m)
            r_ext, det = heldout(m, secs[held])
            if c["chosen_extended"] != c["chosen_default"]:
                m0 = MetaboCast(alpha=c["chosen_default"]).fit(tr)
                r_def, _ = heldout(m0, secs[held])
            else:
                r_def = r_ext
            both = det & np.isfinite(r_ext) & np.isfinite(r_def)
            rows[held] = dict(chosen_default=c["chosen_default"], chosen_extended=c["chosen_extended"],
                              inner_mean_at_default=c["mean"][int(np.where(GRID == c["chosen_default"])[0][0])],
                              inner_mean_at_extended=c["mean"][int(np.where(GRID == c["chosen_extended"])[0][0])],
                              median_default=float(np.nanmedian(r_def[det])),
                              median_extended=float(np.nanmedian(r_ext[det])),
                              per_ion_r_agreement=float(spearmanr(r_def[both], r_ext[both])[0]) if both.sum() > 2 else None,
                              n_detected=int(det.sum()), seconds=round(time.time() - t0))
            print(held, rows[held], flush=True)
        md = [v["median_default"] for v in rows.values()]
        me = [v["median_extended"] for v in rows.values()]
        out["transfer"] = dict(per_section=rows, mean_default=float(np.mean(md)), mean_extended=float(np.mean(me)))
        # ---------------------------------------------------------------- HCC/GBM application configuration
        mz = np.asarray(secs[DESIUM[0]].uns["mz_features"], float)
        sets = [set(secs[s].var_names) for s in DESIUM]
        sets += [{l.strip() for l in open(a.root / f"{p}-expr_features.txt")} for p in PATIENTS]
        sets += [gene_symbols_from_h5(a.root / CYT_H5), gene_symbols_from_h5(a.root / VIS_H5)]
        shared = sorted(set.intersection(*sets))
        app = []
        for s in DESIUM:
            b = secs[s]
            x = b[:, shared].layers["log1p"]
            c_ = ad.AnnData(X=np.asarray(x.todense() if sp.issparse(x) else x, np.float32))
            c_.var_names = shared
            c_.layers["log1p"] = c_.X
            c_.uns["msi"] = b.uns["msi"]
            c_.uns["mz_features"] = mz
            c_.obsm["spatial"] = b.obsm["spatial"]
            app.append(c_)
        del secs
        t0 = time.time()
        m = MetaboCast(n_hvg=None, n_pcs=100, k=6, alpha_grid=GRID).fit(app)
        c = curve(m)
        res = dict(c, n_shared_genes=len(shared), seconds=round(time.time() - t0))
        if c["chosen_extended"] != c["chosen_default"]:
            m0 = MetaboCast(n_hvg=None, n_pcs=100, k=6, alpha=c["chosen_default"]).fit(app)
            del app
            pid = "HCC-2"
            genes = [l.strip() for l in open(a.root / f"{pid}-expr_features.txt")]
            gi = np.array([{g: i for i, g in enumerate(genes)}[g] for g in shared])
            M = scipy.io.mmread(gzip.open(a.root / f"{pid}-expr_counts.mtx.gz")).tocsc()
            tot = np.asarray(M.sum(0)).ravel() + 1e-8
            Xq = M[gi, :].T.tocsr().astype(np.float32)
            Xq = Xq.multiply(1.0 / tot[:, None]).multiply(1e4).tocsr()
            Xq.data = np.log1p(Xq.data)
            meta = pd.read_csv(a.root / f"{pid}-expr_meta.csv", index_col=0)
            bc = [l.strip() for l in open(a.root / f"{pid}-expr_barcodes.txt")]
            meta = meta.reindex(bc)
            region = meta["sample.ident"].astype(str).str.replace(pid, "", regex=False).str.lstrip("-").values
            q = ad.AnnData(X=Xq)
            q.var_names = shared
            q.layers["log1p"] = q.X
            q.obsm["spatial"] = meta[["imagerow", "imagecol"]].values.astype(float)
            q.obs["grp"] = region.astype(str)
            p_ext = m.predict_metabolome(q.copy(), section_key="grp")
            p_def = m0.predict_metabolome(q.copy(), section_key="grp")
            ok = (p_ext.std(0) > 1e-9) & (p_def.std(0) > 1e-9)
            rr = col_spearman(p_ext[:, ok], p_def[:, ok])
            acc = pd.read_csv(ROOT / "reproduce" / "results" / "desium_per_ion_accuracy.csv")
            ts = (acc["leave_one_section_out_r"] >= 0.20).values
            N, P = region == "N", region == "P"
            d_ext = p_ext[P].mean(0) - p_ext[N].mean(0)
            d_def = p_def[P].mean(0) - p_def[N].mean(0)
            res["hcc2"] = dict(per_ion_prediction_spearman_median=float(np.nanmedian(rr)),
                               per_ion_prediction_spearman_p05=float(np.nanpercentile(rr, 5)),
                               n_ions=int(ok.sum()),
                               transfer_supported_NtoP_max_abs_change=float(np.max(np.abs(d_ext[ts] - d_def[ts]))),
                               transfer_supported_NtoP_spearman=float(spearmanr(d_ext[ts], d_def[ts])[0]),
                               transfer_supported_NtoP_sign_agreement=float(np.mean(np.sign(d_ext[ts]) == np.sign(d_def[ts]))),
                               n_transfer_supported=int(ts.sum()))
        out["application"] = res
        print("application", {k: res[k] for k in ("chosen_default", "chosen_extended", "seconds")}, res.get("hcc2"), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
