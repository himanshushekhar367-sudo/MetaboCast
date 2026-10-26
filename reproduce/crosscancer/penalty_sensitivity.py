#!/usr/bin/env python3
"""Ridge-penalty sensitivity of the HCC predictions (hcc_patients table, sheet b).

The HCC/GBM model of regenerate_crosscancer.py takes its ridge penalty by leave-one-section-out cross-validation on
the default grid (metabocast.ALPHA_GRID, 1e-1 to 1e7) and selects 1e7, the largest value. This script repeats the
selection on the grid extended to 1e12, refits the application model at the released penalty (1e7) and at the optimum
of the extended grid, predicts the four HCC patients with normal and tumour regions with both models and compares the
contrasts reported in the paper:

  pvtt     HCC-2 normal-to-PVTT difference in mean predicted log intensity
  region   tumour-minus-normal region difference, per patient
  type     tumour-typed minus normal-typed spots (per-spot histological Type), per patient, pooled over its sections
  edge     the same within the leading-edge section, the one section that holds both types

for every channel in the two upper DESIUM accuracy tiers. The summary reports, for the annotated channels (the 56 of
the hcc_patients table) and for all transfer-supported channels, the Spearman correlation of each contrast between the
two penalties, the fraction with the same sign, the median ratio of their magnitudes, and the specific patterns named in
the text: the annotated transfer-supported channels lower or higher in tumour in all four patients, the ten largest
normal-to-PVTT differences of annotated transfer-supported channels, and the curated type-level contrasts.

Writes --out (JSON summary) and <out stem>_contrasts.csv (one row per channel, contrast and patient).

Usage:
    python reproduce/crosscancer/penalty_sensitivity.py --root <data root> \
        --out reproduce/results/crosscancer/penalty_sensitivity.json
"""
import argparse
import gc
import gzip
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "reproduce"))
from metabocast import MetaboCast                                         # noqa: E402
from metabocast.metabocast import ALPHA_GRID                              # noqa: E402
from metabocast.annotation import ADDUCTS as REF_ADDUCTS, candidate_labels, neutral_mass   # noqa: E402
from loso_benchmark import slim                                          # noqa: E402
from regenerate_crosscancer import (DESIUM, PATIENTS, CYT_H5, VIS_H5, CURATED, SIGNATURE, tier,   # noqa: E402
                                    gene_symbols_from_h5)

PAIRED = ["HCC-1", "HCC-2", "HCC-3", "HCC-4"]          # HCC-5 has no normal region
WIDE = np.concatenate([ALPHA_GRID, [3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11, 3e11, 1e12]])
RELEASED = 1e7


def dense_shared(b, shared, mz):
    """Training section on the shared genes as dense float32 log1p expression, with its MSI and coordinates."""
    x = b[:, shared].layers["log1p"]
    c = ad.AnnData(X=np.asarray(x.todense() if hasattr(x, "todense") else x, np.float32))
    c.var_names = shared; c.layers["log1p"] = c.X
    c.uns["msi"] = b.uns["msi"]; c.uns["mz_features"] = mz; c.obsm["spatial"] = b.obsm["spatial"]
    return c


def hcc_query(root, pid, shared):
    genes = [l.strip() for l in open(root / f"{pid}-expr_features.txt")]
    gi = np.array([{g: i for i, g in enumerate(genes)}[g] for g in shared])
    M = scipy.io.mmread(gzip.open(root / f"{pid}-expr_counts.mtx.gz")).tocsc()
    tot = np.asarray(M.sum(0)).ravel() + 1e-8
    X = M[gi, :].T.tocsr().astype(np.float32)
    X = X.multiply(1.0 / tot[:, None]).multiply(1e4).tocsr(); X.data = np.log1p(X.data)
    bc = [l.strip() for l in open(root / f"{pid}-expr_barcodes.txt")]
    meta = pd.read_csv(root / f"{pid}-expr_meta.csv", index_col=0).reindex(bc)
    region = meta["sample.ident"].astype(str).str.replace(pid, "", regex=False).str.lstrip("-").values.astype(str)
    q = ad.AnnData(X=X); q.var_names = shared; q.layers["log1p"] = q.X
    q.obsm["spatial"] = meta[["imagerow", "imagecol"]].values.astype(float); q.obs["grp"] = region
    return q, region, meta["Type"].astype(str).values


def agreement(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    nz = np.abs(a) > 0
    return dict(n=int(len(a)), spearman=round(float(spearmanr(a, b)[0]), 3),
                same_sign=round(float(np.mean(np.sign(a) == np.sign(b))), 3),
                median_magnitude_ratio=round(float(np.median(np.abs(b[nz]) / np.abs(a[nz]))), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="data root with DESIUM, the HCC and the GBM inputs")
    ap.add_argument("--desium", type=Path, default=None)
    ap.add_argument("--accuracy", type=Path, default=REPO / "reproduce/results/desium_per_ion_accuracy.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = a.root.expanduser().resolve()
    desium = (a.desium or root / "DESIUM/Correlation_NMF_analysis/data").expanduser().resolve()
    t_start = time.time()

    secs = {s_: slim(ad.read_h5ad(desium / f"{s_}.h5ad")) for s_ in DESIUM}     # sparse float32, as in loso_benchmark.py
    mz = np.asarray(secs[DESIUM[0]].uns["mz_features"], float)
    sets = [set(secs[s_].var_names) for s_ in DESIUM]
    sets += [{l.strip() for l in open(root / f"{p}-expr_features.txt")} for p in PATIENTS]
    sets += [gene_symbols_from_h5(root / CYT_H5), gene_symbols_from_h5(root / VIS_H5)]
    shared = sorted(set.intersection(*sets))
    train = [dense_shared(secs[s_], shared, mz) for s_ in DESIUM]
    del secs
    gc.collect()

    # tiers and putative candidates, as in regenerate_crosscancer.py
    acc = pd.read_csv(a.accuracy)
    amz = acc.mz.to_numpy(float)
    idx = [int(np.argmin(np.abs(amz - v))) for v in mz]
    within = np.array([acc.within_section_r.iloc[i] if abs(amz[i] - v) <= 1e-3 else np.nan for i, v in zip(idx, mz)])
    transfer = np.array([acc.leave_one_section_out_r.iloc[i] if abs(amz[i] - v) <= 1e-3 else np.nan
                         for i, v in zip(idx, mz)])
    tiers = np.array([tier(w, t) for w, t in zip(within, transfer)], dtype=object)
    ann = np.array(candidate_labels(mz), dtype=object)
    upper = np.where(np.isin(tiers, ["transfer-supported", "within-section only"]))[0]

    out = dict(n_shared_genes=len(shared), released_alpha=RELEASED)
    with threadpool_limits(limits=1):
        t0 = time.time()
        mw = MetaboCast(n_hvg=None, n_pcs=100, k=6, alpha_grid=WIDE).fit(train)
        score = np.nanmean(np.asarray(mw.alpha_cv_scores_, float), axis=1)
        out["inner_cv_curve"] = {f"{x:g}": round(float(s), 6) for x, s in zip(WIDE, score)}
        out["extended_alpha"] = float(mw.alpha_)
        out["inner_cv_at_released"] = out["inner_cv_curve"][f"{RELEASED:g}"]
        out["inner_cv_at_extended"] = out["inner_cv_curve"][f"{mw.alpha_:g}"]
        print(f"extended grid: alpha {mw.alpha_:g} ({time.time() - t0:.0f}s)", flush=True)
        models = {"released": MetaboCast(n_hvg=None, n_pcs=100, k=6, alpha=RELEASED).fit(train),
                  "extended": mw if mw.alpha_ != RELEASED else None}
        del train
        rows, sd = [], {k: [] for k in models}
        for pid in PAIRED:
            q, region, typ = hcc_query(root, pid, shared)
            P = {k: m.predict_metabolome(q.copy(), section_key="grp") for k, m in models.items() if m is not None}
            for k, p in P.items():
                sd[k].append(float(p.std(0).mean()))
            masks = {"region": (region == "N", region == "T"),
                     "type": (typ == "Normal", typ == "Tumor"),
                     "edge": ((typ == "Normal") & (region == "L"), (typ == "Tumor") & (region == "L"))}
            if pid == "HCC-2":
                masks["pvtt"] = (region == "N", region == "P")
            for c, (lo, hi) in masks.items():
                if not (lo.any() and hi.any()):
                    continue
                d = {k: p[hi][:, upper].mean(0) - p[lo][:, upper].mean(0) for k, p in P.items()}
                for i, j in enumerate(upper):
                    rows.append(dict(contrast=c, patient=pid, mz=round(float(mz[j]), 4), annotation=ann[j], tier=tiers[j],
                                     n_low=int(lo.sum()), n_high=int(hi.sum()),
                                     **{f"delta_{k}": round(float(v[i]), 5) for k, v in d.items()}))
            print(pid, "predicted", flush=True)
    df = pd.DataFrame(rows)
    out["mean_spatial_sd"] = {k: round(float(np.mean(v)), 5) for k, v in sd.items()}

    # ------------------------------------------------------------------ summaries
    if "delta_extended" in df:
        ann_rows = df[df.annotation != ""]
        out["agreement"] = {c: {"annotated_upper_tiers": agreement(g.delta_released[g.annotation != ""],
                                                                   g.delta_extended[g.annotation != ""]),
                                "transfer_supported": agreement(g.delta_released[g.tier == "transfer-supported"],
                                                                g.delta_extended[g.tier == "transfer-supported"])}
                            for c, g in df.groupby("contrast")}
        reg = ann_rows[(ann_rows.contrast == "region") & (ann_rows.tier == "transfer-supported")]
        cons = {}
        for k in ("released", "extended"):
            w = reg.pivot_table(index=["mz", "annotation"], columns="patient", values=f"delta_{k}")
            cons[k] = dict(lower_in_all_four=[f"{m:.4f} {n}" for m, n in w.index[(w < 0).all(1)]],
                           higher_in_all_four=[f"{m:.4f} {n}" for m, n in w.index[(w > 0).all(1)]])
        out["consistent_tumour_normal"] = cons
        pv = ann_rows[(ann_rows.contrast == "pvtt") & (ann_rows.tier == "transfer-supported")].copy()
        pv = pv.reindex(pv.delta_released.abs().sort_values(ascending=False).index).head(10)
        out["pvtt_largest_transfer_supported"] = [dict(mz=r.mz, annotation=r.annotation, released=r.delta_released,
                                                       extended=r.delta_extended) for r in pv.itertuples()]
        cur = {}
        for disp in SIGNATURE:
            # the curated channel: first adduct within 0.01 Da in the order [M-H]-, [M+Cl]-, [M+HCOO]-, as in the figures
            m0 = neutral_mass(CURATED[disp])
            hit = [int(np.argmin(np.abs(mz - (m0 + dl)))) for dl in REF_ADDUCTS.values()]
            hit = [j for j, dl in zip(hit, REF_ADDUCTS.values()) if abs(mz[j] - (m0 + dl)) <= 0.01]
            if not hit or hit[0] not in set(upper):
                continue
            j = hit[0]
            for c in ("type", "edge"):
                g = df[(df.contrast == c) & np.isclose(df.mz, round(float(mz[j]), 4))]
                cur.setdefault(disp, {})[c] = dict(released=g.delta_released.round(3).tolist(),
                                                   extended=g.delta_extended.round(3).tolist(),
                                                   same_sign=int((np.sign(g.delta_released) == np.sign(g.delta_extended)).sum()),
                                                   n=int(len(g)))
        out["curated_type_level"] = cur
    out["seconds"] = round(time.time() - t_start)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out.with_name(a.out.stem + "_contrasts.csv"), index=False)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "inner_cv_curve"}, indent=1))


if __name__ == "__main__":
    main()
