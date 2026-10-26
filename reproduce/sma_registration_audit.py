"""SMA registration audit (sma_registration table and SuppFig_SMA_registration).

Every SMA analysis uses the spot-to-pixel pairing released with the deposit (pipeline/sma_published_pairs/). For each
paired section this script compares two alternatives with it, one that does not use the outcome and one that does:
  geometry  tissue-mask overlap only (pipeline/sma_to_metabocast.register_geometry): orientation, scale and shift
            maximising the overlap between the MALDI tissue pixels and Space Ranger's in-tissue spots
  coupling  the orientation that maximises transcriptome-metabolite coupling (--coupling-h5ad: a folder of .h5ad files
            registered that way); each of their spots is traced back to its raster pixel by its spectrum
For each alternative it reports the share of spots whose pixel lies within one pixel (Chebyshev distance) of the
published pixel and the median over detected ions of the per-ion Spearman correlation between the intensities the
alternative assigns and the published ones. The subject (animal or donor) of each section comes from
pipeline/sma_published_pairs/sections.csv. It also writes <out stem>_tic.csv.gz: for every in-tissue Visium spot,
its coordinates and, for spots in the published pairing, the log total ion count of the paired pixel (the source of
the registration panels of SuppFig_SMA_registration). With --figure it draws those maps.

With --convert-qc (the sma_convert_qc.json written by pipeline/sma_to_metabocast.py) it also writes
reproduce/results/sma_coregistration_qc.csv: per section, the transcriptome-metabolite coupling after registration
(30-PC ridge, random halves) and its shuffled-spot null, a quality check that plays no part in registration.

Usage:
    python reproduce/sma_registration_audit.py --sma <folder containing the 'sma' tree> \
        --out reproduce/results/sma/sma_registration_audit.csv [--coupling-h5ad <folder>] \
        [--convert-qc <sma_h5ad>/sma_convert_qc.json] [--figure overlays.png]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import sma_to_metabocast as conv                      # noqa: E402
from sma_published_pairs import match_spectra         # noqa: E402

LABEL = {"DHB": "DHB", "9AA": "9-AA", "FMP": "FMP-10"}


def per_ion_r(A, B):
    det = ((A > 0).mean(0) > 0.2) & (A.std(0) > 0) & (B.std(0) > 0)
    if not det.any():
        return np.nan
    return float(np.nanmedian(conv._col_spearman(np.log1p(A[:, det]), np.log1p(B[:, det]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sma", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, default=Path(conv.PAIRS_DIR))
    ap.add_argument("--coupling-h5ad", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path, default=None)
    ap.add_argument("--convert-qc", type=Path, default=None)
    args = ap.parse_args()
    subjects = conv.read_subjects(str(args.pairs))
    rows, draw, tics = [], [], []
    for s in conv.find_sections(str(args.sma)):
        sec = s["sample"]
        pairs = conv.read_pairs(sec, str(args.pairs))
        if pairs is None:
            print(f"  [skip] {sec}: no published pairing"); continue
        pos = conv.read_positions(s["tp"])
        msi = pd.read_csv(s["csv"])
        xy = msi.iloc[:, :2].to_numpy().astype(int)
        R = msi.iloc[:, 2:].to_numpy(np.float32)
        in_px = R.sum(1) > 0
        W, H = xy[:, 0].max() + 1, xy[:, 1].max() + 1
        row = -np.ones((W, H), int); row[xy[:, 0], xy[:, 1]] = np.arange(len(xy))
        p = pos.set_index("barcode").loc[pairs.barcode]
        pub = np.c_[pairs.msi_x.to_numpy(), pairs.msi_y.to_numpy()]
        Ypub = R[row[pub[:, 0], pub[:, 1]]]
        reg = conv.register_geometry(pos, xy, in_px)
        gu, gv = conv.map_spots(reg, p.pxl_col, p.pxl_row, W, H)
        inside = gu >= 0
        rg = np.where(inside, row[np.maximum(gu, 0), np.maximum(gv, 0)], -1)
        Ygeo = np.where((rg >= 0)[:, None], R[np.maximum(rg, 0)], 0.0)
        sub = subjects.loc[sec] if subjects is not None and sec in subjects.index else None
        rec = dict(section=f"SMA_{s['panel']}_{sec}", matrix=LABEL.get(s["panel"], s["panel"]),
                   subject=str(sub["subject"]) if sub is not None else "",
                   brain_area=str(sub["brain_area"]) if sub is not None else "",
                   published_pairs=len(pairs),
                   geometry_orientation=("swap" if reg["swap"] else "keep") + (" flipX" if reg["flipx"] else "")
                   + (" flipY" if reg["flipy"] else ""),
                   geometry_iou=reg["iou"], geometry_iou_runner_up=reg["runner_up"]["iou"],
                   geometry_within_1px=float(np.mean(np.maximum(np.abs(gu - pub[:, 0]), np.abs(gv - pub[:, 1])) <= 1)),
                   geometry_map_r=per_ion_r(Ypub, Ygeo))
        if args.coupling_h5ad:
            import anndata as ad
            f = glob.glob(str(args.coupling_h5ad / f"SMA_*_{sec}.h5ad"))
            if f:
                a = ad.read_h5ad(f[0])
                mz_raster = np.round(msi.columns[2:].astype(float).values, 4)
                cols = np.array([int(np.where(mz_raster == m)[0][0]) for m in np.round(np.asarray(a.uns["mz_features"], float), 4)])
                keep = np.isin(a.obs_names, pairs.barcode)
                Yc = np.asarray(a.uns["msi"], np.float64)[keep]
                hit = match_spectra(Yc, R[:, cols].astype(np.float64))
                cu = np.where(hit >= 0, xy[np.maximum(hit, 0), 0], -99); cv = np.where(hit >= 0, xy[np.maximum(hit, 0), 1], -99)
                pp = pairs.set_index("barcode").loc[a.obs_names[keep]]
                rec.update(coupling_spots=int(keep.sum()),
                           coupling_within_1px=float(np.mean(np.maximum(np.abs(cu - pp.msi_x.to_numpy()),
                                                                        np.abs(cv - pp.msi_y.to_numpy())) <= 1)),
                           coupling_map_r=per_ion_r(R[row[pp.msi_x.to_numpy(), pp.msi_y.to_numpy()]][:, cols], Yc))
        rows.append(rec); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
        draw.append((sec, s["panel"], pos, p, np.log1p(Ypub.sum(1)), rec))
        t_in = pos[pos.in_tissue == 1].set_index("barcode")
        tic = pd.Series(np.log1p(Ypub.sum(1)), index=pairs.barcode.to_numpy())
        tics.append(pd.DataFrame({"section": rec["section"], "x": t_in.pxl_col.to_numpy(), "y": t_in.pxl_row.to_numpy(),
                                  "log_tic": tic.reindex(t_in.index).to_numpy().round(3)}))
    if args.convert_qc:
        import json
        qc = json.loads(args.convert_qc.read_text())
        mat = {"DHB": "DHB", "9AA": "9-AA", "FMP": "FMP-10"}
        cls = {"DHB": "lipids", "9AA": "metabolites", "FMP": "neurotransmitters"}
        q = pd.DataFrame([dict(section=f"SMA_{r['panel']}_{r['section']}", matrix=mat[r["panel"]], panel=cls[r["panel"]],
                               species=r.get("species", ""), subject=r.get("subject", ""),
                               region={"Striatum": "striatum", "SubNigra": "substantia nigra"}.get(r.get("brain_area", ""), ""),
                               spots_in_tissue=r["spots_in_tissue"], spots=r["spots"], n_mz=r["n_mz"],
                               within_section_coupling=r["bioR"], shuffled_null=r["nullR"])
                          for r in qc if r.get("status") == "OK"])
        q.to_csv(ROOT / "reproduce" / "results" / "sma_coregistration_qc.csv", index=False)
        print(q.to_string(index=False))
    T = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    T.round(4).to_csv(args.out, index=False)
    pd.concat(tics).to_csv(str(args.out).replace(".csv", "_tic.csv.gz"), index=False)
    print(T.round(3).to_string(index=False))
    if args.figure:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(draw); cols = 5; nrow = int(np.ceil(n / cols))
        fig, axs = plt.subplots(nrow, cols, figsize=(cols * 1.9, nrow * 2.05))
        for ax, (sec, panel, pos, p, tic, rec) in zip(axs.ravel(), draw):
            t = pos.in_tissue.to_numpy() == 1
            ax.scatter(pos.pxl_col[t], -pos.pxl_row[t], s=1.2, color="#D9D9D9", lw=0, rasterized=True)
            ax.scatter(p.pxl_col, -p.pxl_row, c=tic, s=1.2, cmap="viridis", lw=0, rasterized=True)
            ax.set_aspect("equal"); ax.set_axis_off()
            ax.set_title(f"{sec} ({LABEL.get(panel, panel)}, {rec['subject']})", fontsize=5.5)
        for ax in axs.ravel()[n:]:
            ax.set_axis_off()
        fig.tight_layout(); fig.savefig(args.figure, dpi=200); print("saved", args.figure)


if __name__ == "__main__":
    main()
