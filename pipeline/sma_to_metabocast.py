#!/usr/bin/env python3
"""
SMA -> MetaboCast converter  (Vicari et al., Nat Biotechnol 2024 SMA deposit).

Builds one MetaboCast-format .h5ad per PAIRED section from the extracted SMA tree:
    <sample>_RNA/outs/filtered_feature_bc_matrix.h5      (Visium transcriptome)
    <sample>_RNA/outs/spatial/tissue_positions_list.csv  (every capture spot and Space Ranger's in-tissue call)
    <sample>_MSI/<...>.Visium.<matrix>.<date>_smamsi.csv (MALDI raster: x, y + m/z columns; off-tissue pixels are 0)

Registration. By default each spot takes the spectrum of the MALDI pixel that the SMA authors paired with it
(pipeline/sma_published_pairs/<section>.csv, exported from the deposit by sma_published_pairs.R/.py). The authors
aligned the MALDI and Visium images by hand in STUtility and paired every spot with its nearest pixel, so the pairing
was fixed before, and independently of, any prediction made here. For a section without a published pairing (or
with --pairs none) the raster is registered by tissue geometry alone: its orientation (the eight axis-aligned
options), a scale correction within +-10% and the translation are chosen to maximise the overlap (intersection over
union over the capture spots) between the MALDI tissue mask (pixels with any signal) and the Visium tissue mask
(Space Ranger's in-tissue spots, detected on the H&E image). The geometry registration is computed for every section
and, where a published pairing exists, its agreement with that pairing is reported (share of spots whose two pixels
are within one pixel of each other). Spots whose pixel carries no signal are dropped. Transcriptome-metabolite
coupling against a shuffled-spot null (30-PC ridge) is computed afterwards as a quality check and plays no part in
registration.

Sections are grouped by MALDI matrix (DHB lipids, 9-AA metabolites, FMP-10 neurotransmitters); within a group all
sections share a common m/z axis so they pool for leave-one-section-out. The subject (animal or donor) of each
section, from the deposit's metadata.csv via sma_published_pairs/sections.csv, is stored in uns["sma"].

Usage:
    python pipeline/sma_to_metabocast.py "/path/to/SMA" [--pairs pipeline/sma_published_pairs | --pairs none]
    (the folder that contains the 'sma' tree; the .h5ad files and sma_convert_qc.json are written to
     <that folder>/sma_h5ad/ unless --out is given, ready for pipeline/train_and_evaluate.py)

Deps: pip install scanpy anndata scikit-learn scipy pandas numpy
"""
import argparse, os, sys, glob, json, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
try:
    import scanpy as sc, anndata as ad
    from scipy.signal import fftconvolve
    from scipy.spatial import cKDTree
    from scipy.stats import rankdata
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
except ImportError as e:
    print("MISSING DEPENDENCY:", e); print("  pip install scanpy anndata scikit-learn scipy pandas numpy"); sys.exit(1)

ORIENTATIONS = list(itertools.product((False, True), repeat=3))      # (swap, flip x, flip y)
SCALES = np.round(np.arange(0.90, 1.1001, 0.02), 2)                     # correction of the nominal pixel size
SUBPIXEL = np.round(np.arange(-0.5, 0.501, 0.1), 2)
PAIRS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sma_published_pairs")


def panel_of(fname):
    f = fname.lower()
    if ".dhb." in f: return "DHB"
    if ".9aa." in f or "_9aa" in f: return "9AA"
    if ".fmp." in f: return "FMP"
    return "OTHER"


def find_sections(root):
    """Return list of dicts {csv, sample, panel, h5, tp} for real paired sections."""
    out = []
    for csv in sorted(glob.glob(os.path.join(root, "**", "*smamsi.csv"), recursive=True)):
        if "__MACOSX" in csv or os.path.basename(csv).startswith("._"): continue
        msi_dir = os.path.dirname(csv)                       # ..._MSI
        sample  = os.path.basename(msi_dir)
        sample  = sample[:-4] if sample.endswith("_MSI") else sample
        odata   = os.path.dirname(msi_dir)                   # output_data
        outs    = os.path.join(odata, sample + "_RNA", "outs")
        h5      = os.path.join(outs, "filtered_feature_bc_matrix.h5")
        tp      = os.path.join(outs, "spatial", "tissue_positions_list.csv")
        if not os.path.exists(h5):
            h5c = glob.glob(os.path.join(odata, sample + "*_RNA", "outs", "filtered_feature_bc_matrix.h5"))
            h5 = h5c[0] if h5c else h5
            tp = os.path.join(os.path.dirname(h5), "spatial", "tissue_positions_list.csv")
        if not (os.path.exists(h5) and os.path.exists(tp)):
            print(f"  [skip] {sample}: no paired RNA outs (h5={os.path.exists(h5)}, tp={os.path.exists(tp)})"); continue
        out.append(dict(csv=csv, sample=sample, panel=panel_of(os.path.basename(csv)), h5=h5, tp=tp))
    return out


def read_positions(tp):
    """Every capture spot (in and out of tissue)."""
    p = pd.read_csv(tp, header=None)
    if not str(p.iloc[0, 1]).strip().lstrip("-").isdigit():          # Space Ranger >= 2 writes a header row
        p = pd.read_csv(tp)
    p = p.iloc[:, :6]
    p.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    return p


def spot_pitch(pos):
    """Centre-to-centre distance of neighbouring spots in full-resolution pixels (100 um on Visium)."""
    xy = pos[["pxl_col", "pxl_row"]].to_numpy(float)
    return float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))


def to_raster(X, Y, s, orient):
    """Spot centres (full-res pixels) -> raster coordinates before translation, for one orientation and pixel size s."""
    swap, fx, fy = orient
    a, b = X / s, Y / s
    a = -a if fx else a; b = -b if fy else b
    return (b, a) if swap else (a, b)


def _iou(u, v, tissue, mask):
    iu, iv = np.rint(u).astype(int), np.rint(v).astype(int)
    inside = (iu >= 0) & (iu < mask.shape[0]) & (iv >= 0) & (iv < mask.shape[1])
    on = np.zeros(len(u), bool); on[inside] = mask[iu[inside], iv[inside]]
    inter = np.sum(on & tissue)
    return inter / max(np.sum(tissue) + np.sum(on) - inter, 1)


def register_geometry(pos_all, raster_xy, pixel_in_tissue, all_orientations=False):
    """Choose orientation, scale and translation of the raster by tissue overlap alone (see module docstring).
    With all_orientations=True the best overlap of every orientation is also returned (registration audit)."""
    W, H = raster_xy[:, 0].max() + 1, raster_xy[:, 1].max() + 1
    mask = np.zeros((W, H), bool); mask[raster_xy[pixel_in_tissue, 0], raster_xy[pixel_in_tissue, 1]] = True
    X = pos_all.pxl_col.to_numpy(float); Y = pos_all.pxl_row.to_numpy(float)
    tissue = pos_all.in_tissue.to_numpy() == 1
    pitch = spot_pitch(pos_all)
    best_by_orient = {}
    for orient in ORIENTATIONS:
        for f in SCALES:
            u0, v0 = to_raster(X, Y, pitch * f, orient)
            ur, vr = np.rint(u0).astype(int), np.rint(v0).astype(int)
            iu, iv = ur - ur.min(), vr - vr.min()
            V = np.zeros((iu.max() + 1, iv.max() + 1)); S = np.zeros_like(V)
            np.add.at(V, (iu, iv), tissue.astype(float)); np.add.at(S, (iu, iv), 1.0)
            A = np.rint(fftconvolve(mask.astype(float), V[::-1, ::-1]))
            B = np.rint(fftconvolve(mask.astype(float), S[::-1, ::-1]))
            iou = A / np.maximum(tissue.sum() + B - A, 1)
            k, l = np.unravel_index(np.argmax(iou), iou.shape)
            du = ur.min() - (k - (V.shape[0] - 1)); dv = vr.min() - (l - (V.shape[1] - 1))
            best = (-1, None)
            for eu in SUBPIXEL:                                           # refine below one pixel
                for ev in SUBPIXEL:
                    sc_ = _iou(u0 - du - eu, v0 - dv - ev, tissue, mask)
                    if sc_ > best[0]:
                        best = (sc_, (float(du + eu), float(dv + ev)))
            if orient not in best_by_orient or best[0] > best_by_orient[orient]["iou"]:
                best_by_orient[orient] = dict(iou=float(best[0]), scale=float(f), shift=best[1])
    ranked = sorted(best_by_orient.items(), key=lambda kv: -kv[1]["iou"])
    (orient, b), (orient2, b2) = ranked[0], ranked[1]
    out = dict(swap=bool(orient[0]), flipx=bool(orient[1]), flipy=bool(orient[2]), scale=b["scale"],
               pixel_size_px=pitch * b["scale"], shift=b["shift"], iou=round(b["iou"], 4),
               runner_up=dict(swap=bool(orient2[0]), flipx=bool(orient2[1]), flipy=bool(orient2[2]),
                              iou=round(b2["iou"], 4)),
               iou_margin=round(b["iou"] - b2["iou"], 4), spot_pitch_px=pitch)
    if all_orientations:
        out["by_orientation"] = {o: dict(v, pixel_size_px=pitch * v["scale"]) for o, v in best_by_orient.items()}
    return out


def map_spots(reg, X, Y, W, H):
    """Raster pixel (x, y) under each spot centre for a registration; -1 where outside the raster."""
    u0, v0 = to_raster(np.asarray(X, float), np.asarray(Y, float), reg["pixel_size_px"],
                       (reg["swap"], reg["flipx"], reg["flipy"]))
    iu = np.rint(u0 - reg["shift"][0]).astype(int); iv = np.rint(v0 - reg["shift"][1]).astype(int)
    ok = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    return np.where(ok, iu, -1), np.where(ok, iv, -1)


def gene_pcs(g):
    q = g.copy(); sc.pp.normalize_total(q, target_sum=1e4); sc.pp.log1p(q)
    sc.pp.highly_variable_genes(q, n_top_genes=min(2000, q.n_vars-1))
    Xh = q[:, q.var.highly_variable].X
    Xh = np.asarray(Xh.todense()) if hasattr(Xh, "todense") else np.asarray(Xh)
    return PCA(min(30, Xh.shape[1]), random_state=0).fit_transform(Xh)


def _col_spearman(P, Y):
    """Column-wise Spearman correlation (average ranks for ties, as scipy.stats.spearmanr)."""
    R1 = rankdata(P, axis=0); R2 = rankdata(Y, axis=0)
    R1 -= R1.mean(0); R2 -= R2.mean(0)
    den = np.sqrt((R1 * R1).sum(0) * (R2 * R2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (R1 * R2).sum(0) / den


def couple(P, Y, reps=3):
    """Quality check after registration: random-half ridge coupling and a shuffled-spot null."""
    rng = np.random.RandomState(0); n = len(P); bios=[]; nulls=[]
    for _ in range(reps):
        o = rng.permutation(n); tr=o[:n//2]; te=o[n//2:]
        det = (np.expm1(Y[tr]) > 0).mean(0) > 0.2
        pred = Ridge(alpha=100).fit(P[tr], Y[tr]).predict(P[te])
        Yte = Y[te]
        cols = np.where(det & (Yte.std(0) > 1e-9))[0]
        if not len(cols): return 0.0,0.0
        bios.append(np.nanmedian(_col_spearman(pred[:, cols], Yte[:, cols])))
        pm=rng.permutation(len(te))
        nulls.append(np.nanmedian(_col_spearman(pred[:, cols], Yte[pm][:, cols])))
    return float(np.mean(bios)), float(np.mean(nulls))


def read_pairs(sample, pairs_dir):
    """Published spot-pixel pairing for one section (barcode, msi_x, msi_y), or None."""
    if not pairs_dir:
        return None
    f = os.path.join(pairs_dir, f"{sample}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def read_subjects(pairs_dir):
    f = os.path.join(pairs_dir, "sections.csv") if pairs_dir else ""
    return pd.read_csv(f).set_index("section") if f and os.path.exists(f) else None


def convert_section(s, common_mz, pairs_dir=PAIRS_DIR, subjects=None):
    g = sc.read_10x_h5(s["h5"]); g.var_names_make_unique()
    pos_all = read_positions(s["tp"])
    msi = pd.read_csv(s["csv"])
    xy = msi.iloc[:, :2].to_numpy().astype(int)
    mz = np.round(msi.columns[2:].astype(float).values, 4)
    M = msi.iloc[:, 2:].to_numpy(np.float32)
    in_px = M.sum(1) > 0
    reg = register_geometry(pos_all, xy, in_px)                          # tissue geometry only
    W, H = xy[:, 0].max() + 1, xy[:, 1].max() + 1
    row = -np.ones((W, H), int); row[xy[:, 0], xy[:, 1]] = np.arange(len(xy))
    pos = pos_all[pos_all.in_tissue == 1]
    common = [b for b in g.obs_names if b in set(pos.barcode)]
    pos = pos.set_index("barcode").loc[common].reset_index()
    giu, giv = map_spots(reg, pos.pxl_col, pos.pxl_row, W, H)             # geometry pixel under each spot
    pairs = read_pairs(s["sample"], pairs_dir)
    if pairs is not None:
        pr = pairs.drop_duplicates("barcode").set_index("barcode").reindex(pos.barcode)
        iu = pr.msi_x.fillna(-1).astype(int).to_numpy(); iv = pr.msi_y.fillna(-1).astype(int).to_numpy()
        paired = iu >= 0
        agree = float(np.mean(np.maximum(np.abs(giu[paired] - iu[paired]), np.abs(giv[paired] - iv[paired])) <= 1))
        method = "published pairing (Vicari et al. 2024; manual STUtility alignment, nearest pixel)"
    else:
        iu, iv, agree = giu, giv, float("nan")
        method = "tissue geometry (IoU)"
    inside = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    r = np.where(inside, row[np.clip(iu, 0, W - 1), np.clip(iv, 0, H - 1)], -1)
    keep_spot = (r >= 0) & np.where(r >= 0, in_px[np.maximum(r, 0)], False)
    if keep_spot.sum() < 200:
        return None, {"section": s["sample"], "panel": s["panel"], "status": "FEW_SPOTS"}
    g = g[[b for b, k in zip(pos.barcode, keep_spot) if k]].copy(); pos = pos[keep_spot].reset_index(drop=True)
    keep_mz = np.array([np.where(mz == m)[0][0] for m in common_mz])
    Y = M[r[keep_spot]][:, keep_mz].astype(np.float32)
    px = pos.pxl_col.to_numpy(float); py = pos.pxl_row.to_numpy(float)
    bio, null = couple(gene_pcs(g), np.log1p(Y))                       # post-hoc quality check only
    ln = g.copy(); sc.pp.normalize_total(ln, target_sum=1e4); sc.pp.log1p(ln)
    obs = pos.set_index("barcode")
    obs["msi_x"] = xy[r[keep_spot], 0]; obs["msi_y"] = xy[r[keep_spot], 1]
    A = ad.AnnData(X=g.X.copy(), obs=obs, var=g.var.copy())
    A.layers["log1p"] = ln.X.copy()
    A.obsm["spatial"] = np.c_[px, py]
    A.uns["msi"] = Y
    A.uns["mz_features"] = np.asarray(common_mz, float)
    sub = subjects.loc[s["sample"]] if subjects is not None and s["sample"] in subjects.index else None
    meta = dict(registration=method,
                published_pairs=int(len(pairs)) if pairs is not None else 0,
                geometry_swap=reg["swap"], geometry_flipx=reg["flipx"], geometry_flipy=reg["flipy"],
                geometry_scale=reg["scale"], geometry_pixel_size_px=round(reg["pixel_size_px"], 2),
                geometry_shift_x=round(reg["shift"][0], 2), geometry_shift_y=round(reg["shift"][1], 2),
                geometry_iou=reg["iou"], geometry_iou_runner_up=reg["runner_up"]["iou"],
                geometry_iou_margin=reg["iou_margin"],
                geometry_agrees_with_published=round(agree, 4) if agree == agree else None,
                subject=str(sub["subject"]) if sub is not None else "",
                species=str(sub["species"]) if sub is not None else "",
                brain_area=str(sub["brain_area"]) if sub is not None else "",
                bioR=round(bio, 3), nullR=round(null, 3))
    A.uns["sma"] = dict(sample=s["sample"], panel=s["panel"],
                        **{k: ("" if v is None else v) for k, v in meta.items()})
    ok = pairs is not None or reg["iou_margin"] >= 0.02
    rec = {"section": s["sample"], "panel": s["panel"], "spots_in_tissue": int(len(keep_spot)),
           "spots": int(A.n_obs), "n_mz": int(len(common_mz)), **meta,
           "status": "OK" if ok else "REVIEW_orientation_margin"}
    return A, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".", help="folder that contains the extracted 'sma' tree")
    ap.add_argument("--pairs", default=PAIRS_DIR,
                    help="folder of published pair tables (<section>.csv); 'none' registers by tissue geometry only")
    ap.add_argument("--out", default=None, help="output folder (default <root>/sma_h5ad)")
    args = ap.parse_args()
    root = args.root
    pairs_dir = None if str(args.pairs).lower() == "none" else args.pairs
    out = args.out or os.path.join(root, "sma_h5ad"); os.makedirs(out, exist_ok=True)
    subjects = read_subjects(pairs_dir)
    print("="*64+"\nSMA -> MetaboCast converter\n root : "+root+"\n pairs: "+str(pairs_dir)+"\n out  : "+out+"\n"+"="*64)
    secs = find_sections(root)
    if not secs: print("No paired SMA sections found."); return
    from collections import defaultdict
    byp = defaultdict(list)
    for s in secs: byp[s["panel"]].append(s)
    # per-panel common m/z axis (intersection of rounded headers)
    common = {}
    for panel, lst in byp.items():
        sets=[]
        for s in lst:
            hdr = pd.read_csv(s["csv"], nrows=0).columns[2:]
            sets.append(set(np.round(hdr.astype(float).values, 4)))
        cm = sorted(set.intersection(*sets)) if sets else []
        common[panel] = np.array(cm)
        print(f"  panel {panel:4s}: {len(lst)} section(s), common m/z axis = {len(cm)}")
    recs=[]
    for panel, lst in byp.items():
        if len(common[panel]) < 5:
            for s in lst: recs.append({"section":s["sample"],"panel":panel,"status":"NO_COMMON_MZ"})
            continue
        for s in lst:
            try:
                A, rec = convert_section(s, common[panel], pairs_dir, subjects)
            except Exception as ex:
                recs.append({"section":s["sample"],"panel":panel,"status":f"ERROR:{ex}"}); print("  [err]",s["sample"],ex); continue
            recs.append(rec)
            if A is not None:
                fp = os.path.join(out, f"SMA_{panel}_{s['sample']}.h5ad"); A.write(fp)
                agree = rec["geometry_agrees_with_published"]
                print(f"  [OK] {os.path.basename(fp):34s} {rec['spots']:5d}/{rec['spots_in_tissue']} spots x {rec['n_mz']} mz | "
                      f"{'published pairing' if rec['published_pairs'] else 'geometry'} | geometry IoU={rec['geometry_iou']:.3f} "
                      f"(runner-up {rec['geometry_iou_runner_up']:.3f})"
                      + (f", agrees with published for {agree:.0%} of spots" if agree is not None else "")
                      + f" | coupling={rec['bioR']:+.3f} null={rec['nullR']:+.3f} {rec['status']}", flush=True)
    json.dump(recs, open(os.path.join(out,"sma_convert_qc.json"),"w"), indent=2)
    print("\n"+"="*64+"\nQC summary\n"+"="*64)
    ok=[r for r in recs if r.get("status")=="OK"]
    for panel in byp:
        pr=[r for r in recs if r.get("panel")==panel]
        print(f" {panel:4s}: {sum(1 for r in pr if r.get('status')=='OK')}/{len(pr)} OK, common m/z={len(common.get(panel,[]))}")
    print(f" total sections written OK : {len(ok)}")
    if ok:
        print(f" median coupling after registration: "
              f"{np.median([r['bioR'] for r in ok]):+.3f} vs null {np.median([r['nullR'] for r in ok]):+.3f}")
    print(f" wrote .h5ad + sma_convert_qc.json to: {out}")
    print(" next: python pipeline/train_and_evaluate.py \"<folder with the converted .h5ad files>\"")

if __name__ == "__main__":
    main()
