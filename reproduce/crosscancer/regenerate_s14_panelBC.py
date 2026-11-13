#!/usr/bin/env python3
"""
HCC normal->tumour sign concordance for the H&E-only model (crosscancer_summary table; the file name is the script's own label).

The morphology-only model (metabocast.MorphologyMetabolitePredictor) is trained on the 13-feature H&E
block of the DESIUM sections and applied to the HCC H&E images (stranscriptomics/HCC{1..5}/HCC-<n><region>.png,
one image per region). Query features are standardised with the training mean and scale. The transcriptome
column of panel B comes from the cross-cancer run (regenerate_crosscancer.py).

Coordinate mapping
------------------
The per-region PNGs are ~2000 px on the long edge; meta.csv `imagerow`/`imagecol` are in the
~600 px frame recorded in its own `height`/`width` columns. The ratio is a clean 2000/600 in every
section checked, so pixel = image_coord * (png_size / meta_size), verified per region rather than
assumed. Patch radius is derived from the spot grid (0.275 x median nearest-neighbour spacing,
the Visium 55/100 um ratio), because these PNGs carry no scalefactors JSON.

Features are extracted with metabocast.histology._patch_features - the same function DESIUM uses,
not a reimplementation - and standardised per patient, matching add_histology_features.

Usage
-----
    conda activate metabocast
    cd "<repo>/reproduce/crosscancer"

    python regenerate_s14_panelBC.py \
      --root "<data_root>" \
      --crosscancer "<output folder of regenerate_crosscancer.py>" \
      --out "s14_panelBC_$(date +%Y%m%d)" --check

Needs numpy, pandas, scipy, scikit-learn, anndata, pillow. ~10-20 min.
"""
import argparse, hashlib, json, os, platform, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DESIUM = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
PATIENTS = ["HCC-1", "HCC-2", "HCC-3", "HCC-4"]          # HCC-5 has no normal region
FOLDER = {f"HCC-{i}": f"HCC{i}" for i in range(1, 6)}

# the seven literature-expected directions scored in panel C
EXPECTED = {"Glutathione(GSH)": "up", "Ascorbate": "up", "Arachidonate(FA20:4)": "up",
            "DHA(FA22:6)": "up", "Palmitate(FA16:0)": "down", "Stearate(FA18:0)": "down",
            "Hexose": "down"}
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metabocast.annotation import ADDUCTS, neutral_mass      # noqa: E402  the reference of the DESIUM per-ion table

# the seven metabolites -> their entries in the annotation reference
REFNAME = {"Glutathione(GSH)": "Glutathione", "Ascorbate": "Ascorbate", "Arachidonate(FA20:4)": "FA20:4(AA)",
           "DHA(FA22:6)": "FA22:6(DHA)", "Palmitate(FA16:0)": "FA16:0", "Stearate(FA18:0)": "FA18:0", "Hexose": "Hexose"}
MASS = {k: neutral_mass(v) for k, v in REFNAME.items()}
NEG_ADDUCTS = list(ADDUCTS.items())
TOL = 0.01


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ion_index(mz, name):
    for _, dl in NEG_ADDUCTS:
        t = MASS[name] + dl
        i = int(np.argmin(np.abs(mz - t)))
        if abs(mz[i] - t) <= TOL:
            return i
    return None


def spot_radius(xy):
    """0.275 x median nearest-neighbour spacing = the Visium 55/100 um spot-to-pitch ratio."""
    from scipy.spatial import cKDTree
    d = cKDTree(xy).query(xy, k=2)[0][:, 1]
    return max(2, int(round(0.275 * float(np.median(d)))))


def hcc_histology(root, pid, patch_features, n_features, check_only=False):
    """Per-spot H&E features for one HCC patient, standardised across its spots."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    meta = pd.read_csv(root / f"{pid}-expr_meta.csv", index_col=0)
    bc = [l.strip() for l in open(root / f"{pid}-expr_barcodes.txt")]
    meta = meta.reindex(bc)
    region = meta["sample.ident"].astype(str).str.replace(pid, "", regex=False).str.lstrip("-").values
    feats = np.full((len(meta), n_features), np.nan, np.float32)
    report = []
    for reg in sorted(set(region)):
        png = root / "stranscriptomics" / FOLDER[pid] / f"{pid}{reg}.png"
        if not png.exists():
            raise SystemExit(f"missing {png}")
        m = region == reg
        g = meta[m]
        sx = None
        with Image.open(png) as im:
            w, h = im.size
            sx, sy = w / float(g["width"].iloc[0]), h / float(g["height"].iloc[0])
            if abs(sx - sy) / max(sx, sy) > 0.02:
                raise SystemExit(f"{png.name}: x and y scales disagree ({sx:.3f} vs {sy:.3f}) - "
                                 "the image is not a uniform rescale of the recorded frame")
            report.append((f"{pid}{reg}", f"{w}x{h}", f"{g['width'].iloc[0]}x{g['height'].iloc[0]}",
                           round(sx, 3), int(m.sum())))
            if check_only:
                continue
            arr = np.asarray(im.convert("RGB"), np.float32) / 255.0
        xy = np.column_stack([g["imagecol"].values * sx, g["imagerow"].values * sy])
        r = spot_radius(xy)
        sub = np.zeros((len(xy), n_features), np.float32)
        H, W = arr.shape[:2]
        for i, (x, y) in enumerate(xy):
            cx, cy = int(round(x)), int(round(y))
            x0, x1 = max(0, cx - r), min(W, cx + r + 1)
            y0, y1 = max(0, cy - r), min(H, cy + r + 1)
            sub[i] = patch_features(arr[y0:y1, x0:x1])
        feats[m] = sub
        del arr
    if check_only:
        return None, region, report
    if np.isnan(feats).any():
        mu = np.nanmean(feats, 0)
        idx = np.where(np.isnan(feats))
        feats[idx] = np.take(mu, idx[1])
    feats = ((feats - feats.mean(0, keepdims=True)) / (feats.std(0, keepdims=True) + 1e-8)).astype(np.float32)
    return feats, region, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--desium", type=Path, default=None)
    ap.add_argument("--crosscancer", type=Path, default=None,
                    help="directory with HCC-*_pred.npz from regenerate_crosscancer.py (transcriptome column)")
    ap.add_argument("--patients", nargs="+", default=PATIENTS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    repo = (args.repo or Path(__file__).resolve().parents[2]).expanduser().resolve()
    desium = (args.desium or root / "DESIUM/Correlation_NMF_analysis/data").expanduser().resolve()
    sys.path.insert(0, str(repo))
    from metabocast.histology import FEATURE_NAMES, _patch_features
    from metabocast import add_histology_features, MorphologyMetabolitePredictor
    import anndata as ad
    import sklearn

    if args.check:
        print("--- images and coordinate frames ---")
        rows = []
        for pid in args.patients:
            _, _, rep = hcc_histology(root, pid, _patch_features, len(FEATURE_NAMES), check_only=True)
            rows += rep
        print(pd.DataFrame(rows, columns=["region", "png", "meta frame", "scale", "n spots"]).to_string(index=False))
        print("\n--- transcriptome predictions ---")
        cc = args.crosscancer
        print(f"  {'OK  ' if cc and (Path(cc)/'HCC-1_pred.npz').exists() else 'MISSING'} {cc}")
        print("\nCHECK COMPLETE")
        return 0

    out = args.out.expanduser().resolve()
    if out.exists() and not args.force:
        ap.error(f"{out} exists; choose a new path")
    out.mkdir(parents=True, exist_ok=True)
    log(f"python {platform.python_version()} | sklearn {sklearn.__version__}")

    # ---- train the H&E-only model on DESIUM, exactly as the panel-A ablation does ----
    log("fitting MorphologyMetabolitePredictor on DESIUM ...")
    secs, names = [], []
    for s in DESIUM:
        p = desium / f"{s}.h5ad"
        if not p.exists():
            continue
        a = ad.read_h5ad(p)
        try:
            add_histology_features(a, key_added="histology")
        except Exception as e:
            log(f"  skip {s}: {str(e)[:60]}")
            continue
        if "spatial" in a.uns:
            del a.uns["spatial"]
        secs.append(a); names.append(s)
    if len(secs) < 2:
        raise SystemExit("need at least two DESIUM sections with usable histology")
    model = MorphologyMetabolitePredictor().fit(secs)
    mz = np.asarray(model.mz_, float)
    log(f"  trained on {len(secs)} sections ({', '.join(names)}), {len(mz)} ions")
    ions = {n: ion_index(mz, n) for n in EXPECTED}
    missing = [n for n, i in ions.items() if i is None]
    if missing:
        raise SystemExit(f"ions not on the axis: {missing}")
    del secs

    # ---- H&E-only predictions for each HCC patient ----
    he_delta, tx_delta, frames = {}, {}, []
    for pid in args.patients:
        log(f"{pid}: extracting H&E features ...")
        feats, region, rep = hcc_histology(root, pid, _patch_features, len(FEATURE_NAMES))
        for r in rep:
            log(f"    {r[0]}: png {r[1]}, frame {r[2]}, scale {r[3]}, {r[4]} spots")
        shim = ad.AnnData(np.zeros((len(feats), 1), np.float32))
        shim.obsm["histology"] = feats
        pred = model.predict_from_histology(shim)
        N, T = region == "N", region == "T"
        if not (N.any() and T.any()):
            log(f"  {pid}: no paired N/T, skipped"); continue
        he_delta[pid] = {n: float(pred[T, i].mean() - pred[N, i].mean()) for n, i in ions.items()}
        np.savez_compressed(out / f"{pid}_he_only_pred.npz", pred=pred.astype(np.float32), mz=mz,
                            region=region.astype("U8"), histology=feats)
        if args.crosscancer:
            f = Path(args.crosscancer) / f"{pid}_pred.npz"
            if f.exists():
                d = np.load(f, allow_pickle=True)
                tp, tr = d["pred"], d["region"].astype(str)
                tmz = np.asarray(d["mz"], float)
                ti = {n: ion_index(tmz, n) for n in EXPECTED}
                tN, tT = tr == "N", tr == "T"
                tx_delta[pid] = {n: float(tp[tT, ti[n]].mean() - tp[tN, ti[n]].mean()) for n in EXPECTED}
        frames.append(pid)

    def concordant(delta, name):
        want = EXPECTED[name]
        return (delta[name] > 0) if want == "up" else (delta[name] < 0)

    # ---- panel B ----
    rowsB = []
    for pid in frames:
        he = sum(concordant(he_delta[pid], n) for n in EXPECTED)
        row = {"patient": pid, "H&E-only": f"{he}/7", "H&E-only n": he}
        if pid in tx_delta:
            tx = sum(concordant(tx_delta[pid], n) for n in EXPECTED)
            row.update({"transcriptome": f"{tx}/7", "transcriptome n": tx})
        rowsB.append(row)
    B = pd.DataFrame(rowsB)
    tot_he = int(B["H&E-only n"].sum()); n7 = 7 * len(B)
    pooled = {"patient": "pooled", "H&E-only": f"{tot_he}/{n7} ({100*tot_he/n7:.0f}%)", "H&E-only n": tot_he}
    if "transcriptome n" in B:
        tot_tx = int(B["transcriptome n"].sum())
        pooled.update({"transcriptome": f"{tot_tx}/{n7} ({100*tot_tx/n7:.0f}%)", "transcriptome n": tot_tx})
    B = pd.concat([B, pd.DataFrame([pooled])], ignore_index=True)
    B.to_csv(out / "S14_panelB_regenerated.csv", index=False)

    # ---- panel C ----
    rowsC = []
    for n in EXPECTED:
        r = {"metabolite": n, "expected_N_to_T": EXPECTED[n],
             "H&E-only concordant": f"{sum(concordant(he_delta[p], n) for p in frames)}/{len(frames)}"}
        if tx_delta:
            r["transcriptome concordant"] = f"{sum(concordant(tx_delta[p], n) for p in frames if p in tx_delta)}/{len(tx_delta)}"
        for p in frames:
            r[f"{p} H&E delta"] = round(he_delta[p][n], 4)
        for p in frames:
            if p in tx_delta:
                r[f"{p} transcriptome delta"] = round(tx_delta[p][n], 4)
        rowsC.append(r)
    C = pd.DataFrame(rowsC)
    C.to_csv(out / "S14_panelC_regenerated.csv", index=False)

    log("\n=== Panel B: sign concordance of 7 expected directions ===\n" + B.to_string(index=False))
    log("\n=== Panel C: per-metabolite concordance ===\n" +
        C[[c for c in C.columns if "delta" not in c]].to_string(index=False))
    log("\nSubmitted panel B: transcriptome 7/7, 6/7, 1/7, 4/7 (pooled 18/28, 64%); "
        "H&E-only 6/7, 5/7, 5/7, 4/7 (pooled 20/28, 71%).")

    prov = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": Path(__file__).name,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_head": os.popen(f'git -C "{repo}" rev-parse HEAD 2>/dev/null').read().strip() or None,
            "desium_sections_trained_on": names,
            "n_ions": int(len(mz)), "patients": frames,
            "transcriptome_source": Path(args.crosscancer).name if args.crosscancer else None,
            "feature_names": list(FEATURE_NAMES),
            "coordinate_mapping": "pixel = meta imagecol/imagerow * (png size / meta width,height); "
                                  "verified per region, x and y scales required to agree within 2%",
            "patch_radius": "0.275 x median nearest-neighbour spot spacing (Visium 55/100 um)",
            "standardisation": "per patient across its spots, matching add_histology_features",
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "sklearn": sklearn.__version__, "pandas": pd.__version__}}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    log(f"\nwrote {out}")
    for f in sorted(out.iterdir()):
        log(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
    log("done; outputs: S14_panelB_regenerated.csv, S14_panelC_regenerated.csv and provenance.json.")


if __name__ == "__main__":
    main()
