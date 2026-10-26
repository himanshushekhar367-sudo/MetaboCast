#!/usr/bin/env python3
"""
One model, the whole cross-cancer section: HCC (hcc_pvtt and hcc_patients tables, Fig. 7a-c) and both
GBM sections (gbm_protein table, Fig. 7d-f) computed from a single pooled DESIUM fit. Output file names
(S8_*, S9_*, S10_*) are the run's own labels; reproduce/manifest.py gives the supplementary table numbers.

What it writes
--------------
  HCC-{1..5}_pred.npz              per-spot predicted metabolome + region labels
  GBM_CytAssist_pred.npz           predicted metabolome, protein matrix, coordinates
  GBM_Visium_pred.npz              predicted metabolome, niche labels, coordinates
  S8_full_all_ions.csv             HCC-2 Normal->PVTT, every ion: putative candidates (the reference of
                                   metabocast/annotation.py, nearest first), difference in mean predicted log intensity,
                                   the spread of each region and the ion's DESIUM accuracy tier (no per-spot tests:
                                   spots are not independent replicates)
  S8_supported_annotated.csv       the annotated ions in the two upper accuracy tiers (the subset the table reports)
  S9_and_Fig7c_source.csv          cross-patient contrasts of the displayed metabolites (also the Fig. 7c source)
  S10_metabolite_x_protein.csv     Fig. 7d / gbm_protein table Spearman matrix
  S10_long.csv                     the same, one row per metabolite-protein pair
  Fig7e_niches.csv                 per-spot niche assignment + coordinates
  Fig7f_metabolite_by_niche.csv    mean z-scored predicted intensity per niche
  provenance.json                  hashes, versions, every parameter

Usage
-----
    conda activate metabocast
    cd "<repo>/reproduce/crosscancer"

    python regenerate_crosscancer.py \
      --root "<data_root>" \
      --out  "crosscancer_regen_$(date +%Y%m%d)" \
      --check          # drop --check for the real run

Needs numpy, pandas, scipy, scikit-learn, anndata, h5py. ~8 GB RAM, ~3-6 min.
Never overwrites an existing --out.
"""
import argparse, gzip, hashlib, json, os, platform, sys, tarfile, time, warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
from scipy.sparse import csc_matrix
from scipy.stats import rankdata
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

DESIUM = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
PATIENTS = ["HCC-1", "HCC-2", "HCC-3", "HCC-4", "HCC-5"]
CYT_H5 = "CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_filtered_feature_bc_matrix.h5"
CYT_SPATIAL = "CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_spatial.tar.gz"
VIS_H5 = "Parent_Visium_Human_Glioblastoma_filtered_feature_bc_matrix.h5"
VIS_SPATIAL = "Parent_Visium_Human_Glioblastoma_spatial.tar.gz"

SIGNATURE = ["Arachidonate(FA20:4)", "DHA(FA22:6)", "Hexose", "Inosine", "Taurine", "Urate",
             "Glutathione(GSH)", "Ascorbate", "Glutamate", "Palmitate(FA16:0)", "Stearate(FA18:0)",
             "Oleate(FA18:1)", "Linoleate(FA18:2)"]
# DESIUM accuracy tiers, the rule of MetaboCast.reliability_table()
TRANSFER_MIN, WITHIN_MIN = 0.20, 0.30


def tier(within, transfer):
    if np.isfinite(transfer) and transfer >= TRANSFER_MIN:
        return "transfer-supported"
    if np.isfinite(within) and within >= WITHIN_MIN:
        return "within-section only"
    return "low" if (np.isfinite(within) or np.isfinite(transfer)) else "uncalibrated"
# the eleven metabolite rows of the gbm_protein table / Fig. 7f
GBM_PANEL = ["Glutathione(GSH)", "Arachidonate(FA20:4)", "DHA(FA22:6)", "Palmitate(FA16:0)",
             "Stearate(FA18:0)", "Glutamate", "Hexose", "Inosine", "Taurine",
             "N-acetylaspartate", "Urate"]
# curated display name -> group name in the annotation reference (metabocast/annotation.py), for the named panels;
# every channel is annotated from that reference, the one the DESIUM per-ion table uses
CURATED = {"Arachidonate(FA20:4)": "FA20:4(AA)", "DHA(FA22:6)": "FA22:6(DHA)", "Hexose": "Hexose", "Inosine": "Inosine",
           "Taurine": "Taurine", "Urate": "Urate", "Glutathione(GSH)": "Glutathione", "Ascorbate": "Ascorbate",
           "Glutamate": "Glutamate", "Palmitate(FA16:0)": "FA16:0", "Stearate(FA18:0)": "FA18:0",
           "Oleate(FA18:1)": "FA18:1", "Linoleate(FA18:2)": "FA18:2", "N-acetylaspartate": "N-acetylaspartate"}
MASS, ADDUCTS = {}, {}          # neutral masses and adduct shifts, filled from metabocast.annotation in main()
TOL = 0.01


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ion_index(mz, name):
    """Channel of a curated metabolite: its first adduct within TOL, in the order [M-H]-, [M+Cl]-, [M+HCOO]-."""
    for dl in ADDUCTS.values():
        t = MASS[name] + dl
        i = int(np.argmin(np.abs(mz - t)))
        if abs(mz[i] - t) <= TOL:
            return i
    return None


def read_10x_h5(path):
    """Return (features x barcodes CSC, barcodes, names, feature_type, isotype flag)."""
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(int(x) for x in m["shape"][:])
        X = csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        bc = [b.decode() for b in m["barcodes"][:]]
        feat = m["features"]
        name = np.array([b.decode() for b in feat["name"][:]])
        ftype = np.array([b.decode() for b in feat["feature_type"][:]])
        iso = (np.array([b.decode() for b in feat["isotype_control"][:]])
               if "isotype_control" in feat else np.array([""] * len(name)))
    return X, bc, name, ftype, iso


def is_isotype(name, iso):
    """The 10x h5 writes TRUE/FALSE (not True/False); fall back to the IgG naming convention."""
    flag = np.array([str(v).strip().upper() == "TRUE" for v in iso])
    byname = np.array([n.lower().startswith(("mouse_ig", "rat_ig", "rabbit_ig")) for n in name])
    return flag | byname


def read_positions(targz):
    """Visium spatial tarball -> DataFrame with barcode, in_tissue and full-res pixel coords."""
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    with tarfile.open(targz, "r:gz") as t:
        names = t.getnames()
        for cand in ("spatial/tissue_positions.csv", "spatial/tissue_positions_list.csv"):
            if cand not in names:
                continue
            raw = t.extractfile(cand).read().decode()
            header = 0 if raw.split("\n", 1)[0].lower().lstrip().startswith("barcode") else None
            from io import StringIO
            df = pd.read_csv(StringIO(raw), header=header)
            if header is None:
                df.columns = cols[:df.shape[1]]
            df.columns = [c.strip() for c in df.columns]
            return df
    raise SystemExit(f"no tissue_positions file inside {targz}")


def spearman_matrix(A, B):
    """Column-wise Spearman between every column of A and every column of B."""
    Ra = rankdata(A, axis=0).astype(np.float64)
    Rb = rankdata(B, axis=0).astype(np.float64)
    Za = (Ra - Ra.mean(0)) / (Ra.std(0) + 1e-12)
    Zb = (Rb - Rb.mean(0)) / (Rb.std(0) + 1e-12)
    return (Za.T @ Zb) / len(Ra)


def unique_names(names):
    """PTPRC appears twice on the panel; the second one is called PTPRC_2."""
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def gene_symbols_from_h5(path):
    with h5py.File(path, "r") as f:
        ft = f["matrix/features/feature_type"][:]
        nm = f["matrix/features/name"][:]
    return {n.decode() for n, t in zip(nm, ft) if t == b"Gene Expression"}


def preflight(root, repo, desium, acc_csv, patients):
    ok = True
    print("--- packages ---")
    for m in ["numpy", "pandas", "scipy", "sklearn", "anndata", "h5py"]:
        try:
            mod = __import__(m)
            print(f"  OK   {m:10s} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"  FAIL {m:10s} {e}"); ok = False
    print("--- repo ---")
    sys.path.insert(0, str(repo))
    try:
        import importlib
        for mod in ("metabocast.metabocast", "metabocast.validation"):
            importlib.import_module(mod)
        head = os.popen(f'git -C "{repo}" rev-parse --short HEAD 2>/dev/null').read().strip()
        print(f"  OK   metabocast imports (HEAD {head or 'unknown'})")
    except Exception as e:
        print(f"  FAIL cannot import metabocast: {e}"); ok = False
    print("--- inputs ---")
    files = [desium / f"{s}.h5ad" for s in DESIUM]
    for pid in patients:
        files += [root / f"{pid}-expr_{x}" for x in
                  ("counts.mtx.gz", "features.txt", "barcodes.txt", "meta.csv")]
    files += [root / CYT_H5, root / CYT_SPATIAL, root / VIS_H5, root / VIS_SPATIAL]
    total = 0
    for f in files:
        if f.exists():
            total += f.stat().st_size
            print(f"  OK   {f.name[:52]:52s} {f.stat().st_size / 1e6:8.1f} MB")
        else:
            print(f"  FAIL {f.name[:52]:52s} MISSING"); ok = False
    print(f"  {'OK  ' if acc_csv.exists() else 'WARN'} {acc_csv.name}")
    print(f"  total input {total / 1e9:.2f} GB")
    if ok:
        print("--- gene space ---")
        sets = []
        for s in DESIUM:
            with h5py.File(desium / f"{s}.h5ad", "r") as h:
                sets.append({x.decode() if isinstance(x, bytes) else x for x in h["var/_index"][:]})
        for pid in patients:
            sets.append({l.strip() for l in open(root / f"{pid}-expr_features.txt")})
        hcc_only = len(set.intersection(*sets))
        sets.append(gene_symbols_from_h5(root / CYT_H5))
        sets.append(gene_symbols_from_h5(root / VIS_H5))
        shared = set.intersection(*sets)
        print(f"  DESIUM + HCC only        : {hcc_only}")
        print(f"  DESIUM + HCC + both GBM  : {len(shared)}   <- used here")
        print(f"  difference               : {hcc_only - len(shared)} genes")
        print(f"  pooled training matrix ~{14816 * len(shared) * 4 / 1e9:.1f} GB dense")
        print("--- protein panel ---")
        _, _, nm, ft, iso = read_10x_h5(root / CYT_H5)
        ab = ft == "Antibody Capture"
        isoflag = is_isotype(nm, iso)
        print(f"  {int(ab.sum())} antibody features, {int((ab & isoflag).sum())} isotype controls, "
              f"{int((ab & ~isoflag).sum())} informative")
        print(f"  markers: {', '.join(unique_names(list(nm[ab & ~isoflag])))}")
    print("\n" + ("PREFLIGHT PASSED - safe to run the full command" if ok else "PREFLIGHT FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--desium", type=Path, default=None)
    ap.add_argument("--accuracy", type=Path, default=None,
                    help="CSV with mz, within_section_r and leave_one_section_out_r per DESIUM ion; default: "
                         "reproduce/results/desium_per_ion_accuracy.csv")
    ap.add_argument("--patients", nargs="+", default=PATIENTS)
    ap.add_argument("--n-pcs", type=int, default=100)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--niches", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    repo = (args.repo or Path(__file__).resolve().parents[2]).expanduser().resolve()
    desium = (args.desium or root / "DESIUM/Correlation_NMF_analysis/data").expanduser().resolve()
    acc_csv = (args.accuracy or repo / "reproduce/results/desium_per_ion_accuracy.csv").expanduser()
    if args.check:
        return preflight(root, repo, desium, acc_csv, args.patients)
    out = args.out.expanduser().resolve()
    if out.exists() and not args.force:
        ap.error(f"{out} already exists; choose a new path (or --force)")
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    import anndata as ad
    import sklearn

    log(f"python {platform.python_version()} | numpy {np.__version__} | sklearn {sklearn.__version__}")
    log(f"out {out}")

    # ------------------------------------------------------------ gene space
    log("resolving the shared gene space (DESIUM + HCC + both GBM sections) ...")
    def _slim(path):
        """Keep only what training needs (log1p expression as sparse float32, MSI, m/z, coordinates), so the
        seven DESIUM sections fit in memory together; the full files also carry images and float64 layers."""
        import scipy.sparse as sp_
        a = ad.read_h5ad(path)
        x = a.layers["log1p"] if "log1p" in a.layers else a.X
        b = ad.AnnData(X=sp_.csr_matrix(x, dtype=np.float32))
        b.var_names = list(a.var_names); b.obs_names = list(a.obs_names); b.layers["log1p"] = b.X
        b.uns["msi"] = np.asarray(a.uns["msi"], np.float32); b.uns["mz_features"] = np.asarray(a.uns["mz_features"], float)
        b.obsm["spatial"] = np.asarray(a.obsm["spatial"], float)
        del a
        return b
    D = {s: _slim(desium / f"{s}.h5ad") for s in DESIUM}
    mz = np.asarray(D[DESIUM[0]].uns["mz_features"], float)
    for s in DESIUM[1:]:
        if not np.allclose(np.asarray(D[s].uns["mz_features"], float), mz, atol=1e-6):
            ap.error(f"{s} is not on the shared m/z axis")
    sets = [set(a.var_names) for a in D.values()]
    sets += [{l.strip() for l in open(root / f"{pid}-expr_features.txt")} for pid in args.patients]
    sets += [gene_symbols_from_h5(root / CYT_H5), gene_symbols_from_h5(root / VIS_H5)]
    shared = sorted(set.intersection(*sets))
    log(f"  {len(shared)} shared genes, {len(mz)} m/z channels")

    # ------------------------------------------------------------ train once
    def dense_log1p(a):
        v = a[:, shared].layers["log1p"]
        return np.asarray(v.todense() if hasattr(v, "todense") else v, dtype=np.float32)

    from metabocast import MetaboCast
    from metabocast.annotation import ADDUCTS as REF_ADDUCTS, candidate_labels, neutral_mass
    MASS.update({k: neutral_mass(v) for k, v in CURATED.items()}); ADDUCTS.update(REF_ADDUCTS)
    slim = []
    for s_ in DESIUM:
        a = D[s_]
        b = ad.AnnData(X=dense_log1p(a)); b.var_names = shared; b.layers["log1p"] = b.X
        b.uns["msi"] = np.asarray(a.uns["msi"], np.float32); b.uns["mz_features"] = mz
        b.obsm["spatial"] = np.asarray(a.obsm["spatial"], float); slim.append(b)
    del D
    model = MetaboCast(n_hvg=None, n_pcs=args.n_pcs, k=args.k).fit(slim)
    del slim
    log(f"MetaboCast fitted: alpha={model.alpha_} ({model.alpha_selection_}), "
        f"target_transform={model.target_transform}, {len(model.genes_)} genes")

    def predict_query(Xq, coords, groups):
        q = ad.AnnData(X=Xq); q.var_names = shared; q.layers["log1p"] = q.X
        q.obsm["spatial"] = np.asarray(coords, float); q.obs["grp"] = np.asarray(groups).astype(str)
        return model.predict_metabolome(q, section_key="grp").astype(np.float32)

    # ------------------------------------------------------------ DESIUM accuracy tiers + annotation
    if not acc_csv.exists():
        ap.error(f"accuracy table missing: {acc_csv}")
    acc = pd.read_csv(acc_csv)
    amz = acc.mz.to_numpy(float)
    idx = [int(np.argmin(np.abs(amz - v))) for v in mz]
    within = np.array([acc.within_section_r.iloc[i] if abs(amz[i] - v) <= 1e-3 else np.nan for i, v in zip(idx, mz)], float)
    transfer = np.array([acc.leave_one_section_out_r.iloc[i] if abs(amz[i] - v) <= 1e-3 else np.nan for i, v in zip(idx, mz)], float)
    tiers = np.array([tier(w, t) for w, t in zip(within, transfer)], dtype=object)
    ann = np.array(candidate_labels(mz), dtype=object)    # every candidate of the reference, nearest first
    supported = np.where((ann != "") & np.isin(tiers, ["transfer-supported", "within-section only"]))[0]
    transfer_ions = np.where(tiers == "transfer-supported")[0]
    log(f"{int((ann != '').sum())} annotated ions: {int(((ann != '') & (tiers == 'transfer-supported')).sum())} transfer-supported, "
        f"{int(((ann != '') & (tiers == 'within-section only')).sum())} within-section only; "
        f"{len(transfer_ions)} transfer-supported channels in all")

    def acc_cols(i):
        return {"within_section_r": round(float(within[i]), 3) if np.isfinite(within[i]) else None,
                "transfer_r": round(float(transfer[i]), 3) if np.isfinite(transfer[i]) else None,
                "tier": tiers[i]}

    inputs = {}

    # ------------------------------------------------------------ HCC
    means = {}
    for pid in args.patients:
        t0 = time.time()
        genes = [l.strip() for l in open(root / f"{pid}-expr_features.txt")]
        gmap = {g: i for i, g in enumerate(genes)}
        gi = np.array([gmap[g] for g in shared])
        counts = root / f"{pid}-expr_counts.mtx.gz"
        M = scipy.io.mmread(gzip.open(counts)).tocsc()
        tot = np.asarray(M.sum(0)).ravel() + 1e-8
        Xq = M[gi, :].T.tocsr().astype(np.float32)
        Xq = Xq.multiply(1.0 / tot[:, None]).multiply(1e4).tocsr()
        Xq.data = np.log1p(Xq.data)
        meta = pd.read_csv(root / f"{pid}-expr_meta.csv", index_col=0)
        bc = [l.strip() for l in open(root / f"{pid}-expr_barcodes.txt")]
        meta = meta.reindex(bc)
        if len(meta) != Xq.shape[0] or meta["sample.ident"].isna().any():
            ap.error(f"{pid}: barcode/matrix misalignment")
        region = meta["sample.ident"].astype(str).str.replace(pid, "", regex=False).str.lstrip("-").values
        coords = meta[["imagerow", "imagecol"]].values.astype(float)
        pred = predict_query(Xq, coords, region)
        np.savez_compressed(out / f"{pid}_pred.npz", pred=pred, mz=mz,
                            region=region.astype("U8"), coords=coords, barcodes=np.array(bc, "U32"))
        means[pid] = {r: pred[region == r].mean(0) for r in sorted(set(region))}
        inputs[pid] = {"counts_sha256": sha256(counts), "n_spots": int(pred.shape[0]),
                       "regions": {r: int((region == r).sum()) for r in sorted(set(region))}}
        if pid == "HCC-2":
            h2p, h2r = pred, region
        else:
            del pred
        del M, Xq
        log(f"  {pid}: {inputs[pid]['n_spots']} spots {inputs[pid]['regions']} in {time.time()-t0:.0f}s")

    if "HCC-2" in args.patients and {"N", "P"} <= set(h2r):
        A, B = h2p[h2r == "N"], h2p[h2r == "P"]
        delta = B.mean(0) - A.mean(0)
        s8 = pd.DataFrame({
            "mz": np.round(mz, 4), "annotation": ann, "within_section_r": np.round(within, 3),
            "transfer_r": np.round(transfer, 3), "tier": tiers,
            "mean_N": np.round(A.mean(0), 4), "sd_N": np.round(A.std(0), 4),
            "mean_P": np.round(B.mean(0), 4), "sd_P": np.round(B.std(0), 4),
            "delta_NtoP_log": np.round(delta, 4),
            "n_spots_N": int((h2r == "N").sum()), "n_spots_P": int((h2r == "P").sum()),
            "ambiguous_annotation": [bool(a) and ";" in a for a in ann]})
        s8.to_csv(out / "S8_full_all_ions.csv", index=False)
        s8.iloc[supported].sort_values("delta_NtoP_log", ascending=False).to_csv(out / "S8_supported_annotated.csv", index=False)
        log(f"  S8 rebuilt: {len(supported)} annotated ions in the two upper tiers")

    cols, rows = {}, []
    for pid in ["HCC-1", "HCC-2", "HCC-3", "HCC-4"]:
        if pid in means and {"N", "T"} <= set(means[pid]):
            cols[f"{pid} N->T"] = {n: float(means[pid]["T"][i] - means[pid]["N"][i])
                                   for n in SIGNATURE if (i := ion_index(mz, n)) is not None}
    if "HCC-2" in means and {"N", "P"} <= set(means["HCC-2"]):
        cols["HCC-2 N->P"] = {n: float(means["HCC-2"]["P"][i] - means["HCC-2"]["N"][i])
                              for n in SIGNATURE if (i := ion_index(mz, n)) is not None}
    for n in SIGNATURE:
        i = ion_index(mz, n)
        if i is None:
            continue
        rows.append({"metabolite": n, "mz": round(float(mz[i]), 4), **acc_cols(i),
                     "ambiguous_annotation": ";" in ann[i],
                     **{c: round(cols[c][n], 6) for c in cols}})
    s9 = pd.DataFrame(rows)
    s9.to_csv(out / "S9_and_Fig7c_source.csv", index=False)
    log("\n" + s9.to_string(index=False))

    # ------------------------------------------------------------ GBM helper
    def predict_10x(h5_path, spatial_path, label):
        X, bc, name, ftype, iso = read_10x_h5(h5_path)
        pos = read_positions(spatial_path)
        pos["barcode"] = pos["barcode"].astype(str)
        pos = pos.set_index("barcode").reindex(bc)
        keep = pos["in_tissue"].fillna(0).astype(int).values == 1
        if keep.sum() < 100:
            ap.error(f"{label}: only {int(keep.sum())} in-tissue spots matched - check the spatial file")
        coords = pos[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(float)[keep]
        is_gene = ftype == "Gene Expression"
        gname = name[is_gene]
        first = {}
        for j, g in enumerate(gname):
            first.setdefault(g, j)
        missing = [g for g in shared if g not in first]
        if missing:
            ap.error(f"{label}: {len(missing)} shared genes absent - gene space is inconsistent")
        gi = np.array([first[g] for g in shared])
        G = X[is_gene, :][:, keep]
        tot = np.asarray(G.sum(0)).ravel() + 1e-8
        Xq = G[gi, :].T.tocsr().astype(np.float32)
        Xq = Xq.multiply(1.0 / tot[:, None]).multiply(1e4).tocsr()
        Xq.data = np.log1p(Xq.data)
        pred = predict_query(Xq, coords, np.zeros(coords.shape[0]))
        log(f"  {label}: {pred.shape[0]} in-tissue spots predicted")
        return pred, coords, np.array(bc)[keep], X, name, ftype, iso, keep

    # ------------------------------------------------------------ GBM CytAssist -> S10 / Fig 7d
    log("GBM CytAssist (protein co-localisation) ...")
    predC, coordC, bcC, XC, nameC, ftypeC, isoC, keepC = predict_10x(
        root / CYT_H5, root / CYT_SPATIAL, "CytAssist")
    ab = ftypeC == "Antibody Capture"
    informative = ab & ~is_isotype(nameC, isoC)
    n_iso = int((ab & is_isotype(nameC, isoC)).sum())
    if int(informative.sum()) != 31:
        log(f"  NOTE: {int(informative.sum())} informative markers (the published table reports 31)")
    pnames = unique_names(list(nameC[informative]))
    Prot = np.asarray(XC[informative, :][:, keepC].todense()).T.astype(np.float64)
    Prot = np.log1p(Prot)
    panel = [(n, ion_index(mz, n)) for n in GBM_PANEL]
    panel = [(n, i) for n, i in panel if i is not None]
    Met = predC[:, [i for _, i in panel]].astype(np.float64)
    S10 = spearman_matrix(Met, Prot)
    s10 = pd.DataFrame(np.round(S10, 3), index=[n for n, _ in panel], columns=pnames)
    s10.insert(0, "within_section_r", [acc_cols(i)["within_section_r"] for _, i in panel])
    s10.insert(1, "transfer_r", [acc_cols(i)["transfer_r"] for _, i in panel])
    s10.insert(2, "tier", [tiers[i] for _, i in panel])
    s10.insert(3, "ambiguous_annotation", [";" in ann[i] for _, i in panel])
    s10.insert(0, "mz", [round(float(mz[i]), 4) for _, i in panel])
    s10.index.name = "predicted_metabolite"
    s10.to_csv(out / "S10_metabolite_x_protein.csv")
    s10.drop(columns=["mz", "within_section_r", "transfer_r", "tier", "ambiguous_annotation"]).stack().rename("spearman") \
       .reset_index().to_csv(out / "S10_long.csv", index=False)
    np.savez_compressed(out / "GBM_CytAssist_pred.npz", pred=predC, mz=mz, coords=coordC,
                        barcodes=bcC.astype("U32"), protein=Prot,
                        protein_names=np.array(pnames, "U24"))
    log(f"  S10: {len(panel)} metabolites x {len(pnames)} informative markers "
        f"({int(ab.sum())} antibodies, {n_iso} isotype controls dropped)")
    log("  strongest |rho|: " + ", ".join(
        f"{s10.index[i]}~{pnames[j]} {S10[i, j]:+.2f}"
        for i, j in zip(*np.unravel_index(np.argsort(-np.abs(S10), axis=None)[:4], S10.shape))))

    # ------------------------------------------------------------ GBM Visium -> Fig 7e / 7f
    log("GBM Parent Visium (metabolic niches) ...")
    predV, coordV, bcV, *_ = predict_10x(root / VIS_H5, root / VIS_SPATIAL, "Parent Visium")
    Z = predV[:, transfer_ions].astype(np.float64)        # niches from the channels that transfer
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    niche = KMeans(args.niches, n_init=10, random_state=args.seed).fit_predict(Z)
    pd.DataFrame({"barcode": bcV, "pxl_row": coordV[:, 0], "pxl_col": coordV[:, 1],
                  "niche": [f"n{n}" for n in niche]}).to_csv(out / "Fig7e_niches.csv", index=False)
    prof = []
    for n, i in panel:
        v = predV[:, i].astype(np.float64)
        z = (v - v.mean()) / (v.std() + 1e-9)
        prof.append({"metabolite": n, "mz": round(float(mz[i]), 4), **acc_cols(i),
                     "ambiguous_annotation": ";" in ann[i],
                     **{f"n{k}": round(float(z[niche == k].mean()), 3) for k in range(args.niches)}})
    pd.DataFrame(prof).to_csv(out / "Fig7f_metabolite_by_niche.csv", index=False)
    np.savez_compressed(out / "GBM_Visium_pred.npz", pred=predV, mz=mz, coords=coordV,
                        barcodes=bcV.astype("U32"), niche=niche,
                        niche_ion_index=transfer_ions)
    log(f"  niches: {dict(zip(*np.unique([f'n{n}' for n in niche], return_counts=True)))}")
    log(f"  clustered on {len(transfer_ions)} transfer-supported channels")

    prov = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": Path(__file__).name, "script_sha256": sha256(Path(__file__).resolve()),
            "root": root.name,                                   # folder name only, no machine-specific paths
            "git_head": os.popen(f'git -C "{repo}" rev-parse HEAD 2>/dev/null').read().strip() or None,
            "git_uncommitted_changes": bool(os.popen(f'git -C "{repo}" status --porcelain 2>/dev/null').read().strip()),
            "n_shared_genes": len(shared), "n_mz": int(len(mz)),
            "gene_space": "intersection of 7 DESIUM sections, 5 HCC patients and both GBM sections",
            "alpha": float(model.alpha_), "alpha_selection": model.alpha_selection_,
            "target_transform": model.target_transform, "n_pcs": args.n_pcs, "k": args.k, "niches": args.niches,
            "tiers": {"transfer-supported": f"DESIUM leave-one-section-out r >= {TRANSFER_MIN}",
                      "within-section only": f"DESIUM within-section r >= {WITHIN_MIN}, transfer below",
                      "accuracy_table": str(acc_csv)},
            "niche_features": "all transfer-supported channels, z-scored", "seed": args.seed,
            "graph_operator": "D^-1/2 A (metabocast._norm_adj)",
            "training": "all 7 DESIUM sections pooled, no held-out fold",
            "hvg_selection": "none - PCA on all shared genes",
            "patients": inputs,
            "gbm": {"cytassist_sha256": sha256(root / CYT_H5),
                    "visium_sha256": sha256(root / VIS_H5),
                    "n_antibody_features": int(ab.sum()),
                    "n_informative_markers": len(pnames),
                    "isotype_controls_dropped": n_iso},
            "n_annotated": int((ann != "").sum()), "n_supported_annotated": int(len(supported)),
            "n_transfer_supported_channels": int(len(transfer_ions)),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                         "anndata": ad.__version__, "pandas": pd.__version__, "h5py": h5py.__version__}}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    log(f"\nwrote {out}")
    for f in sorted(out.iterdir()):
        log(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
    log("done; derive_figure_sources.py builds the figure and table sources from this folder (the .npz files are large and are not part of reproduce/results).")


if __name__ == "__main__":
    main()
