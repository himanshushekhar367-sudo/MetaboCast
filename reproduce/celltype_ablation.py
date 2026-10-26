"""REDCAT-inspired cell-type-aware ablation (DESIUM LOSO, detection-matched median Spearman).

Compares, under the identical leave-one-sample-out protocol:
  B  diffusion            (baseline MetaboCast)
  C  diffusion + KEGG     (human metabolic-map prior)
  D  diffusion + cell-type (per-spot lineage signatures via extra_key -> [C, A C])
  E  diffusion + cell-type neighbourhood ([C, A C] block -> up to A^2 C)

Cell-type signatures are reference-free lineage marker-set scores (metabocast.add_celltype_features),
the transcriptome-only approximation of REDCAT's finding that metabolism is cell-type-specific.
Reports overall and per-class (lipid / redox / nucleotide / polar).

Usage:  python reproduce/celltype_ablation.py --data <DESIUM_DIR> --out reproduce/results/ablations
        [--annotation reproduce/results/desium_ion_annotation.csv] [--summarise-only]
The DESIUM directory must contain the 7 paired .h5ad sections (transcriptome + uns['msi'] + uns['mz_features']).
Classes come from reproduce/annotate_ions.py (negative-mode candidates; channels whose candidates span more than
one class are left out of the class means). Writes celltype_ablation_persection.csv (median per held-out
section), celltype_ablation_by_class.csv (mean per-ion Spearman by class) and celltype_ablation_per_ion.npz
(per-ion Spearman per held-out section and condition). --summarise-only rebuilds the class table from the saved
per-ion file (for example after the annotation changes) without refitting.
"""
import argparse, sys, os, gc, warnings, numpy as np, anndata as ad, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # this repository's metabocast
from scipy.stats import spearmanr
from metabocast import MetaboCast, add_celltype_features
from metabocast.metabocast import _norm_adj

ap = argparse.ArgumentParser(description="Cell-type-aware feature ablation (DESIUM leave-one-section-out)")
ap.add_argument("--data", default=os.environ.get("DESIUM_DIR", "data/DESIUM"))
ap.add_argument("--annotation", default=os.path.join(HERE, "results", "desium_ion_annotation.csv"))
ap.add_argument("--out", default=".")
ap.add_argument("--summarise-only", action="store_true")
args = ap.parse_args()
DDIR = args.data
os.makedirs(args.out, exist_ok=True)
SECS = ["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
ANN = pd.read_csv(args.annotation)
has = ANN["n_candidates"] > 0
cls = ANN["class"].where(has & (ANN["class"] != "ambiguous"), "").fillna("").astype(str).to_numpy(dtype=object)
anno = ANN["annotation"].where(has, "").fillna("").astype(str).to_numpy(dtype=object)   # pandas >= 3 keeps NaN under astype(str)
CONDS = {"B_diffusion": dict(use_kegg=False, extra_key=None),
         "C_+KEGG": dict(use_kegg=True, extra_key=None),
         "D_+celltype": dict(use_kegg=False, extra_key="celltype"),
         "E_+celltype_nbhd": dict(use_kegg=False, extra_key="celltype_nbhd")}


def by_class(res):
    LIPID = {"fatty acid","phospholipid","sphingolipid","glycerolipid","lysophospholipid","lipid headgroup","acylcarnitine","sterol"}
    NUCL = {"nucleobase","nucleoside","nucleotide"}; POLAR = {"amino acid","TCA acid","sugar","sugar phosphate","carboxylic acid"}
    single = (ANN["n_candidates"] == 1).to_numpy()                  # name-based group: one candidate only
    redox = np.array([s and a.lower().startswith(("glutathione", "gssg", "ascorbate", "urate", "taurine"))
                      for s, a in zip(single, anno)]) | (cls == "peptide/redox")
    groups = {"overall": None, "lipid": np.array([c in LIPID for c in cls]), "redox": redox,
              "nucleotide": np.array([c in NUCL for c in cls]), "polar": np.array([c in POLAR for c in cls])}
    print("\nclass          n   " + "".join(f"{k:>18s}" for k in CONDS))
    rows = []
    for g, mask in groups.items():
        n = "det" if mask is None else str(int(mask.sum()))
        vals = [float(np.nanmean(res[name][0]) if mask is None else np.nanmean(res[name][1][:, mask])) for name in CONDS]
        print(f"{g:12s} {n:>4s} " + "".join(f"{v:>+18.4f}" for v in vals))
        if mask is not None:
            rows.append([g, int(mask.sum())] + vals)
    pd.DataFrame(rows, columns=["class", "n_ions"] + list(CONDS)).round(4).to_csv(
        os.path.join(args.out, "celltype_ablation_by_class.csv"), index=False)


if args.summarise_only:
    Z = np.load(os.path.join(args.out, "celltype_ablation_per_ion.npz"))
    per = pd.read_csv(os.path.join(args.out, "celltype_ablation_persection.csv"), index_col=0)
    if len(ANN) != len(Z["mz"]) or not np.allclose(ANN["mz"].to_numpy(float), Z["mz"], atol=1e-3):
        sys.exit("annotation table is not on the DESIUM m/z axis")
    by_class({name: (per[name].to_numpy(), Z[name]) for name in CONDS})
    sys.exit(0)

secs = {}
for s in SECS:
    a = ad.read_h5ad(f"{DDIR}/{s}.h5ad")
    ct = add_celltype_features(a)
    L = np.asarray(a.layers["log1p"].todense() if hasattr(a.layers["log1p"], "todense") else a.layers["log1p"], np.float32)
    top = np.argsort(L.var(0))[-4000:]
    b = ad.AnnData(X=np.ascontiguousarray(L[:, top]))
    b.var_names = list(np.asarray(a.var_names)[top]); b.obs_names = list(a.obs_names); b.layers["log1p"] = b.X
    xy = np.asarray(a.obsm["spatial"], float); b.obsm["spatial"] = xy; b.obsm["celltype"] = ct
    A = _norm_adj(xy, None, 6); b.obsm["celltype_nbhd"] = np.hstack([ct, (A @ ct).astype(np.float32)]).astype(np.float32)
    b.uns["msi"] = np.asarray(a.uns["msi"], np.float32); b.uns["mz_features"] = a.uns["mz_features"]
    secs[s] = b; del a, L; gc.collect()
AXIS = np.asarray(secs[SECS[0]].uns["mz_features"], float)
if len(ANN) != len(AXIS) or not np.allclose(ANN["mz"].to_numpy(float), AXIS, atol=1e-3):
    sys.exit("annotation table is not on the DESIUM m/z axis")

def perion(true, pred, det):
    r = np.full(true.shape[1], np.nan)
    for k in np.where(det)[0]:
        if true[:, k].std() > 1e-9 and pred[:, k].std() > 1e-9:
            r[k] = spearmanr(true[:, k], pred[:, k])[0]
    return r

res = {}
for name, kw in CONDS.items():
    meds, ions = [], []
    for s in SECS:
        tr = [secs[x] for x in SECS if x != s]; te = secs[s]
        m = MetaboCast(n_hvg=2000, **kw); m.fit(tr)
        pred = m.predict_metabolome(te.copy())
        true = np.log1p(np.asarray(te.uns["msi"], float)); det = (np.asarray(te.uns["msi"], float) > 0).mean(0) > 0.20
        r = perion(true, pred, det); meds.append(np.nanmedian(r[det])); ions.append(r)
        del m, pred; gc.collect()
    res[name] = (np.array(meds), np.vstack(ions))
    print(f"{name:18s} overall {np.nanmean(res[name][0]):+.4f}", flush=True)
    pd.DataFrame({k: res[k][0] for k in res}, index=SECS).to_csv(os.path.join(args.out, "celltype_ablation_persection.csv"))

pd.DataFrame({k: res[k][0] for k in res}, index=SECS).to_csv(os.path.join(args.out, "celltype_ablation_persection.csv"))
by_class(res)
np.savez_compressed(os.path.join(args.out, "celltype_ablation_per_ion.npz"),
                    mz=AXIS, sections=np.array(SECS),
                    **{name: res[name][1].astype(np.float32) for name in CONDS})
