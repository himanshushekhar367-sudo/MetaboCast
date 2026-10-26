"""Warburg substrate-to-product flux test (flux_baselines table, lower block; reproduce/manifest.py).

Question: does a transcriptome-only MetaboCast prediction recover a flux contrast, not only a static pool?
The Warburg readout z[lactate, m/z 89.025] - z[hexose, m/z 179.055] (both [M-H]-; the hexose channel stands for
glucose and its C6H12O6 isomers) is used as a one-dimensional score to separate tumour from stroma spots. Three feature sources are compared:
  (1) ENZYME scores: glycolysis and TCA-cycle KEGG gene-set activity (mean z-scored log1p expression), two features;
  (2) MEASURED metabolites (needs MSI; the reference);
  (3) leave-one-section-out MetaboCast-PREDICTED metabolites (what a transcriptome-only user obtains).
Classifier: 5-fold cross-validated logistic regression, ROC-AUC, mean over sections.

Default: tumour versus stroma spots (pathology annotations "Tumor" and "Stroma"; other annotations excluded),
scored in sections with at least 25 spots of each, i.e. the four breast sections (the lung sections carry no
stroma annotation). Models for the predicted arm are trained on the six other sections.
Sensitivity analysis: --labels rest scores tumour against all other spots in all seven sections.

Run: python reproduce/flux_validation.py [--labels stroma|rest] [--no-predicted] --out flux.csv
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # data and work locations; see reproduce/paths.py
import numpy as np, anndata as ad, pandas as pd, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from metabocast import MetaboCast
from metabocast.metabocast import _default_kegg_gmt
warnings.filterwarnings("ignore")

DATA = os.environ.get("DESIUM_DIR", P.DESIUM)
SAMPLES = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
LACTATE, GLUCOSE = 90.0317, 180.0634             # neutral monoisotopic masses
DEPROTONATED = -1.0073                            # [M-H]-
MIN_SPOTS = 25


def ions(mz, neutral, tol=0.01):
    """Indices of the [M-H]- ion (lactate m/z 89.025, glucose m/z 179.055)."""
    return [i for i, v in enumerate(mz) if abs(v - (neutral + DEPROTONATED)) <= tol]


def zc(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-8)


def warburg(Y, mz):
    li, gi = ions(mz, LACTATE), ions(mz, GLUCOSE)
    if not li or not gi:
        raise ValueError("lactate or glucose ion missing from the m/z axis")
    return zc(Y[:, li].mean(1)) - zc(Y[:, gi].mean(1))


def gene_sets():
    sets = {}
    for ln in open(_default_kegg_gmt()):
        p = ln.rstrip("\n").split("\t")
        sets[p[0]] = [g for g in p[2:] if g]
    gly = next(v for k, v in sets.items() if k.startswith("Glycolysis"))
    tca = next(v for k, v in sets.items() if k.startswith("Citrate cycle"))
    return gly, tca


def enzyme_scores(X, names, gly, tca):
    pos = {g: i for i, g in enumerate(names)}
    def score(gs):
        idx = [pos[g] for g in gs if g in pos]
        S = X[:, idx]
        return ((S - S.mean(0)) / (S.std(0) + 1e-8)).mean(1)
    return np.c_[score(gly), score(tca)]


def labels(a, mode):
    ann = a.obs["annotation"].astype(str).to_numpy() if "annotation" in a.obs else None
    if ann is None:
        return None, None
    tum = ann == "Tumor"
    keep = (tum | (ann == "Stroma")) if mode == "stroma" else np.ones(len(ann), bool)
    return tum, keep


def auc(F, y):
    F = np.asarray(F, float).reshape(len(y), -1)
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=500), F, y, cv=5, scoring="roc_auc")))


def _slim(a):
    from scipy.sparse import csr_matrix
    x = a.layers["log1p"] if "log1p" in a.layers else a.X
    b = ad.AnnData(X=csr_matrix(x, dtype=np.float32), obs=a.obs[["annotation"]].copy() if "annotation" in a.obs else None)
    b.var_names = list(a.var_names); b.layers["log1p"] = b.X
    b.uns["msi"] = np.asarray(a.uns["msi"], np.float32); b.uns["mz_features"] = np.asarray(a.uns["mz_features"], float)
    b.obsm["spatial"] = np.asarray(a.obsm["spatial"], float)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", choices=["stroma", "rest"], default="stroma")
    ap.add_argument("--no-predicted", action="store_true", help="skip the model arm (measured and enzyme only)")
    ap.add_argument("--out", default=None, help="CSV of per-section AUCs")
    args = ap.parse_args()
    secs = {s: _slim(ad.read_h5ad(f"{DATA}/{s}.h5ad")) for s in SAMPLES}   # each section read once
    gly, tca = gene_sets()
    rows = []
    for s in SAMPLES:
        a = secs[s]
        tum, keep = labels(a, args.labels)
        if tum is None or tum[keep].sum() < MIN_SPOTS or (~tum[keep]).sum() < MIN_SPOTS:
            print(f"  {s:16s} skipped (fewer than {MIN_SPOTS} tumour or comparison spots)", flush=True); continue
        y = tum[keep].astype(int)
        mz = np.asarray(a.uns["mz_features"], float)
        meas = warburg(np.log1p(np.asarray(a.uns["msi"], float)), mz)
        X = np.asarray(a.layers["log1p"].todense(), np.float32)
        enz = enzyme_scores(np.asarray(X, float), list(a.var_names), gly, tca)
        row = dict(section=s, n_tumour=int(y.sum()), n_other=int((1 - y).sum()), measured=auc(meas[keep], y), enzyme=auc(enz[keep], y))
        if not args.no_predicted:
            m = MetaboCast(n_hvg=3000).fit([secs[o] for o in SAMPLES if o != s])        # leave-one-section-out
            pred = m.predict_metabolome(a.copy())
            row["predicted"] = auc(warburg(pred, np.asarray(m.mz_, float))[keep], y)
            del m, pred
        rows.append(row)
        print(f"  {s:16s} AUC measured={row['measured']:.3f}  predicted={row.get('predicted', float('nan')):.3f}  enzyme-only={row['enzyme']:.3f}", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        print(f"\n=== Warburg flux discrimination ({args.labels} labels; 5-fold CV ROC-AUC, mean over {len(df)} sections) ===")
        print(f"  measured metabolites : {df.measured.mean():.3f}")
        if "predicted" in df:
            print(f"  MetaboCast predicted: {df.predicted.mean():.3f}")
        print(f"  enzyme-only (KEGG)   : {df.enzyme.mean():.3f}")
        if args.out:
            df.assign(labels=args.labels).to_csv(args.out, index=False)
            print("saved", args.out)


if __name__ == "__main__":
    main()
