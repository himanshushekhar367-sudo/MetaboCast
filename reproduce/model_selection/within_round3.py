"""Round 2 (within-section): training-target transforms and representation ensembles on top of the best round-1
settings. Same protocol as within_screen.py. Target transforms only touch TRAINING labels:
  yR = per-ion rank-Gaussian transform of training spots (Spearman-aligned objective)
  yS = training labels averaged over a kNN graph built on training spots only (label denoising, no test leakage)
Scores are always computed against the untouched measured log1p intensities."""
import sys, os, json, time, argparse, warnings
import numpy as np, h5py
from collections import defaultdict
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, diags, eye
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_lib import *
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--sections", nargs="+", default=DESIUM)
ap.add_argument("--targets", default="y,yR,yS")
ap.add_argument("--sets", default="pca,nmf,pca+enz")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def path_of(s):
    return f"{SMA_DIR}/{s}.h5ad" if s.startswith("SMA_") else f"{DESIUM_DIR}/{s}.h5ad"


def pca_rep(Xtr, Xall, k):
    k = min(k, Xtr.shape[0], Xtr.shape[1])
    return PCA(k, svd_solver="randomized", random_state=0).fit(Xtr).transform(Xall).astype(np.float32)


def mean_adj(c, k=6):
    n = len(c); _, idx = cKDTree(c).query(c, k=min(k + 1, n))
    r = np.repeat(np.arange(n), idx.shape[1] - 1); cc = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(r)), (r, cc)), shape=(n, n)); A = ((A + A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1 / np.asarray(A.sum(1)).ravel()).astype(np.float32)) @ A


def nested_ridge(Ftr, Ytr_fit, Ytr_eval, ib, Fte):
    """Inner leave-one-block-out alpha choice (global and per ion); fit targets may be transformed, inner
    validation is scored against Ytr_eval (training labels only)."""
    sc = []
    for b in np.unique(ib):
        itr, iva = ib != b, ib == b
        rp = RidgePath(Ftr[itr], Ytr_fit[itr])
        sc.append(np.vstack([col_pearson(p, Ytr_eval[iva]) for p in rp.predict_path(Ftr[iva], ALPHAS)]))
    gi, pi = select_alpha(np.nanmean(np.stack(sc), 0))
    rp = RidgePath(Ftr, Ytr_fit)
    return rp.predict(Fte, ALPHAS[gi]), rp.predict(Fte, ALPHAS[pi]), float(ALPHAS[gi]), rp.predict(Fte, 200.0)


def run_section(sec):
    t0 = time.time()
    with h5py.File(path_of(sec), "r") as f:
        x = read_matrix(f, "layers/log1p"); msi = f["uns/msi"][:].astype(np.float32); co = f["obsm/spatial"][:].astype(float)
    y = np.log1p(msi); det = (y > 0).mean(0) > .2
    top = np.where(det)[0][np.argsort(y[:, det].var(0))[-1000:]]; y = y[:, top]
    A = norm_adj(co); folds = KMeans(5, n_init=3, random_state=0).fit_predict(co)
    with h5py.File(path_of(sec), "r") as f:
        names = var_names(f)
    upper = np.mean([n == n.upper() for n in names[:2000]]) > 0.5
    names_k = names if upper else [n.upper() for n in names]
    kg = set().union(*load_gmt().values())
    enz = np.array([i for i, g in enumerate(names_k) if g in kg])
    preds = defaultdict(lambda: np.full(y.shape, np.nan, np.float32)); al = defaultdict(list)
    for fold in range(5):
        tr, te = folds != fold, folds == fold
        hvg = np.argsort(x[tr].var(0))[-3000:]; xh = x[:, hvg]
        T_P = diffuse(pca_rep(xh[tr], xh, 100), A, 2)
        T_N = diffuse(nmf_rep(xh[tr], xh, 100), A, 2) if "nmf" in args.sets.split(",") else None
        xz = (xh - xh.mean(0)) / (xh.std(0) + 1e-6)          # section-wise gene z-scores (as in LOSO "secz")
        T_Z = diffuse(pca_rep(xz[tr], xz, 100), A, 2)
        E50 = pca_rep(x[tr][:, enz], x[:, enz], 50)
        FS = {"pca": T_P, "secz": T_Z, "pca+enz": block_scale([T_P, diffuse(E50, A, 2)], tr)}
        if T_N is not None: FS["nmf"] = T_N
        ib = folds[tr]; ytr = y[tr]
        yR = rank_gauss(ytr)
        yS = mean_adj(co[tr]) @ ytr
        for name, F in FS.items():
            if name not in args.sets.split(","):
                continue
            for tgt, yfit, yev in [c for c in (("y", ytr, ytr), ("yR", yR, yR), ("yS", yS, ytr)) if c[0] in args.targets.split(",")]:
                g, i, a, a200 = nested_ridge(F[tr], yfit, yev, ib, F[te])
                preds[f"{name}|{tgt}|aG"][te] = g; preds[f"{name}|{tgt}|aI"][te] = i; al[f"{name}|{tgt}"].append(a)
                if name == "pca" and tgt == "y":
                    preds["pca|y|a200"][te] = a200
        for name in ("pca", "secz"):
            for tgt, yfit in (("yR", yR),):
                pe = et_on_ypcs(FS[name][tr], yfit, FS[name][te])
                preds[f"{name}|{tgt}|ET"][te] = pe
                if f"{name}|{tgt}|aG" in preds:
                    preds[f"ens({name}|{tgt}|aG,ET)"][te] = 0.5 * (preds[f"{name}|{tgt}|aG"][te] + pe)
        print(f"  {sec} fold {fold} {time.time()-t0:.0f}s", flush=True)
    res, per = {}, {}
    for k, p in preds.items():
        r = col_spearman(p, y); r[~((y.std(0) > 1e-9) & (p.std(0) > 1e-9))] = np.nan
        res[k] = float(np.nanmedian(r)); per[k] = r.astype(np.float32)
    np.savez_compressed(f"{args.out}/{sec}_per_ion.npz", top=top, **{k.replace("|", "__"): v for k, v in per.items()})
    return res, dict(al)


summary = {}
with threadpool_limits(limits=1):
    for sec in args.sections:
        res, al = run_section(sec)
        summary[sec] = dict(scores=res, alpha_global=al)
        json.dump(summary, open(f"{args.out}/within_summary.json", "w"), indent=1)
        print(f"{sec}: base pca|y|a200={res['pca|y|a200']:.4f}; " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(res.items(), key=lambda kv: -kv[1])[:5]), flush=True)
print("DONE")
