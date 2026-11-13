"""Leave-one-section-out improvement screen (fast exact ridge from per-section sufficient statistics).
Base protocol for DESIUM: pooled 2,000 HVGs, PCA-100 fitted on training sections,
2-hop diffusion, ridge alpha=200, per-ion Spearman on ions detected in >20% of held-out spots, per-section median.
For SMA use --hvg train to mirror the released package (sma_transfer table).
Variants: representations, nested alpha (inner leave-one-section-out on TRAINING sections only; with a single
training section its five spatial blocks are the inner units), within-section centring (wc) and training-target
transforms (yR rank-Gaussian per training section; yS kNN-smoothed training labels), ExtraTrees and ensembles.
Held-out labels are used only for scoring."""
import sys, os, json, time, argparse, warnings
import numpy as np, h5py
from collections import defaultdict
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, diags, eye
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_lib import *
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--sections", nargs="+", default=DESIUM)
ap.add_argument("--hvg", choices=["pooled", "train"], default="pooled")
ap.add_argument("--sets", default="pca,pca_h1,pca_h0,pca50,pca200,nmf,enz,secz,secrank,pca+kegg,nmf+kegg,pca+nmf")
ap.add_argument("--target-sets", default="pca,nmf,secz", help="sets that also get yR/yS targets and wc")
ap.add_argument("--targets", default="y,yR,yS")
ap.add_argument("--et-sets", default="pca,nmf")
ap.add_argument("--et-trees", type=int, default=100)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
gmt = load_gmt(); kegg_genes = set().union(*gmt.values())
LALPHAS = np.array([1, 10, 100, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6])
S = args.sections
SETS = args.sets.split(","); TSETS = args.target_sets.split(","); TARGETS = args.targets.split(","); ETS = [e for e in args.et_sets.split(",") if e]


def path_of(s):
    return f"{SMA_DIR}/{s}.h5ad" if s.startswith("SMA_") else f"{DESIUM_DIR}/{s}.h5ad"


def mean_adj(c, k=6):
    n = len(c); _, idx = cKDTree(c).query(c, k=min(k + 1, n))
    r = np.repeat(np.arange(n), idx.shape[1] - 1); cc = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(r)), (r, cc)), shape=(n, n)); A = ((A + A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1 / np.asarray(A.sum(1)).ravel()).astype(np.float32)) @ A


t0 = time.time()
names = {}
for s in S:
    with h5py.File(path_of(s), "r") as f:
        names[s] = var_names(f)
genes = sorted(set.intersection(*[set(v) for v in names.values()]))
upper = np.mean([g == g.upper() for g in genes[:2000]]) > 0.5
gk = genes if upper else [g.upper() for g in genes]
cols = {s: np.array([{g: i for i, g in enumerate(names[s])}[g] for g in genes]) for s in S}
stats = {}
for s in S:
    s1, s2, n = gene_stats(path_of(s)); stats[s] = (s1[cols[s]], s2[cols[s]], n)
enz = np.array([i for i, g in enumerate(gk) if g in kegg_genes])
D = {}
for s in S:
    with h5py.File(path_of(s), "r") as f:
        msi = f["uns/msi"][:].astype(np.float32); co = f["obsm/spatial"][:].astype(float)
    Y = np.log1p(msi); tic = msi.sum(1)
    D[s] = dict(co=co, A=norm_adj(co), y=Y, det=(msi > 0).mean(0) > 0.2,
                ytic=np.log1p(msi / (tic[:, None] + 1e-12) * np.median(tic)).astype(np.float32))
    if "yR" in TARGETS: D[s]["yR"] = rank_gauss(Y)
    if "yRtic" in TARGETS: D[s]["yRtic"] = rank_gauss(D[s]["ytic"])
    if "yS" in TARGETS: D[s]["yS"] = (mean_adj(co) @ Y).astype(np.float32)
    if len(S) == 2:   # single-training-section groups: fixed spatial blocks for inner CV
        D[s]["blk"] = KMeans(5, n_init=3, random_state=0).fit_predict(co)


def hv_from(secs):
    s1 = sum(stats[s][0] for s in secs); s2 = sum(stats[s][1] for s in secs); n = sum(stats[s][2] for s in secs)
    return np.argsort(s2 / n - (s1 / n) ** 2)[-2000:]


def load_cols(s, idx):
    with h5py.File(path_of(s), "r") as f:
        return read_matrix(f, "layers/log1p", cols[s][idx])


if args.hvg == "pooled":
    HV = hv_from(S)
    for s in S:
        D[s]["Xg"] = load_cols(s, HV)
for s in S:
    E = load_cols(s, enz); D[s]["Xe"] = E
    PW = pathway_scores(E, [gk[i] for i in enz], gmt)
    D[s]["PWT"] = np.hstack([PW, D[s]["A"] @ PW]).astype(np.float32)
print(f"loaded {len(S)} sections, {len(genes)} shared genes, {len(enz)} KEGG genes, {time.time()-t0:.0f}s", flush=True)


def scaled(blocks_by_sec, tr):
    """Concatenate feature blocks, each scaled by its mean training-column variance."""
    nb = len(next(iter(blocks_by_sec.values()))); sc = []
    for b in range(nb):
        M = np.vstack([blocks_by_sec[s][b] for s in tr]); sc.append(np.sqrt(M.var(0).mean() + 1e-12))
    return {s: np.hstack([blocks_by_sec[s][b] / sc[b] for b in range(nb)]).astype(np.float32) for s in blocks_by_sec}


def fit_eval(F, tr, held, target="y", wc=False, score_on=None):
    """Ridge at alpha=200, nested global alpha and nested per-ion alpha. Training targets may be transformed;
    inner validation is scored on the untransformed training labels (score_on)."""
    if score_on is None:
        score_on = {"yS": "y", "yRtic": "yRtic", "ytic": "ytic"}.get(target, target)
    def prep(s):
        f = F[s]; yf = D[s][target]
        if wc:
            f = f - f.mean(0); yf = yf - yf.mean(0)
        return f, yf
    P = {s: prep(s) for s in tr}
    if len(tr) >= 2:
        units = [(P[s][0], P[s][1], D[s][score_on], D[s]["det"]) for s in tr]
    else:
        s = tr[0]; b = D[s]["blk"]
        units = [(P[s][0][b == k], P[s][1][b == k], D[s][score_on][b == k], D[s]["det"]) for k in range(5)]
    grams = [SectionGram(u[0], u[1]) for u in units]
    m = D[held]["y"].shape[1]
    inner = np.full((len(LALPHAS), len(units), m), np.nan, np.float32)
    for k, (fv, _, yv, dv) in enumerate(units):
        j = np.where(dv)[0]
        gr = GramRidge([g for i, g in enumerate(grams) if i != k], cols=j)
        for a, p in enumerate(gr.predict_path(fv, LALPHAS)):
            inner[a, k, j] = col_pearson(p, yv[:, j])
    with np.errstate(all="ignore"):
        g = np.nanmean(np.nanmedian(inner, axis=2), axis=1); per = np.nanmean(inner, axis=1)
    gi = int(np.nanargmax(g)); pi = np.where(np.isnan(per), -np.inf, per).argmax(0); pi[np.all(np.isnan(per), 0)] = gi
    gr = GramRidge(grams)
    fh = F[held] - F[held].mean(0) if wc else F[held]
    return {"a200": gr.predict(fh, 200.0), "aG": gr.predict(fh, LALPHAS[gi]), "aI": gr.predict(fh, LALPHAS[pi])}, float(LALPHAS[gi])


def score(pred, held, on="y"):
    y = D[held][on]; det = D[held]["det"]; r = np.full(y.shape[1], np.nan, np.float32)
    ok = det & (y.std(0) > 1e-9); r[ok] = col_spearman(pred[:, ok], y[:, ok]); return r


results = defaultdict(dict); per_ion = defaultdict(dict); alpha_log = defaultdict(dict)
z = lambda p: (p - p.mean(0)) / (p.std(0) + 1e-8)
with threadpool_limits(limits=1):
    for held in S:
        t1 = time.time(); tr = [s for s in S if s != held]
        if args.hvg == "train":
            HV = hv_from(tr)
            for s in S:
                D[s]["Xg"] = load_cols(s, HV)
        def pca_fit(key, k, transform=None):
            Xs = {s: (D[s][key] if transform is None else transform(D[s][key])) for s in S}
            p = PCA(min(k, Xs[tr[0]].shape[1] - 1), svd_solver="randomized", random_state=0).fit(np.vstack([Xs[s] for s in tr]))
            return {s: p.transform(Xs[s]).astype(np.float32) for s in S}
        T = lambda R, h=2: {s: diffuse(R[s], D[s]["A"], h) for s in S}
        need = set(SETS) | set(TSETS) | set(ETS)
        FS = {}
        P100 = pca_fit("Xg", 100); FS["pca"] = T(P100)
        if any(n in need for n in ("nmf", "nmf+kegg", "pca+nmf")):
            tn = time.time()
            nm = NMF(100, init="nndsvda", max_iter=200, random_state=0).fit(np.clip(np.vstack([D[s]["Xg"] for s in tr]), 0, None))
            N100 = {s: nm.transform(np.clip(D[s]["Xg"], 0, None)).astype(np.float32) for s in S}
            FS["nmf"] = T(N100); print(f"  NMF {time.time()-tn:.0f}s", flush=True)
        cand = {"pca_h1": lambda: T(P100, 1), "pca_h0": lambda: P100, "pca50": lambda: T(pca_fit("Xg", 50)),
                "pca200": lambda: T(pca_fit("Xg", 200)), "enz": lambda: T(pca_fit("Xe", 100)),
                "secz": lambda: T(pca_fit("Xg", 100, lambda X: (X - X.mean(0)) / (X.std(0) + 1e-6))),
                "secrank": lambda: T(pca_fit("Xg", 100, rank_gauss)),
                "pca+kegg": lambda: {s: np.hstack([FS["pca"][s], D[s]["PWT"]]) for s in S},
                "nmf+kegg": lambda: scaled({s: [FS["nmf"][s], D[s]["PWT"]] for s in S}, tr),
                "pca+nmf": lambda: scaled({s: [FS["pca"][s], FS["nmf"][s]] for s in S}, tr),
                "pca+enz": lambda: (lambda E: scaled({s: [FS["pca"][s], E[s]] for s in S}, tr))(T(pca_fit("Xe", 50)))}
        for n, fn in cand.items():
            if n in need:
                FS[n] = fn()
        aG = {}
        for name in [n for n in FS if n in SETS or n in TSETS]:
            combos = [("y", False)]
            if name in TSETS:
                combos = [(t, w) for t in TARGETS for w in (False, True)]
            for tgt, wc in combos:
                pr, ga = fit_eval(FS[name], tr, held, tgt, wc)
                tag = name + ("" if tgt == "y" else f"|{tgt}") + ("|wc" if wc else "")
                alpha_log[tag][held] = ga; aG[tag] = pr["aG"]
                for k, p in pr.items():
                    per_ion[f"{tag}|{k}"][held] = score(p, held, "ytic" if "tic" in tgt else "y")
        # different target: total-ion-count-normalised MSI, scored against the TIC-normalised measurement
        pr, _ = fit_eval(FS["pca"], tr, held, "ytic", False, score_on="ytic")
        per_ion["pca|a200|TICtarget"][held] = score(pr["a200"], held, "ytic")
        per_ion["pca|aG|TICtarget"][held] = score(pr["aG"], held, "ytic")
        # representation ensembles (z-scored per ion within the held-out section: rank-insensitive to scale)
        for suf in [""] + [f"|{t}" for t in TARGETS if t != "y"] + [f"|{t}|wc" for t in TARGETS if t != "y"] + ["|wc"]:
            if f"pca{suf}" in aG and f"nmf{suf}" in aG:
                per_ion[f"ens(pca,nmf){suf}|aG"][held] = score(z(aG[f"pca{suf}"]) + z(aG[f"nmf{suf}"]), held, "ytic" if "tic" in suf else "y")
        for name in ETS:
            te = time.time()
            pe = et_on_ypcs(np.vstack([FS[name][s] for s in tr]), np.vstack([D[s]["y"] for s in tr]), FS[name][held],
                            n_estimators=args.et_trees, leaf=10)
            per_ion[f"{name}|ET"][held] = score(pe, held)
            per_ion[f"{name}|ens(ridge aG,ET)"][held] = score(z(pe) + z(aG[name]), held)
            print(f"  ET {name} {time.time()-te:.0f}s", flush=True)
        for k, d in per_ion.items():
            if held in d:
                results[k][held] = float(np.nanmedian(d[held]))
        json.dump(dict(per_section=results, alpha_global=alpha_log), open(f"{args.out}/loso_summary.json", "w"), indent=1)
        top = sorted(((k, v[held]) for k, v in results.items() if held in v), key=lambda kv: -kv[1])[:6]
        print(f"{held}: base pca|a200={results.get('pca|a200', {}).get(held, float('nan')):.4f} ({time.time()-t1:.0f}s); top: " +
              ", ".join(f"{k}={v:.3f}" for k, v in top), flush=True)
np.savez_compressed(f"{args.out}/loso_per_ion.npz", **{f"{k.replace('|','__')}__{s}": v for k, d in per_ion.items() for s, v in d.items()})
mean = {k: float(np.mean([v[s] for s in S])) for k, v in results.items() if len(v) == len(S)}
json.dump(dict(per_section=results, mean=mean, alpha_global=alpha_log), open(f"{args.out}/loso_summary.json", "w"), indent=1)
for k, v in sorted(mean.items(), key=lambda kv: -kv[1]):
    print(f"{k:34s} {v:+.4f}")
print("DONE")
