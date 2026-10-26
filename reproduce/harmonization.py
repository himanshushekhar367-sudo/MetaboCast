"""Does transcriptome harmonization close the cross-section transfer gap? (harmonisation table and SuppFig_harmony)

Leave-one-section-out on the seven DESIUM sections with the released MetaboCast estimator (estimator.py:
rank-Gaussian targets per training section, cross-validated ridge penalty, ridge + ExtraTrees ensemble).
Only the transcriptome representation changes between variants:
  per-fold PCA        PCA fitted on the six training sections (as in the package)
  pooled PCA          PCA fitted on all seven sections (transductive: the query transcriptome is known)
  pooled + Harmony    pooled PCA scores integrated across sections with Harmony (unsupervised; no metabolome)
  centred PCA         per-fold PCA scores centred within each section
  z-scored features   diffusion features z-scored within each section
Features are [P, AP, A^2 P] on a 6-nearest-neighbour spot graph; 2,000 highly variable genes pooled over the
sections. Score: median Spearman over held-out ions detected in more than 20% of spots.
Run: python reproduce/harmonization.py   (writes $METABOCAST_WORK/harmonization.json)
"""
import gc, json, os, sys, time, warnings
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P                                    # noqa: E402  data and work locations
import numpy as np, pandas as pd                     # noqa: E402
from sklearn.decomposition import PCA                # noqa: E402
from scipy.spatial import cKDTree                    # noqa: E402
from scipy.sparse import csr_matrix, diags, eye      # noqa: E402
from estimator import make                           # noqa: E402
from within_benchmark import read_section, col_spearman   # noqa: E402
warnings.filterwarnings("ignore")

SECS = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]


def nadj(c, k=6):
    n = len(c); _, idx = cKDTree(c).query(c, k=min(k + 1, n))
    r = np.repeat(np.arange(n), idx.shape[1] - 1); cc = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(r)), (r, cc)), shape=(n, n)); A = ((A + A.T) > 0).astype(np.float32) + eye(n, dtype=np.float32)
    return diags((1 / np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32)) @ A


def main():
    names, X, Y, DET, ADJ = {}, {}, {}, {}, {}
    for s in SECS:
        nm, x, raw, _, c = read_section(Path(P.DESIUM) / f"{s}.h5ad")
        names[s], X[s] = nm, x; Y[s] = np.log1p(raw); DET[s] = (raw > 0).mean(0) > 0.2; ADJ[s] = nadj(c)
    genes = sorted(set.intersection(*[set(names[s]) for s in SECS]))
    for s in SECS:
        pos = {g: i for i, g in enumerate(names[s])}; X[s] = X[s][:, [pos[g] for g in genes]]
    s1 = sum(X[s].sum(0) for s in SECS); s2 = sum((X[s] * X[s]).sum(0) for s in SECS); n = sum(len(X[s]) for s in SECS)
    hv = np.argsort(s2 / n - (s1 / n) ** 2)[-2000:]
    X = {s: np.ascontiguousarray(X[s][:, hv]) for s in SECS}; gc.collect()
    lens = {s: len(X[s]) for s in SECS}

    def split(M):
        out, i = {}, 0
        for s in SECS:
            out[s] = M[i:i + lens[s]]; i += lens[s]
        return out

    Xall = np.vstack([X[s] for s in SECS])
    Ppool = split(PCA(100, svd_solver="randomized", random_state=0).fit_transform(Xall).astype(np.float32))
    import harmonypy
    batch = pd.DataFrame({"section": np.concatenate([[s] * lens[s] for s in SECS])})
    Z = np.asarray(harmonypy.run_harmony(np.vstack([Ppool[s] for s in SECS]), batch, ["section"], max_iter_harmony=10,
                                         random_state=0).Z_corr, np.float32)
    if Z.shape[0] != Xall.shape[0]:
        Z = Z.T
    Pharm = split(Z); del Xall; gc.collect()

    def tfeat(Pm, s):
        return np.hstack([Pm, ADJ[s] @ Pm, ADJ[s] @ (ADJ[s] @ Pm)]).astype(np.float32)

    def features(mode, held):
        tr = [s for s in SECS if s != held]
        if mode in ("per-fold PCA", "centred PCA", "z-scored features"):
            pca = PCA(100, svd_solver="randomized", random_state=0).fit(np.vstack([X[s] for s in tr]))
            emb = {s: pca.transform(X[s]).astype(np.float32) for s in SECS}
        else:
            emb = Ppool if mode == "pooled PCA" else Pharm
        if mode == "centred PCA":
            emb = {s: emb[s] - emb[s].mean(0) for s in SECS}
        F = {s: tfeat(emb[s], s) for s in SECS}
        if mode == "z-scored features":
            F = {s: (F[s] - F[s].mean(0)) / (F[s].std(0) + 1e-8) for s in SECS}
        return F

    res = {}
    for mode in ["per-fold PCA", "pooled PCA", "pooled + Harmony", "centred PCA", "z-scored features"]:
        per = {}
        for held in SECS:
            t0 = time.time(); tr = [s for s in SECS if s != held]; F = features(mode, held)
            lab = np.concatenate([[s] * lens[s] for s in tr])
            m = make().fit(np.vstack([F[s] for s in tr]), np.vstack([Y[s] for s in tr]), lab, lab)
            r = col_spearman(m.predict(F[held]), Y[held])
            per[held] = float(np.nanmedian(r[DET[held]]))
            print(f"  {mode:18s} {held:18s} {per[held]:+.3f} ({time.time() - t0:.0f}s)", flush=True)
            del m, F; gc.collect()
        v = np.array([per[s] for s in SECS])
        res[mode] = dict(overall=[float(v.mean()), float(v.std(ddof=1))], breast=float(v[:4].mean()), lung=float(v[4:].mean()), per=per)
        print(f"{mode:18s} mean {v.mean():+.3f} | breast {v[:4].mean():+.3f} | lung {v[4:].mean():+.3f}", flush=True)
    out = Path(P.WORK) / "harmonization.json"
    json.dump(res, open(out, "w"), indent=1); print("saved", out)


if __name__ == "__main__":
    main()
