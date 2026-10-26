"""Paired-supervision baselines under the leave-one-section-out design of loso_benchmark.py (DESIUM).

Each held-out section is predicted by models fitted on the other sections only and scored on the ions detected
in more than 20% of its spots (detection-matched median Spearman), exactly as MetaboCast in loso_benchmark.py:
  graph-free       the released estimator on the PCA features without the diffused copies ('no diffusion' in the
                   tables and figures)
  KEGG features    the released estimator on KEGG pathway scores and one neighbour-averaged copy only
  joint NMF        genes (training highly variable genes) and metabolites factorised together (20 factors);
                   held-out metabolites are reconstructed from the held-out genes
  single gene      for each ion, signed univariate regression on the training gene most correlated with it

Usage:
    python reproduce/loso_baselines.py --data <DESIUM_DIR> --out reproduce/results/loso_baselines.json
Writes the JSON summary and <out>_per_ion.npz (per-ion held-out Spearman, key "<method>__<section>").
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import NMF
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
from metabocast import MetaboCast                       # noqa: E402
from metabocast.metabocast import _norm_adj, _dense     # noqa: E402
from metabocast.validation import single_gene_predict    # noqa: E402
from loso_benchmark import slim, col_spearman            # noqa: E402

DESIUM = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]


class GraphFree(MetaboCast):
    """The released estimator without diffusion: features are the PCA scores alone."""
    def _diffusion_features(self, P, coords, groups, extra=None, PW=None):
        return P


class KeggFeatures(MetaboCast):
    """The released estimator on KEGG pathway scores and one neighbour-averaged copy (no transcriptome PCs)."""
    def __init__(self, **kw):
        super().__init__(use_kegg=True, **kw)

    def _diffusion_features(self, P, coords, groups, extra=None, PW=None):
        A = _norm_adj(coords, groups, self.k)
        return np.hstack([PW, A @ PW]).astype(np.float32)


def score(pred, raw):
    y = np.log1p(raw); det = (raw > 0).mean(0) > 0.2
    ok = det & (y.std(0) > 1e-9) & (np.std(pred, 0) > 1e-9)
    r = np.full(y.shape[1], np.nan, np.float32); r[ok] = col_spearman(pred[:, ok], y[:, ok])
    return r, float(np.nanmedian(r[det]))


def matrix(a, genes):
    pos = {g: i for i, g in enumerate(a.var_names)}
    return _dense(a.layers["log1p"][:, [pos[g] for g in genes]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sections", nargs="+", default=DESIUM)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-hvg", type=int, default=3000)
    args = ap.parse_args()
    secs = {s: slim(ad.read_h5ad(args.data / f"{s}.h5ad")) for s in args.sections}
    shared = sorted(set.intersection(*[set(a.var_names) for a in secs.values()]))
    res, per_ion = {}, {}
    with threadpool_limits(limits=1):
        for held in args.sections:
            t0 = time.time(); tr = [s for s in args.sections if s != held]; out = {}
            raw = np.asarray(secs[held].uns["msi"], float)
            for name, cls in (("graph-free", GraphFree), ("KEGG features", KeggFeatures)):
                m = cls().fit([secs[s] for s in tr])
                out[name] = score(m.predict_metabolome(secs[held].copy()), raw)
            Xtr = np.vstack([matrix(secs[s], shared) for s in tr])
            hvg = np.argsort(Xtr.var(0))[-args.n_hvg:]
            Xtr = Xtr[:, hvg]; Xte = matrix(secs[held], shared)[:, hvg]
            Ytr = np.vstack([np.log1p(np.asarray(secs[s].uns["msi"], np.float32)) for s in tr])
            gscale, mscale = Xtr.mean() + 1e-8, Ytr.mean() + 1e-8
            cat = np.hstack([Xtr / gscale, Ytr / mscale]).clip(0, None)
            nmf = NMF(n_components=20, init="nndsvda", max_iter=250, random_state=0).fit(cat)
            hg, hm = nmf.components_[:, :Xtr.shape[1]], nmf.components_[:, Xtr.shape[1]:]
            out["joint NMF"] = score((Xte / gscale) @ np.linalg.pinv(hg) @ hm * mscale, raw)
            out["single gene"] = score(single_gene_predict(Xtr, Ytr, Xte), raw)
            res[held] = {k: v[1] for k, v in out.items()}; res[held]["seconds"] = round(time.time() - t0)
            for k, v in out.items():
                per_ion[f"{k}__{held}"] = v[0]
            print(held, res[held], flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(dict(per_section=res), indent=1))
    methods = ["graph-free", "KEGG features", "joint NMF", "single gene"]
    summary = dict(per_section=res, mean={m: float(np.mean([res[s][m] for s in res])) for m in methods},
                   sd={m: float(np.std([res[s][m] for s in res])) for m in methods})
    args.out.write_text(json.dumps(summary, indent=1))
    np.savez_compressed(str(args.out).replace(".json", "_per_ion.npz"), **per_ion)
    print(json.dumps(summary["mean"], indent=1))


if __name__ == "__main__":
    main()
