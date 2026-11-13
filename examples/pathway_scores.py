#!/usr/bin/env python3
"""
Score per-spot metabolic pathway activity for any spatial transcriptomics section from enzyme expression, in the
style of scMetabolism (AUCell or mean z-score), with optional spatial smoothing. No metabolomics required; these are
gene-programme scores, not metabolite measurements.

USAGE
    python examples/pathway_scores.py --query your_section.h5ad \
           --gmt data/genesets/scMetab_KEGG.gmt --out pathways/ --method aucell --smooth 1 --plot

Outputs a spots x pathways activity table, writes the scores into the section's .obsm, and
(optionally) plots the most spatially-variable pathways. Add --smooth 1 (or 2) to diffuse
activity over the tissue graph — the improvement over per-spot scoring for spatial data.
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metabocast import PathwayScorer, find_coordinates


def ensure_spatial(adata):
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = find_coordinates(adata)[0]


def main():
    ap = argparse.ArgumentParser(description="scMetabolism-style metabolic pathway activity (spatially aware).")
    ap.add_argument("--query", required=True, help="spatial transcriptomics .h5ad")
    ap.add_argument("--gmt", default="data/genesets/scMetab_KEGG.gmt", help="metabolic gene sets (.gmt)")
    ap.add_argument("--out", default="pathways", help="output directory")
    ap.add_argument("--method", choices=["mean_z", "aucell"], default="aucell")
    ap.add_argument("--smooth", type=int, default=0, help="graph-diffusion hops (0=off; 1-2 = spatially smoothed)")
    ap.add_argument("--normalize", choices=["auto", "raw", "none"], default="auto")
    ap.add_argument("--top", type=int, default=20, help="most spatially-variable pathways to report")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    q = sc.read_h5ad(args.query); q.var_names_make_unique()
    print(f"query: {q.n_obs} spots x {q.n_vars} genes")
    ensure_spatial(q)
    x = q.X
    mx = float(x.max())
    if args.normalize == "raw" or (args.normalize == "auto" and mx > 50):
        sc.pp.normalize_total(q, target_sum=1e4); sc.pp.log1p(q)
        print("  normalisation: CP10k + log1p")
    q.layers["log1p"] = q.X.copy()

    ps = PathwayScorer(args.gmt, method=args.method)
    df = ps.score(q, layer="log1p", spatial_smooth=args.smooth)
    print(f"  scored {df.shape[1]} metabolic pathways (method={args.method}"
          f"{', spatial-smoothed x'+str(args.smooth) if args.smooth else ''})")

    df.to_csv(os.path.join(args.out, "pathway_scores.csv"))
    q.write(os.path.join(args.out, "query_with_pathways.h5ad"))
    var = df.var(0).sort_values(ascending=False)
    print(f"\nsaved:\n  {args.out}/pathway_scores.csv   (spots x {df.shape[1]} pathways)")
    print(f"  {args.out}/query_with_pathways.h5ad   (.obsm['pathway_scores'])")
    print(f"\ntop {min(args.top, len(var))} most spatially-variable metabolic pathways:")
    for name, v in var.head(args.top).items():
        print(f"  {v:7.4f}  {name}")

    if args.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xy = np.asarray(q.obsm["spatial"], float); top = var.head(6).index
        fig, axs = plt.subplots(2, 3, figsize=(12, 7.5))
        for ax, name in zip(axs.ravel(), top):
            v = df[name].values; lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            ax.scatter(xy[:, 0], xy[:, 1], c=v, s=6, cmap="viridis", vmin=lo, vmax=hi, linewidths=0)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name[:42], fontsize=8)
        for ax in axs.ravel()[len(top):]:
            ax.axis("off")
        fig.suptitle("Metabolic pathway activity (top spatially-variable)", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "pathway_maps.png"), dpi=150)
        print(f"  {args.out}/pathway_maps.png")


if __name__ == "__main__":
    main()
