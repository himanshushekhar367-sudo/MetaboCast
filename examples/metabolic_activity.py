#!/usr/bin/env python3
"""
Metabolite-grounded metabolic pathway activity.

Enzyme-based scores (scMetabolism, AUCell, ssGSEA, VISION) rate a metabolic pathway from its enzyme
transcripts only, so they cannot tell whether the pathway's products are present. This example adds the
pathway's annotated metabolites, measured or predicted by MetaboCast: a pathway scores high only where its
enzymes and its putative metabolites agree, and the per-pathway enzyme-metabolite coherence shows which calls
the metabolites support. Predicted metabolites carry their own error (see reliability_table()), and agreement
between enzymes and metabolite abundance is not a flux measurement.

USAGE
    # with a trained model (predicts metabolites for any transcriptome-only section):
    python examples/metabolic_activity.py --query your_section.h5ad --model model.pkl \
           --gmt data/genesets/scMetab_KEGG.gmt --out metabolic/ --plot
    # or validate on a paired section that already has measured metabolites in .uns['msi']:
    python examples/metabolic_activity.py --query paired_section.h5ad --measured \
           --gmt data/genesets/scMetab_KEGG.gmt --out metabolic/ --plot
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metabocast import MetaboCast, MetabolicActivity


def main():
    ap = argparse.ArgumentParser(description="Metabolite-grounded metabolic pathway activity.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--gmt", default="data/genesets/scMetab_KEGG.gmt")
    ap.add_argument("--model", default=None, help="trained model bundle (.pkl) to PREDICT metabolites")
    ap.add_argument("--measured", action="store_true", help="use measured metabolites in .uns['msi'] instead of a model")
    ap.add_argument("--adducts", choices=["neg", "pos"], default="neg", help="MS ionisation mode for m/z annotation")
    ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--out", default="metabolic")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    q = sc.read_h5ad(args.query); q.var_names_make_unique()
    if "log1p" not in q.layers:
        sc.pp.normalize_total(q, target_sum=1e4); sc.pp.log1p(q); q.layers["log1p"] = q.X.copy()
    model = None
    if args.model and not args.measured:
        model = MetaboCast.load(args.model)

    ma = MetabolicActivity(args.gmt, adducts=args.adducts)
    act, conf, src = ma.score(q, model=model, spatial_smooth=args.smooth)
    print(f"metabolite source: {src} | pathways scored: {act.shape[1]} | "
          f"metabolite-supported: {int(conf['metabolite_supported'].sum())}")

    conf.to_csv(os.path.join(args.out, "pathway_confidence.csv"), index=False)
    act.to_csv(os.path.join(args.out, "metabolic_activity.csv"))
    q.write(os.path.join(args.out, "query_with_metabolic_activity.h5ad"))

    sup = conf[conf.metabolite_supported].copy()
    print("\n=== enzyme-metabolite coherence per pathway (higher = metabolically real) ===")
    print(sup[["pathway", "n_genes", "n_metabolites", "enzyme_metabolite_coherence"]].to_string(index=False))
    # pathways whose enzymes are high while their annotated metabolites do not follow (low or negative coherence)
    bad = sup[sup.enzyme_metabolite_coherence < 0.0]
    if len(bad):
        print("\n=== pathways with enzyme activity but low enzyme-metabolite coherence ===")
        print(bad[["pathway", "enzyme_metabolite_coherence"]].to_string(index=False))
    print(f"\nsaved: {args.out}/pathway_confidence.csv, metabolic_activity.csv, query_with_metabolic_activity.h5ad")

    if args.plot and "spatial" in q.obsm:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xy = np.asarray(q.obsm["spatial"], float)
        top = sup.sort_values("enzyme_metabolite_coherence", ascending=False)["pathway"].head(6).tolist()
        fig, axs = plt.subplots(2, 3, figsize=(12, 7.5))
        for ax, name in zip(axs.ravel(), top):
            v = act[name].values; lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            ax.scatter(xy[:, 0], xy[:, 1], c=v, s=6, cmap="magma", vmin=lo, vmax=hi, linewidths=0)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
            c = float(sup.set_index("pathway").loc[name, "enzyme_metabolite_coherence"])
            ax.set_title(f"{name[:34]}\ncoherence r={c:+.2f}", fontsize=8)
        for ax in axs.ravel()[len(top):]:
            ax.axis("off")
        fig.suptitle("Metabolite-grounded metabolic activity (top enzyme-metabolite-coherent pathways)", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "metabolic_activity_maps.png"), dpi=150)
        print(f"  {args.out}/metabolic_activity_maps.png")


if __name__ == "__main__":
    main()
