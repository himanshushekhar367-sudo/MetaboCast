#!/usr/bin/env python3
"""
Predict the spatial metabolome of a transcriptome-only section with a trained MetaboCast model.

Works with spatial transcriptomics from any platform once it is in AnnData (.h5ad) form:
Visium / Visium HD, Slide-seq, Stereo-seq, Xenium, CosMx, MERSCOPE/MERFISH and others. The query needs
gene expression (gene symbols, or Ensembl IDs plus a symbol column in .var) and per-observation
coordinates; metabocast.prepare_query finds the coordinates, normalises raw counts and, with
--bin-size, aggregates single-cell or subcellular data into pseudo-spots.

USAGE
    python examples/predict_on_your_data.py --model metabocast_model.pkl --query section.h5ad --out results/
    # single-cell resolution data (e.g. Xenium, coordinates in micrometres): aggregate to 50 um pseudo-spots
    python examples/predict_on_your_data.py --model metabocast_model.pkl --query xenium.h5ad --bin-size 50 --out results/

OUTPUTS (in --out)
    query_with_predictions.h5ad   predictions in .obsm['metabocast_pred'] (columns follow .uns['metabocast_mz'])
    predicted_metabolites.csv     every ion with putative candidates, chemical class, held-out accuracy and reliability tier
    predicted_maps.png            maps of the most spatially structured predictions (with --plot)

Predictions on tissue unlike the training data are hypotheses; check the reliability tier, the gene-panel
overlap printed below and docs/USAGE.md before interpreting them.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anndata as ad                                                            # noqa: E402
from metabocast import MetaboCast, prepare_query, gene_panel_overlap           # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict the spatial metabolome for a transcriptome-only section.")
    ap.add_argument("--model", required=True, help="model bundle (.pkl), e.g. the shipped metabocast_model.pkl")
    ap.add_argument("--query", required=True, help="query .h5ad (any spatial transcriptomics platform)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--normalize", choices=["auto", "raw", "none"], default="auto",
                    help="auto (default): CP10k + log1p when the data look like raw counts; a supplied "
                         "layers['log1p'] is used as is")
    ap.add_argument("--counts-layer", default=None, help="layer holding raw counts (e.g. 'counts')")
    ap.add_argument("--symbol-column", default=None, help="var column with gene symbols (auto-detected for Ensembl IDs)")
    ap.add_argument("--spatial-key", default=None, help="obsm key or 'xcol,ycol' obs columns with coordinates")
    ap.add_argument("--bin-size", type=float, default=None,
                    help="aggregate observations into square pseudo-spots of this side (coordinate units); "
                         "recommended for single-cell and subcellular platforms")
    ap.add_argument("--section-key", default=None,
                    help="obs column identifying tissue sections in a multi-section query (auto-detected)")
    ap.add_argument("--uppercase-genes", action="store_true",
                    help="upper-case gene symbols (approximate ortholog matching, e.g. mouse data with a human model)")
    ap.add_argument("--top", type=int, default=25, help="number of ions to print")
    ap.add_argument("--plot", action="store_true", help="save maps of the most spatially structured predictions")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    model = MetaboCast.load(args.model)
    mz = np.asarray(model.mz_, float)
    print(f"model: {len(mz)} ions, {len(model.genes_)} training genes, "
          f"{'with' if getattr(model, 'conf_width_', None) is not None else 'without'} uncertainty widths")

    raw = ad.read_h5ad(args.query)
    print(f"query: {raw.n_obs} observations x {raw.n_vars} genes ({os.path.basename(args.query)})")
    section_key = args.section_key or next(
        (k for k in ("section", "sample", "library_id", "orig.ident") if k in raw.obs and raw.obs[k].nunique() > 1), None)
    q = prepare_query(raw, symbol_column=args.symbol_column, spatial_key=args.spatial_key, normalize=args.normalize,
                      counts_layer=args.counts_layer, bin_size=args.bin_size, section_key=section_key,
                      uppercase_genes=args.uppercase_genes)
    del raw

    ov = gene_panel_overlap(model, q)
    print(f"  gene-panel overlap: {ov['training_genes']:.1%} of training genes, {ov['hvg']:.1%} of the model's "
          f"highly variable genes")
    if ov["model_symbol_case"] != ov["query_symbol_case"] and "mixed" not in (ov["model_symbol_case"], ov["query_symbol_case"]):
        print(f"  [warning] gene symbols look {ov['query_symbol_case']}-case but the model's are "
              f"{ov['model_symbol_case']}-case: probably a different species. Train on paired data from the "
              f"query species, or use --uppercase-genes for approximate symbol matching.")
    if ov["hvg"] < 0.5:
        print("  [warning] fewer than half of the model's highly variable genes are measured; absent genes are "
              "imputed at their training mean and predictions lose information. Treat them with caution.")

    pred = model.predict_metabolome(q, section_key=section_key)
    q.uns["metabocast_mz"] = mz

    rel = model.reliability_table(q)
    dyn = pred.std(0)
    tab = pd.DataFrame({"mz": np.round(mz, 4), "predicted_mean": pred.mean(0), "predicted_sd": dyn})
    for c in ("annotation", "chemical_class", "train_predictability_spearman", "transfer_predictability_spearman",
              "conf_width", "trust_tier", "uncertainty_method", "calibration_unit"):
        if c in rel:
            tab[c] = rel[c].to_numpy()
    order = np.argsort(-dyn)
    tab = tab.iloc[order].reset_index(drop=True)
    tab.to_csv(os.path.join(args.out, "predicted_metabolites.csv"), index=False)
    q.write(os.path.join(args.out, "query_with_predictions.h5ad"))

    print(f"\nsaved {args.out}/query_with_predictions.h5ad and {args.out}/predicted_metabolites.csv")
    print(f"\n{min(args.top, len(mz))} most spatially structured predictions:")
    print(tab.head(args.top).to_string(index=False))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xy = np.asarray(q.obsm["spatial"], float)
        fig, axs = plt.subplots(2, 3, figsize=(11, 7))
        for k, ax in enumerate(axs.ravel()):
            if k >= min(6, len(mz)):
                ax.axis("off")
                continue
            j = order[k]
            v = pred[:, j]
            ax.scatter(xy[:, 0], xy[:, 1], c=v, s=6, cmap="magma", vmin=np.percentile(v, 2),
                       vmax=np.percentile(v, 98), linewidths=0)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
            name = tab["annotation"].iloc[k] if "annotation" in tab else ""
            name = name if isinstance(name, str) else ""
            ax.set_title(f"m/z {mz[j]:.3f}" + (f"\n{name}" if name else ""), fontsize=8)
        fig.suptitle("MetaboCast predicted metabolome (most spatially structured ions)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "predicted_maps.png"), dpi=150)
        print(f"  {args.out}/predicted_maps.png")


if __name__ == "__main__":
    main()
