#!/usr/bin/env python3
"""
Quick start: predict the spatial metabolome of a transcriptome-only section with the shipped model.
No mass-spectrometry data are needed.

    python examples/quickstart.py your_section.h5ad --model metabocast_model.pkl

`your_section.h5ad` needs gene expression (gene symbols) and spatial coordinates; raw counts are
normalised and coordinates are found automatically (see metabocast.prepare_query).
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd

from metabocast import MetaboCast, prepare_query, gene_panel_overlap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help="path to a .h5ad section (genes + coordinates)")
    ap.add_argument("--model", default="metabocast_model.pkl", help="pre-trained model bundle")
    ap.add_argument("--out", default="predicted_metabolites.csv", help="output CSV of predictions (spots x ions)")
    args = ap.parse_args()

    model = MetaboCast.load(args.model)
    adata = prepare_query(ad.read_h5ad(args.query))
    pred = model.predict_metabolome(adata)        # (n_spots x n_ions); also in adata.obsm['metabocast_pred']

    cols = [f"mz_{m:.4f}" for m in np.asarray(model.mz_)]
    pd.DataFrame(pred, columns=cols, index=adata.obs_names).to_csv(args.out)
    ov = gene_panel_overlap(model, adata)
    print(f"Predicted {pred.shape[1]} ions for {pred.shape[0]} spots -> {args.out}")
    print(f"Gene-panel overlap: {ov['training_genes']:.1%} of training genes, {ov['hvg']:.1%} of highly variable genes"
          + ("  (low overlap: interpret with caution)" if ov["hvg"] < 0.5 else ""))
    print("Per-ion reliability: model.reliability_table(adata)")


if __name__ == "__main__":
    main()
