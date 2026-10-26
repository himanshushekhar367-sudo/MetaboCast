#!/usr/bin/env python3
"""Attach lightweight per-spot H&E features to a MetaboCast-format AnnData file.

The input `.h5ad` must contain Space Ranger-style embedded images in
`adata.uns['spatial'][library_id]['images']` and full-resolution spot coordinates
in `adata.obsm['spatial']`.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anndata as ad
import numpy as np

from metabocast import add_histology_features


def main():
    ap = argparse.ArgumentParser(description="Add cheap H&E patch features to adata.obsm['histology'].")
    ap.add_argument("--query", required=True, help="Input .h5ad with embedded Visium histology image.")
    ap.add_argument("--out", required=True, help="Output .h5ad path.")
    ap.add_argument("--library_id", default=None, help="Optional explicit adata.uns['spatial'] library id.")
    ap.add_argument("--obs_library_key", default="orig.ident", help="obs column used to infer library id.")
    ap.add_argument("--image_key", choices=["hires", "lowres"], default="hires")
    ap.add_argument("--patch_radius_scale", type=float, default=1.0, help="Multiplier on half spot diameter.")
    ap.add_argument("--key_added", default="histology", help="obsm key for generated features.")
    args = ap.parse_args()

    a = ad.read_h5ad(args.query)
    add_histology_features(
        a,
        key_added=args.key_added,
        library_id=args.library_id,
        obs_library_key=args.obs_library_key,
        image_key=args.image_key,
        patch_radius_scale=args.patch_radius_scale,
    )
    a.write_h5ad(args.out)
    src = a.uns[args.key_added + "_source"]
    print(f"saved {args.out}")
    print(f"  {args.key_added}: {a.obsm[args.key_added].shape[0]} spots x {a.obsm[args.key_added].shape[1]} features")
    print(f"  source: library={src['library_id']} image={src['image_key']} radius={src['patch_radius_pixels']} px")
    print(f"  finite: {np.isfinite(a.obsm[args.key_added]).all()}")


if __name__ == "__main__":
    main()
