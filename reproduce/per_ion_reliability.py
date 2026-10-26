#!/usr/bin/env python3
"""Per-ion accuracy on the DESIUM m/z axis, averaged over the seven sections.

Reads the per-ion Spearman correlations written by within_benchmark.py (within-section spatial-block CV) and
loso_benchmark.py (leave-one-section-out) and writes desium_per_ion_accuracy.csv: m/z, mean within-section r and
mean leave-one-section-out r per ion. These two columns define the reliability tiers (transfer-supported:
leave-one-section-out r >= 0.20; within-section only: within-section r >= 0.30 with weaker transfer) used by the
HCC and GBM analyses and the per-ion table.

Usage:
    python reproduce/per_ion_reliability.py --within reproduce/results/within_desium_per_ion.npz \
        --loso reproduce/results/loso_desium_per_ion.npz --out reproduce/results
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DESIUM = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--within", type=Path, required=True, help="within_desium_per_ion.npz")
    ap.add_argument("--loso", type=Path, required=True, help="loso_desium_per_ion.npz")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    a = ap.parse_args()
    W, L = np.load(a.within), np.load(a.loso)
    mz = np.asarray(W[f"mz__{DESIUM[0]}"], float)
    with warnings.catch_warnings():                     # ions never detected stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        within = np.nanmean([W[f"MetaboCast__{s}"] for s in DESIUM], 0)
        loso = np.nanmean([L[s] for s in DESIUM], 0)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mz": np.round(mz, 4), "within_section_r": within, "leave_one_section_out_r": loso}).round(4).to_csv(
        a.out / "desium_per_ion_accuracy.csv", index=False)
    print(f"{len(mz)} ions; median within-section r {np.nanmedian(within):.3f}, "
          f"{int(np.nansum(within >= 0.30))} ions with r >= 0.30; median leave-one-section-out r {np.nanmedian(loso):.3f}, "
          f"{int(np.nansum(loso >= 0.20))} ions with r >= 0.20")


if __name__ == "__main__":
    main()
