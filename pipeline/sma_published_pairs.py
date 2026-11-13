#!/usr/bin/env python3
"""Per-section pair tables (barcode, msi_x, msi_y) from the registration published with the SMA deposit.

Run after sma_published_pairs.R. Mouse sections are taken from its output after checking that every paired pixel
exists in the section's MALDI raster and carries signal. For the human sections each exported spot spectrum is located
in the section's raster (same m/z columns): candidate pixels share its pattern of non-zero channels, and the match must
agree in every channel within a relative tolerance of 1e-5. Spots without exactly one matching pixel are dropped and
counted. sections.csv records each section's MALDI matrix, subject (animal or donor), condition and brain area, taken
from the deposit's metadata.csv, together with the pair counts.

Usage:
    python pipeline/sma_published_pairs.py --sma <folder containing the 'sma' tree> --export <out dir of the R script> \
        --metadata <deposit metadata.csv> --out pipeline/sma_published_pairs
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sma_to_metabocast import find_sections      # noqa: E402

MATRIX = {"DHB": "DHB", "9-AA": "9-AA", "FMP-10": "FMP-10"}


def read_raster(csv):
    msi = pd.read_csv(csv)
    return msi.iloc[:, :2].to_numpy().astype(int), list(msi.columns[2:]), msi.iloc[:, 2:].to_numpy(np.float64)


def match_spectra(S, R, rtol=1e-5):
    """Index of the raster row (R) with the same spectrum as each row of S; -1 where there is none or more than one."""
    keys = {}
    for i, k in enumerate(np.packbits(R > 0, axis=1)):
        keys.setdefault(k.tobytes(), []).append(i)
    out = np.full(len(S), -1)
    for j, s in enumerate(S):
        if not (s > 0).any():
            continue
        hits = [i for i in keys.get(np.packbits(s > 0).tobytes(), [])
                if np.allclose(R[i], s, rtol=rtol, atol=1e-6)]
        if len(hits) == 1:
            out[j] = hits[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sma", required=True, help="folder containing the deposit's 'sma' tree")
    ap.add_argument("--export", required=True, help="output folder of sma_published_pairs.R")
    ap.add_argument("--metadata", required=True, help="the deposit's metadata.csv")
    ap.add_argument("--out", default=os.path.join(HERE, "sma_published_pairs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = pd.read_csv(args.metadata)
    meta = meta[(meta.Protocol == "Sma") & (meta["Data.Type"] != "RNA")].set_index("Sample.ID")
    rows = []
    for s in find_sections(args.sma):
        sec = s["sample"]
        xy, cols, R = read_raster(s["csv"])
        index = {(a, b): i for i, (a, b) in enumerate(xy)}
        signal = R.sum(1) > 0
        mouse = os.path.join(args.export, f"{sec}.csv")
        human = os.path.join(args.export, "human_msi", f"{sec}.csv.gz")
        if os.path.exists(mouse):
            P = pd.read_csv(mouse)
            r = np.array([index.get((a, b), -1) for a, b in zip(P.msi_x, P.msi_y)])
            n_in = len(P)
            ok = (r >= 0) & np.where(r >= 0, signal[np.maximum(r, 0)], False)
            P = P[ok]
            source = "knn_spatial_df_filtered_list"
        elif os.path.exists(human):
            H = pd.read_csv(human)
            col_idx = [cols.index(c) for c in H.columns[1:]]               # same m/z labels as the raster header
            r = match_spectra(H.iloc[:, 1:].to_numpy(np.float64), R[:, col_idx])
            n_in = len(H)
            dup = pd.Series(r[r >= 0]).duplicated(keep=False)
            if dup.any():                                                 # a pixel claimed by two spots: drop both
                bad = set(pd.Series(r[r >= 0])[dup].tolist()); r = np.where(np.isin(r, list(bad)), -1, r)
            ok = r >= 0
            P = pd.DataFrame({"barcode": H.barcode[ok].to_numpy(), "msi_x": xy[r[ok], 0], "msi_y": xy[r[ok], 1]})
            source = "hPDStr.multi.list (matched by spectrum)"
        else:
            print(f"  [skip] {sec}: no published pairing exported"); continue
        P.to_csv(os.path.join(args.out, f"{sec}.csv"), index=False)
        m = meta.loc[sec] if sec in meta.index else None
        rows.append(dict(section=sec, matrix=MATRIX.get(m.Matrix, m.Matrix) if m is not None else s["panel"],
                         subject=m["sample"] if m is not None else "", condition=m.Condition if m is not None else "",
                         species=("human" if str(m.Condition).startswith("h") else "mouse") if m is not None else "",
                         brain_area=m["Brain.area"] if m is not None else "", source=source,
                         pairs_published=int(n_in), pairs_kept=int(len(P))))
        print(f"  {sec}: {len(P)}/{n_in} published pairs kept ({source})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "sections.csv"), index=False)
    print("wrote", os.path.join(args.out, "sections.csv"))


if __name__ == "__main__":
    main()
