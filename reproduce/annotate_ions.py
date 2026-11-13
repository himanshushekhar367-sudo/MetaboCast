"""Putative exact-mass candidates for the DESIUM m/z axis (Figs 3 and 4, SuppFig_best_worst and the per_ion table).

DESIUM was acquired in negative ion mode, so each m/z channel is compared only with negative-mode adducts
([M-H]-, [M+Cl]-, [M+HCOO]-) of the curated reference in metabocast/annotation.py: common tissue metabolites and
representative lipid species, with monoisotopic masses from molecular formulae. The same reference annotates the
HCC/GBM applications and MetaboCast.reliability_table(). Every candidate within 0.01 Da is kept and listed nearest
first; isomers that share a formula carry one group name (Hexose covers glucose, fructose, galactose and inositol);
a channel with more than one candidate is flagged as ambiguous, and its chemical class is used only when all
candidates share it. Mass errors are observed minus theoretical m/z. These are putative exact-mass candidates (no
fragmentation, isotope-pattern or standard evidence), not structural identifications.

Run: python reproduce/annotate_ions.py --data <DESIUM_DIR> --out reproduce/results/desium_ion_annotation.csv
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metabocast.annotation import ADDUCTS, REFERENCE, TOL, annotate, formula_mass  # noqa: E402,F401

REF = REFERENCE          # (group name, molecular formula, chemical class)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, type=Path, help="DESIUM folder with the paired .h5ad sections")
    ap.add_argument("--section", default="BC_515_Section_1", help="any section: all seven share one m/z axis")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    import h5py
    with h5py.File(args.data / f"{args.section}.h5ad", "r") as f:
        mz = f["uns/mz_features"][:]
    df = annotate(mz, tol=TOL)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    a = df.n_candidates > 0
    print(f"{int(a.sum())} of {len(df)} channels have a negative-mode candidate; {int(df.ambiguous.sum())} have more "
          f"than one; {int((a & (df['class'] != 'ambiguous')).sum())} have a single chemical class")


if __name__ == "__main__":
    main()
