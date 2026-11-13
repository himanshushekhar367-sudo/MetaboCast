#!/usr/bin/env python3
"""
Figure-source tables for the HCC and GBM figures from one regenerate_crosscancer.py run
(Figs 5d and 7, SuppFig_HCC2_subsections, SuppFig_GSH_patients, SuppFig_HCC_typelevel, SuppFig_GBM_maps).

Writes to --out:
  HCC2_maps.csv                         per-spot HCC-2 predictions of the plotted ions, with region labels
  HCC2_trajectory.csv                   HCC-2 mean prediction per region minus the normal region
  HCC_NtoT_annotated_by_patient.csv     tumour-minus-normal difference per patient for the annotated ions in the two
                                        upper DESIUM accuracy tiers (transfer-supported, within-section only)
  HCC_GSH_per_patient.csv               mean predicted glutathione in normal and tumour regions per patient
  HCC_TypeLevel_TumorCell_vs_NormalCell.csv
                                        tumour-minus-normal difference by per-spot histological Type, pooled over the
                                        patient's sections
  HCC_TypeLevel_LeadingEdge.csv         the same contrast within the leading-edge section only, which holds hundreds
                                        of tumour- and of normal-typed spots in every patient
  GBM_Visium_maps.csv                   per-spot GBM Visium predictions of the plotted ions, with niches
  and copies of the run's S8_*, S9_*, S10_* and Fig7e/f tables (the run's own file names; reproduce/results/README.md maps
  them to the supplementary tables).

Usage
    python reproduce/crosscancer/derive_figure_sources.py --run <regenerate_crosscancer output> \
        --root <data root holding HCC-<n>-expr_meta.csv> --out reproduce/results/crosscancer
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metabocast.annotation import ADDUCTS, neutral_mass      # noqa: E402  the reference of the DESIUM per-ion table

NEG = list(ADDUCTS.items())
# plotted ions -> their entries in the annotation reference
REFNAME = {"GSH": "Glutathione", "Ascorbate": "Ascorbate", "Arachidonate": "FA20:4(AA)", "DHA": "FA22:6(DHA)",
           "Palmitate": "FA16:0", "Stearate": "FA18:0", "Hexose": "Hexose", "Glutamate": "Glutamate", "Oleate": "FA18:1",
           "Linoleate": "FA18:2", "Inosine": "Inosine", "Taurine": "Taurine", "Urate": "Urate"}
MASS = {k: neutral_mass(v) for k, v in REFNAME.items()}
FULL = {"Arachidonate(FA20:4)": "Arachidonate", "DHA(FA22:6)": "DHA", "Hexose": "Hexose", "Inosine": "Inosine",
        "Taurine": "Taurine", "Urate": "Urate", "Glutathione(GSH)": "GSH", "Ascorbate": "Ascorbate", "Glutamate": "Glutamate",
        "Palmitate(FA16:0)": "Palmitate", "Stearate(FA18:0)": "Stearate", "Oleate(FA18:1)": "Oleate",
        "Linoleate(FA18:2)": "Linoleate"}
PAIRED = ["HCC-1", "HCC-2", "HCC-3", "HCC-4"]          # HCC-5 has no normal region
COPY = ["S8_full_all_ions.csv", "S8_supported_annotated.csv", "S9_and_Fig7c_source.csv", "S10_metabolite_x_protein.csv",
        "S10_long.csv", "Fig7e_niches.csv", "Fig7f_metabolite_by_niche.csv"]


def ion(mz, mass):
    """Column of the first negative-mode adduct of `mass` on the m/z axis (within 0.01 Da), else None."""
    for _, d in NEG:
        i = int(np.argmin(np.abs(mz - (mass + d))))
        if abs(mz[i] - (mass + d)) <= 0.01:
            return i
    return None


def derive(run, root, out):
    out.mkdir(parents=True, exist_ok=True)
    s8 = pd.read_csv(run / "S8_full_all_ions.csv")
    rel = s8[(s8.annotation.fillna("") != "") & s8.tier.isin(["transfer-supported", "within-section only"])]

    z = np.load(run / "HCC-2_pred.npz", allow_pickle=True)
    mz, p, reg, co = np.asarray(z["mz"], float), z["pred"], z["region"].astype(str), z["coords"]
    cols = {n: p[:, ion(mz, m)] for n, m in MASS.items() if ion(mz, m) is not None}
    pd.DataFrame({"x": co[:, 1], "y": co[:, 0], "region": reg, **cols}).to_csv(out / "HCC2_maps.csv", index=False)
    rows = []
    for n, v in cols.items():
        base = v[reg == "N"].mean()
        rows += [dict(ion=n, region=r, delta_vs_N=float(v[reg == r].mean() - base)) for r in ["N", "L", "P", "T"] if (reg == r).any()]
    pd.DataFrame(rows).to_csv(out / "HCC2_trajectory.csv", index=False)

    by_patient, gsh, typ, edge = [], [], [], []
    for pid in PAIRED:
        z = np.load(run / f"{pid}_pred.npz", allow_pickle=True)
        p, reg, bc = z["pred"], z["region"].astype(str), z["barcodes"].astype(str)
        N, T = reg == "N", reg == "T"
        if not (N.any() and T.any()):
            continue
        for _, r in rel.iterrows():
            j = int(np.argmin(np.abs(mz - r.mz)))
            by_patient.append(dict(patient=pid, mz=r.mz, annotation=r.annotation, within_section_r=r.within_section_r,
                                   transfer_r=r.transfer_r, tier=r.tier, ambiguous=bool(r.ambiguous_annotation),
                                   delta_NtoT=float(p[T, j].mean() - p[N, j].mean())))
        g = ion(mz, MASS["GSH"])
        gsh.append(dict(patient=pid, mean_N=float(p[N, g].mean()), mean_T=float(p[T, g].mean()), n_N=int(N.sum()), n_T=int(T.sum())))
        meta = pd.read_csv(root / f"{pid}-expr_meta.csv", index_col=0).reindex(bc)
        ty = meta["Type"].astype(str).values
        nn, tt = ty == "Normal", ty == "Tumor"
        for full, short in FULL.items():
            j = ion(mz, MASS[short])
            if j is not None and nn.any() and tt.any():
                typ.append(dict(patient=pid, metabolite=full, delta=float(p[tt, j].mean() - p[nn, j].mean()),
                                n_normal=int(nn.sum()), n_tumour=int(tt.sum())))
        # the same contrast inside one section: the leading-edge section holds both tumour- and normal-typed spots
        le = reg == "L"
        nl, tl = nn & le, tt & le
        for full, short in FULL.items():
            j = ion(mz, MASS[short])
            if j is not None and nl.any() and tl.any():
                edge.append(dict(patient=pid, metabolite=full, delta=float(p[tl, j].mean() - p[nl, j].mean()),
                                 n_normal=int(nl.sum()), n_tumour=int(tl.sum())))
    pd.DataFrame(by_patient).to_csv(out / "HCC_NtoT_annotated_by_patient.csv", index=False)
    pd.DataFrame(gsh).to_csv(out / "HCC_GSH_per_patient.csv", index=False)
    pd.DataFrame(typ).to_csv(out / "HCC_TypeLevel_TumorCell_vs_NormalCell.csv", index=False)
    pd.DataFrame(edge).to_csv(out / "HCC_TypeLevel_LeadingEdge.csv", index=False)

    z = np.load(run / "GBM_Visium_pred.npz", allow_pickle=True)
    p, co, gmz = z["pred"], z["coords"], np.asarray(z["mz"], float)
    maps = {"Arachidonate(FA20:4)": "Arachidonate", "DHA(FA22:6)": "DHA", "Hexose": "Hexose",
            "Glutathione(GSH)": "GSH", "Palmitate(FA16:0)": "Palmitate"}
    pd.DataFrame({"x": co[:, 1], "y": co[:, 0], "niche": z["niche"],
                  **{k: p[:, ion(gmz, MASS[v])] for k, v in maps.items()}}).to_csv(out / "GBM_Visium_maps.csv", index=False)
    for f in COPY:
        if (run / f).exists():
            pd.read_csv(run / f).to_csv(out / f, index=False)
    print(f"wrote figure sources to {out}; annotated ions in the two upper tiers: {len(rel)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="output folder of regenerate_crosscancer.py")
    ap.add_argument("--root", type=Path, required=True, help="data root holding HCC-<n>-expr_meta.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    derive(a.run, a.root, a.out)


if __name__ == "__main__":
    main()
