#!/usr/bin/env python3
"""
Summary measures for the transcriptome-only cancers (HCC, GBM), where no MSI exists.

There is no measured metabolome in HCC or GBM, so accuracy cannot be computed there. This script reports
what can be measured for one regenerate_crosscancer.py run:

  1. Measured accuracy of the ions the HCC/GBM analyses rely on, on DESIUM where MSI exists: within-section r
     and leave-one-section-out r (transfer to an unseen section, the closer analogue of applying the model to
     new tissue), and the accuracy tier that MetaboCast.reliability_table() assigns (transfer-supported:
     transfer r >= 0.20; within-section only: within r >= 0.30 with weaker transfer; low).
  2. How many annotated ions fall in each tier.
  3. Agreement with seven literature-expected normal-to-tumour directions across four HCC patients
     (glutathione, ascorbate, arachidonate and DHA up; palmitate, stearate and hexose down), reported as counts.
     The 28 calls share one model, seven metabolites and four patients; they are not independent replicates,
     so no test is applied.
  4. Saturated-fatty-acid changes (palmitate and stearate) across patients, and sign agreement of the
     cross-patient contrasts.
  5. GBM CytAssist: strength of predicted-metabolite / measured-protein co-localisation.

Usage
    python summarise_crosscancer.py --run <crosscancer_dir> --accuracy reproduce/results/desium_per_ion_accuracy.csv \
        --out <dir>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metabocast.annotation import ADDUCTS, candidate_labels, neutral_mass   # noqa: E402  reference of the per-ion table

# metabolites the analyses name -> their entries in the annotation reference
REFNAME = {"Glutathione(GSH)": "Glutathione", "Ascorbate": "Ascorbate", "Arachidonate(FA20:4)": "FA20:4(AA)",
           "DHA(FA22:6)": "FA22:6(DHA)", "Palmitate(FA16:0)": "FA16:0", "Stearate(FA18:0)": "FA18:0", "Hexose": "Hexose",
           "Glutamate": "Glutamate", "Oleate(FA18:1)": "FA18:1", "Linoleate(FA18:2)": "FA18:2", "Inosine": "Inosine",
           "Taurine": "Taurine", "N-acetylaspartate": "N-acetylaspartate", "Urate": "Urate", "Xanthine": "Xanthine",
           "AMP": "AMP", "Lactate": "Lactate"}
MASS = {k: neutral_mass(v) for k, v in REFNAME.items()}
EXPECTED = {"Glutathione(GSH)": 1, "Ascorbate": 1, "Arachidonate(FA20:4)": 1, "DHA(FA22:6)": 1,
            "Palmitate(FA16:0)": -1, "Stearate(FA18:0)": -1, "Hexose": -1}
NEG = list(ADDUCTS.items())
PAT = ["HCC-1", "HCC-2", "HCC-3", "HCC-4"]


def tier(within, transfer, within_min=0.30, transfer_min=0.20):
    """The rule of MetaboCast.reliability_table()."""
    if np.isfinite(transfer) and transfer >= transfer_min:
        return "transfer-supported"
    if np.isfinite(within) and within >= within_min:
        return "within-section only"
    return "low" if (np.isfinite(within) or np.isfinite(transfer)) else "uncalibrated"


def ion(mz, name):
    for _, d in NEG:
        i = int(np.argmin(np.abs(mz - (MASS[name] + d))))
        if abs(mz[i] - (MASS[name] + d)) <= 0.01:
            return i
    return None


def adduct_of(v, mass):
    """The adduct by which channel v matches a neutral mass (the first within 0.01 Da)."""
    return next((a for a, d in NEG if abs(v - (mass + d)) <= 0.01), None)


def accuracy_on(mz, path):
    """Within-section and leave-one-section-out r of every ion on the m/z axis (NaN where not detected)."""
    a = pd.read_csv(path); amz = a.mz.to_numpy(float)
    idx = [int(np.argmin(np.abs(amz - v))) for v in mz]
    pick = lambda col: np.array([a[col].iloc[i] if abs(amz[i] - v) <= 1e-3 else np.nan for i, v in zip(idx, mz)], float)
    return pick("within_section_r"), pick("leave_one_section_out_r")


def run_scores(d, rel, loso, mz):
    out = {}
    ann_ion = {n: ion(mz, n) for n in MASS}
    # ---- HCC normal -> tumour
    delta = {}
    for pid in PAT:
        f = d / f"{pid}_pred.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True); p, r = z["pred"], z["region"].astype(str)
        if {"N", "T"} <= set(r):
            delta[pid] = {n: float(p[r == "T", i].mean() - p[r == "N", i].mean()) for n, i in ann_ion.items() if i is not None}
        if pid == "HCC-2" and {"N", "P"} <= set(r):
            out["HCC-2 N->P"] = {n: float(p[r == "P", i].mean() - p[r == "N", i].mean()) for n, i in ann_ion.items() if i is not None}
    hits = {pid: {n: int(np.sign(delta[pid][n]) == e) for n, e in EXPECTED.items()} for pid in delta}
    k = sum(sum(v.values()) for v in hits.values()); n = 7 * len(hits)
    out["concordance"] = {"k": k, "n": n, "per_patient": {p: sum(v.values()) for p, v in hits.items()},
                          "per_metabolite": {m: sum(hits[p][m] for p in hits) for m in EXPECTED},
                          "note": "descriptive counts; the calls are not independent replicates"}
    out["sfa_depletion_patients"] = {m: int(sum(delta[p][m] < 0 for p in delta)) for m in ("Palmitate(FA16:0)", "Stearate(FA18:0)")}
    if "HCC-2 N->P" in out:
        out["HCC-2 N->P expected directions"] = int(sum(np.sign(out["HCC-2 N->P"][m]) == e for m, e in EXPECTED.items()))
    out["delta_NtoT"] = delta
    # ---- signature consistency: ions whose sign agrees in >= 3 of 4 patients
    sig = [m for m in MASS if all(m in delta[p] for p in delta)]
    agree = {m: int(max(sum(delta[p][m] > 0 for p in delta), sum(delta[p][m] < 0 for p in delta))) for m in sig}
    out["signature_sign_agreement_>=3of4"] = int(sum(v >= 3 for v in agree.values()))
    out["signature_n"] = len(sig)
    # ---- DESIUM accuracy of the ions the analyses use
    out["ion_r_within"] = {m: (None if i is None or not np.isfinite(rel[i]) else round(float(rel[i]), 3)) for m, i in ann_ion.items()}
    out["ion_r_loso"] = {m: (None if i is None or not np.isfinite(loso[i]) else round(float(loso[i]), 3)) for m, i in ann_ion.items()}
    out["ion_tier"] = {m: (None if i is None else tier(rel[i], loso[i])) for m, i in ann_ion.items()}
    # ---- GBM CytAssist co-localisation
    f = d / "S10_metabolite_x_protein.csv"
    if f.exists():
        s10 = pd.read_csv(f, index_col=0)
        M = s10.drop(columns=[c for c in ("mz", "within_section_r", "transfer_r", "tier", "ambiguous_annotation") if c in s10]).abs()
        out["gbm_max_abs_rho_per_metabolite"] = {k: round(float(v), 3) for k, v in M.max(1).items()}
        out["gbm_mean_max_abs_rho"] = round(float(M.max(1).mean()), 3)
        out["gbm_median_abs_rho"] = round(float(np.median(M.values)), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("run", "accuracy", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mz = np.asarray(np.load(a.run / "HCC-2_pred.npz")["mz"], float)
    ann = np.array(candidate_labels(mz), dtype=object)          # the reference of the DESIUM per-ion table
    rel, loso = accuracy_on(mz, a.accuracy)
    r = run_scores(a.run, rel, loso, mz)
    annotated = ann != ""
    tiers = np.array([tier(w, t) for w, t in zip(rel, loso)])
    r["n_annotated"] = int(annotated.sum())
    r["n_annotated_by_tier"] = {t: int((annotated & (tiers == t)).sum()) for t in
                                ("transfer-supported", "within-section only", "low", "uncalibrated")}
    r["median_within_r_annotated"] = float(np.nanmedian(rel[annotated]))
    r["median_loso_r_annotated"] = float(np.nanmedian(loso[annotated]))
    r["median_within_r_all"] = float(np.nanmedian(rel)); r["median_loso_r_all"] = float(np.nanmedian(loso))
    (a.out / "crosscancer_summary.json").write_text(json.dumps(r, indent=1, default=float))
    idx = {m: ion(mz, m) for m in MASS}
    rows = [{"metabolite": m, "m/z": None if idx[m] is None else round(float(mz[idx[m]]), 4),
             "adduct": None if idx[m] is None else adduct_of(mz[idx[m]], MASS[m]),
             "within-section r": r["ion_r_within"][m], "leave-one-section-out r": r["ion_r_loso"][m],
             "tier": r["ion_tier"][m],
             **{f"{p} tumour - normal": round(r["delta_NtoT"].get(p, {}).get(m, np.nan), 4) for p in PAT}} for m in MASS]
    pd.DataFrame(rows).to_csv(a.out / "crosscancer_ion_reliability.csv", index=False)
    print(f"annotated ions by tier: {r['n_annotated_by_tier']} of {r['n_annotated']}; median within r (annotated) "
          f"{r['median_within_r_annotated']:.3f}; median LOSO r (annotated) {r['median_loso_r_annotated']:.3f}")
    c = r["concordance"]
    print(f"expected N->T directions: {c['k']}/{c['n']} (descriptive); per patient {c['per_patient']}")
    print(f"per metabolite {c['per_metabolite']}; SFA lower in tumour {r['sfa_depletion_patients']}; "
          f"HCC-2 N->P matches {r.get('HCC-2 N->P expected directions')}/7; signature sign agreement {r['signature_sign_agreement_>=3of4']}/{r['signature_n']}")
    if "gbm_mean_max_abs_rho" in r:
        print(f"GBM CytAssist mean max|rho| {r['gbm_mean_max_abs_rho']}, median |rho| {r['gbm_median_abs_rho']}")


if __name__ == "__main__":
    main()
