"""Outer-test evaluation of the released uncertainty widths, with patients as the held-out units.

The released model takes its per-ion widths from cross-fitted residuals: each training section is predicted by a
model fitted without it, and the 90% quantile of the pooled absolute residuals is attached to the model fitted on
all sections. This script checks that procedure on data it never touched. Each patient in turn is the outer test
unit (the two BC_515 sections are one patient and are always held out together) and takes no part in fitting,
penalty selection or width estimation. On the remaining patients:

  cross-fitted widths   every remaining patient is predicted by a model fitted on the others; the per-ion 90%
                        quantile of the pooled absolute residuals is attached to the model fitted on all remaining
                        patients, which then predicts the outer patient (the released procedure, one level down)
  split calibration     a model fitted on all but one remaining patient is calibrated on that patient alone and
                        evaluated on the outer patient (one fixed predictor, separate calibration data); averaged
                        over the choice of calibration patient

Coverage is the fraction of (spot, ion) pairs of the outer section whose absolute error is within the width,
over ions detected in more than 20% of that section's spots. The same outer fits give leave-one-patient-out
accuracy (per-ion Spearman on the outer section).

Usage:
    python reproduce/coverage_outer.py --data <DESIUM_DIR> --out reproduce/results/coverage/coverage_outer.json [--outer BC_515 LC_091]
    python reproduce/coverage_outer.py --merge reproduce/results/coverage/part_*.json --out reproduce/results/coverage/coverage_outer.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
from metabocast import MetaboCast                 # noqa: E402
from loso_benchmark import slim, col_spearman      # noqa: E402

PATIENTS = {"BC_515": ["BC_515_Section_1", "BC_515_Section_2"], "BC_525": ["BC_525"], "BC_823": ["BC_823"],
            "LC_091": ["LC_091"], "LC_170": ["LC_170"], "LC_276": ["LC_276"]}


def fit(secs, units):
    return MetaboCast().fit([secs[s] for u in units for s in PATIENTS[u]])


def abs_res(model, secs, unit):
    """Absolute log1p residuals and predictions of every section of a unit."""
    out = {}
    for s in PATIENTS[unit]:
        pred = model.predict_metabolome(secs[s].copy())
        y = np.log1p(np.asarray(secs[s].uns["msi"], float))
        out[s] = (np.abs(y - pred).astype(np.float32), pred.astype(np.float32))
    return out


def run(secs, outer_units, q):
    res, arrays = {}, {}
    for o in outer_units:
        t0 = time.time()
        rest = [u for u in PATIENTS if u != o]
        cal_res, split_cov, split_width = {}, {s: [] for s in PATIENTS[o]}, []
        for c in rest:                                                  # inner cross-fitting
            m = fit(secs, [u for u in rest if u != c])
            rc = abs_res(m, secs, c)
            cal_res[c] = np.vstack([v[0] for v in rc.values()])
            w_c = np.quantile(cal_res[c], q, axis=0)                     # split calibration on patient c
            split_width.append(w_c)
            for s, (r_o, _) in abs_res(m, secs, o).items():
                det = (np.asarray(secs[s].uns["msi"]) > 0).mean(0) > 0.2
                split_cov[s].append(float(np.mean(r_o[:, det] <= w_c[det])))
            print(f"  outer {o}: inner fit without {c} done ({time.time() - t0:.0f}s)", flush=True)
        w = np.quantile(np.vstack(list(cal_res.values())), q, axis=0)   # cross-fitted widths
        final = fit(secs, rest)
        for s, (r_o, pred) in abs_res(final, secs, o).items():
            raw = np.asarray(secs[s].uns["msi"], float); y = np.log1p(raw)
            det = (raw > 0).mean(0) > 0.2
            ok = det & (y.std(0) > 1e-9) & (pred.std(0) > 1e-9)
            r = np.full(y.shape[1], np.nan, np.float32); r[ok] = col_spearman(pred[:, ok], y[:, ok])
            cov_ion = np.full(y.shape[1], np.nan, np.float32); cov_ion[det] = (r_o[:, det] <= w[det]).mean(0)
            sd = y.std(0)
            res[s] = dict(patient=o, n_spots=int(len(y)), n_detected=int(det.sum()),
                          coverage_crossfit=float(np.mean(r_o[:, det] <= w[det])),
                          coverage_split_mean=float(np.mean(split_cov[s])), coverage_split_by_calibration_patient=
                          dict(zip(rest, [round(v, 4) for v in split_cov[s]])),
                          median_width=float(np.median(w[det])), median_relative_width=float(np.median(w[det] / (sd[det] + 1e-8))),
                          lopo_median_r=float(np.nanmedian(r[det])), alpha=float(final.alpha_), seconds=round(time.time() - t0))
            arrays[f"lopo_r__{s}"] = r; arrays[f"coverage__{s}"] = cov_ion; arrays[f"width__{s}"] = w.astype(np.float32)
            print(s, {k: v for k, v in res[s].items() if k != "coverage_split_by_calibration_patient"}, flush=True)
    return res, arrays


def summarise(res):
    v = list(res.values())
    by_patient = {}
    for s, d in res.items():
        by_patient.setdefault(d["patient"], []).append(d)
    pat = {p: dict(coverage_crossfit=float(np.mean([d["coverage_crossfit"] for d in L])),
                   coverage_split_mean=float(np.mean([d["coverage_split_mean"] for d in L])),
                   lopo_median_r=float(np.mean([d["lopo_median_r"] for d in L]))) for p, L in by_patient.items()}
    return dict(per_section=res, per_patient=pat,
                mean_coverage_crossfit_sections=float(np.mean([d["coverage_crossfit"] for d in v])),
                mean_coverage_split_sections=float(np.mean([d["coverage_split_mean"] for d in v])),
                mean_coverage_crossfit_patients=float(np.mean([p["coverage_crossfit"] for p in pat.values()])),
                mean_coverage_split_patients=float(np.mean([p["coverage_split_mean"] for p in pat.values()])),
                lopo_mean_of_sections=float(np.mean([d["lopo_median_r"] for d in v])),
                lopo_sd_of_sections=float(np.std([d["lopo_median_r"] for d in v])),
                nominal=0.90, units="patients (BC_515 sections 1 and 2 held out together)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--outer", nargs="*", default=list(PATIENTS))
    ap.add_argument("--q", type=float, default=0.90)
    ap.add_argument("--merge", nargs="*", type=Path, default=None, help="combine partial runs (JSON + NPZ pairs)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.merge:
        res, arrays = {}, {}
        for p in args.merge:
            res.update(json.loads(p.read_text())["per_section"])
            arrays.update(dict(np.load(str(p).replace(".json", ".npz"))))
    else:
        secs = {s: slim(ad.read_h5ad(args.data / f"{s}.h5ad")) for u in PATIENTS for s in PATIENTS[u]}
        with threadpool_limits(limits=1):
            res, arrays = run(secs, args.outer, args.q)
    summary = summarise(res)
    args.out.write_text(json.dumps(summary, indent=1))
    np.savez_compressed(str(args.out).replace(".json", ".npz"), **arrays)
    print(f"cross-fitted coverage {summary['mean_coverage_crossfit_sections']:.3f}, split {summary['mean_coverage_split_sections']:.3f}; "
          f"leave-one-patient-out median r {summary['lopo_mean_of_sections']:.4f}; saved {args.out}")


if __name__ == "__main__":
    main()
