#!/usr/bin/env python3
"""Build the shipped model bundle (metabocast_model.pkl) with the package defaults.

The model is MetaboCast() fitted on all seven paired DESIUM sections. Its per-ion reliability comes from
held-out data only:
  train_predictability_     within-section spatial-block CV Spearman, mean over the seven sections
                            (reproduce/within_benchmark.py)
  transfer_predictability_  leave-one-section-out Spearman, mean over the seven held-out sections; with
                            train_predictability_ it sets the tiers of reliability_table() (transfer-supported:
                            transfer r >= 0.20; within-section only: within r >= 0.30 with weaker transfer)
  conf_width_               90% quantile of absolute leave-one-section-out residuals, pooled over held-out
                            spots (log1p units; cross-fitted residuals); an empirical width, not a coverage
                            guarantee. uncertainty_ records its coverage on patients that took no part in
                            fitting, penalty selection or width estimation (reproduce/coverage_outer.py, read
                            from --coverage), and, separately, an internal check that reuses the same residuals
                            (widths from six sections' residuals, coverage on the seventh), which is not an
                            independent test because every residual comes from a model fitted with the other
                            sections. reproduce/update_shipped_uncertainty.py writes the same metadata into the
                            released pickle without refitting it.
Usage:
    python reproduce/build_shipped_model.py --data <DESIUM_DIR> \
        --within reproduce/results/within_desium_per_ion.npz --out metabocast_model.pkl \
        --coverage reproduce/results/coverage/coverage_outer.json \
        --curve reproduce/results/shipped_model/alpha_curve.json
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
from metabocast import MetaboCast, ALPHA_GRID          # noqa: E402
from loso_benchmark import slim, col_spearman    # noqa: E402

DESIUM = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
INTERNAL_NOTE = ("not independent: widths from six sections' cross-fitted residuals, coverage on the seventh, and every "
                 "residual comes from a model fitted with the other sections")


def outer_test_summary(path):
    """Coverage of the released width procedure on patients excluded from fitting, penalty selection and width
    estimation (written by reproduce/coverage_outer.py)."""
    outer = json.loads(Path(path).read_text())
    return {"design": "each patient held out of fitting, penalty selection and width estimation; widths from "
                      "cross-fitted residuals of the remaining patients (the released procedure)",
            "unit": outer["units"], "nominal": outer["nominal"],
            "mean_coverage_sections": round(outer["mean_coverage_crossfit_sections"], 4),
            "mean_coverage_patients": round(outer["mean_coverage_crossfit_patients"], 4),
            "per_section": {k: round(v["coverage_crossfit"], 4) for k, v in outer["per_section"].items()},
            "single_patient_split_calibration_mean": round(outer["mean_coverage_split_sections"], 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--within", type=Path, required=True, help="per-ion npz written by within_benchmark.py")
    ap.add_argument("--out", type=Path, default=ROOT / "metabocast_model.pkl")
    ap.add_argument("--q", type=float, default=0.90)
    ap.add_argument("--coverage", type=Path, default=ROOT / "reproduce/results/coverage/coverage_outer.json",
                    help="outer-test coverage written by reproduce/coverage_outer.py")
    ap.add_argument("--curve", type=Path, default=None,
                    help="also write the inner cross-validation score of every penalty in the grid (JSON)")
    args = ap.parse_args()
    secs = {s: slim(ad.read_h5ad(args.data / f"{s}.h5ad")) for s in DESIUM}
    with threadpool_limits(limits=1):
        t0 = time.time()
        model = MetaboCast().fit([secs[s] for s in DESIUM])
        print(f"final model: alpha={model.alpha_:g} ({model.alpha_selection_}), {len(model.genes_)} genes, "
              f"{len(model.mz_)} ions, {time.time() - t0:.0f}s", flush=True)
        n_ion = len(model.mz_)
        loso_r = np.full((len(DESIUM), n_ion), np.nan)
        resid = [[] for _ in range(n_ion)]
        for k, held in enumerate(DESIUM):
            t1 = time.time()
            m = MetaboCast().fit([secs[s] for s in DESIUM if s != held])
            pred = m.predict_metabolome(secs[held].copy())
            raw = np.asarray(secs[held].uns["msi"], float); y = np.log1p(raw)
            det = (raw > 0).mean(0) > 0.2
            ok = det & (y.std(0) > 1e-9) & (pred.std(0) > 1e-9)
            loso_r[k, ok] = col_spearman(pred[:, ok], y[:, ok])
            a = np.abs(y - pred)
            for j in range(n_ion):
                resid[j].append(a[:, j].astype(np.float32))
            print(f"  LOSO {held}: median r {np.nanmedian(loso_r[k, det]):.4f} ({time.time() - t1:.0f}s)", flush=True)
    model.conf_width_ = np.array([np.quantile(np.concatenate(r), args.q) for r in resid], np.float32)
    # Does a width learned on held-out sections cover a new section? For each section, widths from the
    # other six sections' held-out residuals; coverage measured on this section (spots x detected ions).
    cov = {}
    for k, held in enumerate(DESIUM):
        w = np.array([np.quantile(np.concatenate([r[i] for i in range(len(DESIUM)) if i != k]), args.q) for r in resid])
        det = (np.asarray(secs[held].uns["msi"]) > 0).mean(0) > 0.2
        cov[held] = float(np.mean([np.mean(resid[j][k] <= w[j]) for j in np.where(det)[0]]))
        print(f"  section-level width, coverage on held-out {held}: {cov[held]:.3f}", flush=True)
    print(f"internal check (shared residuals), mean coverage on the left-out section: {np.mean(list(cov.values())):.3f}", flush=True)
    outer_test = outer_test_summary(args.coverage)
    print(f"outer test (independent patients): mean coverage {outer_test['mean_coverage_sections']:.3f} over sections, "
          f"{outer_test['mean_coverage_patients']:.3f} over patients", flush=True)
    model.transfer_predictability_ = np.nanmean(loso_r, axis=0).astype(np.float32)
    W = np.load(args.within)
    within = np.vstack([W[f"MetaboCast__{s}"] for s in DESIUM])
    mz_w = np.asarray(W[f"mz__{DESIUM[0]}"], float)
    if mz_w.shape != model.mz_.shape or not np.allclose(mz_w, model.mz_, atol=1e-3):
        raise SystemExit("within-section per-ion file is not on the model's m/z axis")
    model.train_predictability_ = np.nanmean(within, axis=0).astype(np.float32)
    model.uncertainty_ = {"method": "leave-one-section-out residuals (7 DESIUM sections)", "coverage_requested": args.q,
                          "unit": "spot", "n_units": int(sum(len(r) for r in resid[0])), "coverage_guaranteed": False,
                          "outer_test_coverage": outer_test,
                          "internal_check_coverage": {"note": INTERNAL_NOTE, "per_section": cov},
                          "predictability": "train_predictability_ = within-section spatial-block CV Spearman (mean of 7 "
                                            "sections); transfer_predictability_ = leave-one-section-out Spearman (mean of 7)"}
    model.save(str(args.out))
    if args.curve:
        S = np.asarray(model.alpha_cv_scores_, float)                       # penalties x held-out sections
        args.curve.write_text(json.dumps(dict(alpha_grid=[float(a) for a in ALPHA_GRID],
                                              mean_over_heldout_sections=np.nanmean(S, 1).tolist(),
                                              per_section=S.tolist(), selected=float(model.alpha_),
                                              selection=model.alpha_selection_), indent=1))
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB); median within r "
          f"{np.nanmedian(model.train_predictability_):.3f}, median transfer r {np.nanmedian(model.transfer_predictability_):.3f}")


if __name__ == "__main__":
    main()
