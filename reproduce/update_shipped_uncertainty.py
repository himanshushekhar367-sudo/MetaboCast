#!/usr/bin/env python3
"""Write the coverage metadata of the released model (metabocast_model.pkl) without refitting it.

model.uncertainty_ then reports (i) outer_test_coverage, the coverage of the released width procedure on patients
excluded from fitting, penalty selection and width estimation (reproduce/coverage_outer.py), and (ii)
internal_check_coverage, the earlier per-section check that reuses the same cross-fitted residuals and is therefore
not an independent test. Only this metadata changes: the fitted arrays, widths and predictions are untouched, and
the scikit-learn version recorded with the fitted estimators is kept, because their state is the one saved with
that version. A full rebuild with reproduce/build_shipped_model.py writes the same metadata.

Usage:
    python reproduce/update_shipped_uncertainty.py [--model metabocast_model.pkl]
        [--coverage reproduce/results/coverage/coverage_outer.json]
"""
import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
from build_shipped_model import INTERNAL_NOTE, outer_test_summary      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=ROOT / "metabocast_model.pkl")
    ap.add_argument("--coverage", type=Path, default=ROOT / "reproduce/results/coverage/coverage_outer.json")
    args = ap.parse_args()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = pickle.loads(args.model.read_bytes())
    saved = sorted({str(getattr(w.message, "original_sklearn_version", "")) for w in caught} - {""})
    m = obj["model"] if isinstance(obj, dict) else obj
    u = dict(m.uncertainty_)
    internal = u.pop("heldout_section_coverage", None)
    if internal is None:
        internal = u.get("internal_check_coverage", {}).get("per_section")
    u["outer_test_coverage"] = outer_test_summary(args.coverage)
    u["internal_check_coverage"] = {"note": INTERNAL_NOTE, "per_section": internal}
    order = ["method", "coverage_requested", "unit", "n_units", "coverage_guaranteed", "outer_test_coverage",
             "internal_check_coverage", "predictability"]
    m.uncertainty_ = {k: u[k] for k in order if k in u} | {k: v for k, v in u.items() if k not in order}
    if len(saved) == 1:                       # keep the version stamp of the unchanged fitted estimators
        import sklearn.base
        sklearn.base.__version__ = saved[0]
    args.model.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    o = u["outer_test_coverage"]
    print(f"updated {args.model.name}: outer-test coverage {o['mean_coverage_sections']:.3f} over sections, "
          f"{o['mean_coverage_patients']:.3f} over patients (nominal {o['nominal']}); estimators stamped "
          f"scikit-learn {saved[0] if len(saved) == 1 else 'as installed'}")


if __name__ == "__main__":
    main()
