#!/usr/bin/env python3
"""Train MetaboCast on every paired cohort in a folder and report leave-one-section-out accuracy.

Every ``.h5ad`` file under DATA_DIR that holds a paired section (gene expression, ``uns['msi']``,
``uns['mz_features']`` and ``obsm['spatial']``) is loaded. Sections are grouped into cohorts that can be
pooled: the same m/z axis (compared by value) and the same gene-symbol convention (human-style upper case
or mouse-style title case). For every cohort with at least two sections the script

  1. fits MetaboCast on all sections and saves the model (``<out>/metabocast_<cohort>.pkl``);
  2. runs leave-one-section-out: fit on the other sections, predict the held-out one and score each ion
     detected in more than 20% of its spots by Spearman correlation (detection-matched median, as in the
     manuscript);
  3. writes ``<out>/summary.json`` and ``<out>/loso_per_section.csv``.

Usage:
    python pipeline/train_and_evaluate.py DATA_DIR [--out results/training] [--use-kegg]
                                          [--estimator ensemble|ridge] [--n-hvg 3000] [--no-loso]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metabocast import MetaboCast, group_sections_by_mz_axis   # noqa: E402
from metabocast.io import symbol_case                         # noqa: E402


def is_paired(a):
    return "msi" in a.uns and "mz_features" in a.uns and "spatial" in a.obsm


def slim(a):
    """Keep what fitting and scoring need, as compact arrays (large sections fit in memory together)."""
    x = a.layers["log1p"] if "log1p" in a.layers else a.X
    b = ad.AnnData(X=csr_matrix(x, dtype=np.float32))
    b.var_names = list(map(str, a.var_names)); b.obs_names = list(map(str, a.obs_names))
    b.var_names_make_unique()
    b.layers["log1p"] = b.X
    b.uns["msi"] = np.asarray(a.uns["msi"], np.float32)
    b.uns["mz_features"] = np.asarray(a.uns["mz_features"], float)
    b.obsm["spatial"] = np.asarray(a.obsm["spatial"], float)[:, :2]
    return b


def col_spearman(P, Y):
    R1 = rankdata(P, axis=0); R2 = rankdata(Y, axis=0)
    R1 -= R1.mean(0); R2 -= R2.mean(0)
    den = np.sqrt((R1 * R1).sum(0) * (R2 * R2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (R1 * R2).sum(0) / den
    r[~(den > 1e-12)] = np.nan
    return r


def detection_matched_median(pred, raw, min_detection=0.2):
    det = (raw > 0).mean(0) > min_detection
    if not det.any():
        return float("nan"), 0
    r = col_spearman(pred[:, det], np.log1p(raw[:, det]))
    return float(np.nanmedian(r)), int(det.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path, help="folder searched recursively for paired .h5ad sections")
    ap.add_argument("--out", type=Path, default=Path("results/training"))
    ap.add_argument("--use-kegg", action="store_true", help="add the KEGG metabolic-pathway prior")
    ap.add_argument("--estimator", choices=["ensemble", "ridge"], default="ensemble")
    ap.add_argument("--n-hvg", type=int, default=3000)
    ap.add_argument("--no-loso", action="store_true", help="only fit and save one model per cohort")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sections, names = [], []
    for p in sorted(args.data.rglob("*.h5ad")):
        if args.out.resolve() in p.resolve().parents:
            continue
        try:
            a = ad.read_h5ad(p)
        except Exception as e:                      # unreadable file: report and continue
            print(f"  skip {p}: {e}")
            continue
        if not is_paired(a):
            print(f"  skip {p.name}: not a paired section (needs uns['msi'], uns['mz_features'], obsm['spatial'])")
            continue
        sections.append(slim(a)); names.append(p.stem)
        print(f"  found {p.name}: {a.n_obs} spots, {a.n_vars} genes, {np.asarray(a.uns['msi']).shape[1]} ions")
        del a
    if not sections:
        raise SystemExit("no paired sections found")

    cohorts = []
    for idx in group_sections_by_mz_axis(sections):
        by_case = {}
        for i in idx:
            by_case.setdefault(symbol_case(list(sections[i].var_names)), []).append(i)
        cohorts += list(by_case.items())

    summary, rows = {}, []
    for case, idx in cohorts:
        n_ion = sections[idx[0]].uns["msi"].shape[1]
        tag = f"{n_ion}ions_{case}_n{len(idx)}"
        members = [names[i] for i in idx]
        print(f"\ncohort {tag}: {', '.join(members)}")
        if len(idx) < 2:
            print("  a single section: fitted with spatial-block cross-validation; no leave-one-section-out score")
        kw = dict(n_hvg=args.n_hvg, estimator=args.estimator, use_kegg=args.use_kegg)
        t0 = time.time()
        model = MetaboCast(**kw).fit([sections[i] for i in idx])
        path = args.out / f"metabocast_{tag}.pkl"
        model.save(path)
        print(f"  saved {path} (alpha {model.alpha_:g}, {model.alpha_selection_}; {time.time() - t0:.0f}s)")
        res = dict(sections=members, n_ions=int(n_ion), symbol_case=case, model=str(path), alpha=float(model.alpha_))
        if len(idx) >= 2 and not args.no_loso:
            loso = {}
            for h in idx:
                m = MetaboCast(**kw).fit([sections[i] for i in idx if i != h])
                pred = m.predict_metabolome(sections[h].copy())
                r, n_det = detection_matched_median(pred, np.asarray(sections[h].uns["msi"], float))
                loso[names[h]] = r
                rows.append(dict(cohort=tag, section=names[h], loso_median_spearman=r, n_detected_ions=n_det))
                print(f"  held out {names[h]:28s} median Spearman {r:+.3f} ({n_det} detected ions)")
            v = np.array(list(loso.values()), float)
            res.update(loso=loso, loso_mean=float(np.nanmean(v)),
                       loso_sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"))
            print(f"  leave-one-section-out mean {res['loso_mean']:+.3f}")
        summary[tag] = res

    json.dump(summary, open(args.out / "summary.json", "w"), indent=2)
    if rows:
        pd.DataFrame(rows).to_csv(args.out / "loso_per_section.csv", index=False)
    print(f"\nwrote {args.out / 'summary.json'}" + (f" and {args.out / 'loso_per_section.csv'}" if rows else ""))


if __name__ == "__main__":
    main()
