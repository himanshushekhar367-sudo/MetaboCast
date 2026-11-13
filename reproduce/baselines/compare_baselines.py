"""Compare MetaboCast (within-section CV and leave-one-section-out) with transcript-only metabolic inference
(scFEA, MGFEA) on DESIUM ions whose negative-mode annotation matches an scFEA compound
(flux_baselines table and SuppFig_flux; numbers in reproduce/manifest.py).

Inputs: the tool outputs written by run_scfea.sh / run_mgfea.py (in $METABOCAST_BASELINE_DIR), the ion
annotation (annotate_ions.py) and MetaboCast's per-ion scores from within_benchmark.py and loso_benchmark.py.
Run: python reproduce/baselines/compare_baselines.py [--results reproduce/results] [--out reproduce/results/baselines]
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths as P  # noqa: E402  data and work locations; see reproduce/paths.py
import numpy as np, pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--results", default=os.path.join(P.REPO, "reproduce", "results"))
ap.add_argument("--out", default=None, help="output folder (default: $METABOCAST_BASELINE_DIR)")
args = ap.parse_args()
B = P.BASELINES
OUT = args.out or B
os.makedirs(OUT, exist_ok=True)
samples = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
ann = pd.read_csv(os.path.join(args.results, "desium_ion_annotation.csv")); ann["idx"] = np.arange(len(ann))
W = np.load(os.path.join(args.results, "within_desium_per_ion.npz"))
L = np.load(os.path.join(args.results, "loso_desium_per_ion.npz"))
with np.errstate(all="ignore"):
    mc_within = np.nanmean(np.vstack([W[f"MetaboCast__{s}"] for s in samples]), axis=0)
    mc_loso = np.nanmean(np.vstack([L[s] for s in samples]), axis=0)
neg = ann[(ann.n_candidates > 0) & ~ann.ambiguous.astype(bool)]   # single negative-mode candidate only
# DESIUM annotation -> scFEA compound name
M = {"AMP": "AMP", "Glutamate": "Glutamate", "Aspartate": "Aspartate", "Glutathione": "Glutathione", "Arginine": "Arginine",
     "Glutamine": "Glutamine", "Serine": "Serine", "Methionine": "Methionine", "Phenylalanine": "Phenylalanine", "Tyrosine": "Tyrosine",
     "Cysteine": "Cysteine", "Fumarate": "Fumarate", "Leucine/Ile": "Leucine", "Histidine": "Histidine", "Proline": "Proline",
     "Malate": "Malate", "Citrate/isocitrate": "Citrate", "Cholesterol": "Cholesterol", "Threonine": "Threonine", "Lactate": "Lactate",
     "Hypoxanthine": "Hypoxanthine", "Hexose": "Glucose", "Xanthine": "Xanthine", "Hexose-P": "G6P", "UMP": "UMP",
     "Lysine": "lysine", "FA16:0": "Fatty Acid"}
neg = neg[neg.annotation.isin(M)].copy(); neg["compound"] = neg.annotation.map(M)
rows = []
for _, r in neg.iterrows():
    j = int(r.idx); rec = dict(mz=r.mz, annotation=r.annotation, adduct=r.adduct, cls=r["class"], compound=r.compound,
                               metabocast_within=mc_within[j], metabocast_loso=mc_loso[j])
    for tool in ("scFEA", "MGFEA"):
        vals = []
        for s in samples:
            Y = np.load(f"{B}/input/{s}_msi.npy")[:, j]
            if (np.expm1(Y) > 0).mean() <= 0.2 or Y.std() < 1e-9: continue
            if tool == "scFEA":
                f = f"{B}/output/{s}_balance.csv"
                if not os.path.exists(f): continue
                bal = pd.read_csv(f, index_col=0)
                if r.compound not in bal.columns: continue
                x = bal[r.compound].values
            else:
                f = f"{B}/mgfea_out/{s}_balance.npy"
                if not os.path.exists(f): continue
                comp = open(f"{B}/mgfea_out/{s}_compounds.txt").read().split("\n")
                names = [c.split(";")[-1].strip() for c in comp]
                k = [i for i, n in enumerate(names) if n.lower() == r.compound.lower()]
                if not k: continue
                x = np.load(f)[:, k[0]]
            if np.std(x) < 1e-12: vals.append(0.0); continue
            vals.append(spearmanr(x, Y)[0])
        rec[f"{tool}_r"] = np.nanmean(vals) if vals else np.nan
        rec[f"{tool}_n_sections"] = len(vals)
    rows.append(rec)
T = pd.DataFrame(rows).sort_values("metabocast_within", ascending=False)
T.to_csv(os.path.join(OUT, "baseline_comparison_per_ion.csv"), index=False)
summ = {}
for tool in ("scFEA", "MGFEA"):
    t = T.dropna(subset=[f"{tool}_r"])
    if len(t) < 3: continue
    d_loso = t.metabocast_loso - t[f"{tool}_r"]; d_with = t.metabocast_within - t[f"{tool}_r"]
    summ[tool] = dict(n_ions=len(t), n_sections_mean=float(t[f"{tool}_n_sections"].mean()),
                      tool_median=float(t[f"{tool}_r"].median()), tool_mean=float(t[f"{tool}_r"].mean()),
                      tool_abs_median=float(t[f"{tool}_r"].abs().median()),
                      mc_loso_median=float(t.metabocast_loso.median()), mc_within_median=float(t.metabocast_within.median()),
                      loso_wins=int((d_loso > 0).sum()), within_wins=int((d_with > 0).sum()),
                      p_loso=float(wilcoxon(d_loso).pvalue), p_within=float(wilcoxon(d_with).pvalue))
json.dump(summ, open(os.path.join(OUT, "baseline_summary.json"), "w"), indent=1)
print(T.round(3).to_string()); print(json.dumps(summ, indent=1))
