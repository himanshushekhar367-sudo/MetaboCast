"""Tests behind the P values of the Spearman correlations reported in the manuscript (Additional file 2: Table S21, block G).

Block G holds the tests that sit outside the Holm-adjusted families A-F and are reported unadjusted:

  * accuracy against zero: one-sample exact two-sided Wilcoxon signed-rank tests over the seven DESIUM sections, or over
    the 39 ions of the flux comparison;
  * control comparisons: MetaboCast against the coordinate-only regressor over all detected ions, the whole-section
    against the training-block ion panel, the transcriptome against H&E features within sections, and H&E features
    against shuffled H&E features;
  * Spearman correlation tests across ion-section pairs (interval width and Moran's I against per-ion accuracy);
  * correlations between two maps over the spots of one section (measured against held-out predicted ion maps, and
    predicted ions against antibody counts on the GBM CytAssist section), tested with the modified t-test of
    Dutilleul P, Clifford P, Richardson S, Hemon D (Biometrics 1993;49:305-14) applied to ranks. Neighbouring spots are
    not independent, so the number of spots is replaced by an effective sample size estimated from the spatial
    autocorrelation of both maps in 13 equal-width distance classes up to the largest inter-spot distance. The
    implementation follows the R package SpatialPack (modified.ttest, default 13 classes) and reproduces its effective
    sample sizes and P values.

usage: python reproduce/spatial_correlation_tests.py [--results reproduce/results]
writes <results>/rigor/spatial_correlation_tests.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f as fdist, rankdata, spearmanr, wilcoxon

DESIUM = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
FAMILY = "G. Further tests, reported unadjusted"
WX1 = "exact two-sided Wilcoxon signed-rank (one-sample)"
WX = "exact two-sided Wilcoxon signed-rank"


class SpatialCorrelationTest:
    """Dutilleul's modified t-test for the correlation of two maps observed at the same spots."""

    def __init__(self, coords, nclass=13):
        d = squareform(pdist(np.asarray(coords, float)))
        self.n = n = len(d); self.nclass = nclass
        bounds = d.max() / nclass * np.arange(1, nclass + 1)
        cls = np.searchsorted(bounds, d, side="left").clip(max=nclass - 1)       # first upper bound >= distance
        np.fill_diagonal(cls, -1)
        self.masks = [cls == k for k in range(nclass)]
        self.card = np.array([m.sum() for m in self.masks], float)              # ordered spot pairs per class
        self.rowcnt = np.stack([m.sum(1) for m in self.masks], 1).astype(float)

    def _moran(self, z):
        z = z - z.mean(); v = (z ** 2).mean()
        return np.array([(m @ z) @ z / c / v if c else 0.0 for m, c in zip(self.masks, self.card)])

    def test(self, x, y):
        """Spearman correlation of x and y with its spatially corrected P value."""
        x, y = rankdata(x), rankdata(y)
        r = float(np.corrcoef(x, y)[0, 1]); ix, iy = self._moran(x), self._moran(y); n = self.n
        rx, ry = 1.0 + self.rowcnt @ ix, 1.0 + self.rowcnt @ iy                    # row sums of the correlation matrices
        sxx, syy, sxy = rx.sum(), ry.sum(), (rx * ry).sum()
        trxy = n + (self.card * ix * iy).sum() + (sxx * syy / n - 2.0 * sxy) / n
        ess = (n - sxx / n) * (n - syy / n) / trxy + 1.0; df = ess - 2.0
        return dict(rho=r, ess=float(ess), p=float(fdist.sf(df * r * r / (1.0 - r * r), 1.0, df)), n=n)


def boot(d, stat=np.mean, n=5000, seed=0):
    d = np.asarray(d, float); r = np.random.RandomState(seed)
    m = np.array([stat(d[i]) for i in r.randint(0, len(d), (n, len(d)))])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pval(p):
    """Small P values as text so that the 4-decimal rounding of the table builder keeps them."""
    return float(p) if p >= 1e-3 else f"{p:.2e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    R = ap.parse_args().results
    J = lambda *p: json.load(open(os.path.join(R, *p)))
    rows = []

    def add(comparison, unit, n, effect, ci, improved, test, p):
        rows.append(dict(family=FAMILY, comparison=comparison, unit=unit, n=int(n), effect=float(effect), ci=ci,
                         improved=improved, test=test, p=p))

    def against_zero(label, unit, v, stat):
        v = np.asarray(v, float)
        add(label, unit, len(v), stat(v), boot(v, stat), f"{int((v > 0).sum())}/{len(v)} above 0", WX1, pval(wilcoxon(v).pvalue))

    def paired(label, a, b):
        d = np.asarray(a, float) - np.asarray(b, float)
        add(label, "section", len(d), d.mean(), boot(d), f"{int((d > 0).sum())}/{len(d)}", WX, pval(wilcoxon(d).pvalue))

    # accuracy against zero and control comparisons, DESIUM within sections (Table S2)
    ctrl = J("within_controls.json")
    pooled = lambda m, k: [ctrl[s]["pooled"][m][k] for s in DESIUM]
    against_zero("MetaboCast within-section accuracy against zero", "section", pooled("MetaboCast", "full_panel"), np.mean)
    paired("MetaboCast − coordinate-only kNN, all detected ions", pooled("MetaboCast", "all_detected"),
           pooled("coordinates only", "all_detected"))
    pf = {s: pd.DataFrame(ctrl[s]["per_fold"]) for s in DESIUM}
    paired("Whole-section − training-block ion panel, per-block accuracy", [pf[s].metabocast_full_panel.mean() for s in DESIUM],
           [pf[s].metabocast_train_panel.mean() for s in DESIUM])
    # the 39 ions of the flux comparison (Table S3)
    fl = pd.read_csv(os.path.join(R, "baselines", "baseline_comparison_per_ion.csv"))
    for col, label in [("scFEA_r", "scFEA per-ion r against zero"), ("MGFEA_r", "MGFEA per-ion r against zero"),
                       ("metabocast_loso", "MetaboCast held-out-section per-ion r against zero (39 ions)")]:
        against_zero(label, "ion", fl[col].dropna(), np.median)
    # held-out-section and held-out-patient accuracy (Table S7)
    against_zero("MetaboCast held-out-section accuracy against zero", "section",
                 [J("loso_desium.json")["per_section"][s]["median"] for s in DESIUM], np.mean)
    cov = J("coverage", "coverage_outer.json")["per_section"]
    against_zero("MetaboCast held-out-patient accuracy against zero", "section", [cov[s]["lopo_median_r"] for s in DESIUM], np.mean)
    # histology controls (Table S9)
    hw = pd.read_csv(os.path.join(R, "ablations", "histology_within.csv"))
    ho = pd.read_csv(os.path.join(R, "ablations", "histology_only_desium.csv"))
    secs = list(ho[ho.test == "within-sample 5-fold"].section)
    get = lambda df, cond, s: float(df[cond & (df.iloc[:, 1] == s)].median_spearman.iloc[0])
    paired("Transcriptome-only − H&E-only, within sections",
           [get(hw, hw.model == "MetaboCast", s) for s in secs], [get(ho, ho.test == "within-sample 5-fold", s) for s in secs])
    paired("H&E-only − H&E-only with shuffled features, held-out sections",
           [get(ho, ho.test == "LOSO (cross-section)", s) for s in secs], [get(ho, ho.test.str.startswith("shuffle"), s) for s in secs])
    # Spearman correlation tests across ion-section pairs
    C = np.load(os.path.join(R, "within_desium_conformal.npz")); W = np.load(os.path.join(R, "within_desium_per_ion.npz"))
    xs, ys, per = [], [], []
    for s in DESIUM:
        d = C[f"{s}__dets"]; acc = W[f"MetaboCast__{s}"][d].astype(float); rw = C[f"{s}__relwidth"]; ok = np.isfinite(acc) & np.isfinite(rw)
        xs.append(rw[ok]); ys.append(acc[ok]); per.append(spearmanr(rw[ok], acc[ok]))
    add("Relative interval width vs per-ion accuracy (Fig. 4b)", "ion–section pair", sum(map(len, xs)),
        spearmanr(np.concatenate(xs), np.concatenate(ys))[0],
        f"within sections {min(p[0] for p in per):+.2f} to {max(p[0] for p in per):+.2f}", "n/a", "Spearman correlation test",
        "< 1e-100 in each section" if max(p[1] for p in per) < 1e-100 else pval(max(p[1] for p in per)))
    pdt = pd.read_csv(os.path.join(R, "rigor", "predictability_determinants.csv")).dropna(subset=["moran", "r"])
    per = [spearmanr(g.moran, g.r) for _, g in pdt.groupby("sec")]
    add("Moran's I vs per-ion within-section accuracy (Additional file 1: Figure S5a)", "ion–section pair", len(pdt),
        spearmanr(pdt.moran, pdt.r)[0], f"within sections {min(p[0] for p in per):+.2f} to {max(p[0] for p in per):+.2f}", "n/a",
        "Spearman correlation test", "< 1e-100 in each section" if max(p[1] for p in per) < 1e-100 else pval(max(p[1] for p in per)))
    # correlations between maps over spots
    def map_test(label, coords, x, y):
        t = SpatialCorrelationTest(coords).test(x, y)
        add(label, "spot", t["n"], t["rho"], "n/a", "n/a", f"Dutilleul modified t-test on ranks (effective n {t['ess']:.0f})", pval(t["p"]))
    M = np.load(os.path.join(R, "within_desium_maps.npz"))
    for ion in ["EPA", "Xanthine", "Fumarate", "Tryptophan"]:
        map_test(f"LC_091 measured vs held-out predicted, {ion if ion == 'EPA' else ion.lower()} [M−H]−", C["LC_091__coords"],
                 M[f"LC_091__{ion}__measured"], M[f"LC_091__{ion}__predicted"])
    F = np.load(os.path.join(R, "sma", "loso_fmp_mouse_maps.npz"))
    for sec, label in [("SMA_FMP_V11T16-085_A1", "substantia nigra V11T16-085_A1"), ("SMA_FMP_V11L12-109_A1", "striatum V11L12-109_A1")]:
        m, p, c = F[f"{sec}__FMP557__measured"], F[f"{sec}__FMP557__predicted"], F[f"{sec}__FMP557__coords"]
        ok = np.isfinite(m) & np.isfinite(p)
        map_test(f"FMP m/z 557.2, {label} (held-out section), measured vs predicted", c[ok], m[ok], p[ok])
    G = np.load(os.path.join(R, "crosscancer", "GBM_CytAssist_panel.npz"))
    for i, ion in enumerate(G["ions"]):
        cor = [spearmanr(G["pred"][:, i], G["protein"][:, j])[0] for j in range(G["protein"].shape[1])]
        j = int(np.nanargmax(np.abs(cor)))
        map_test(f"GBM CytAssist, predicted {ion} vs {G['protein_names'][j]} (strongest marker for this ion)", G["coords"],
                 G["pred"][:, i], G["protein"][:, j])
    out = os.path.join(R, "rigor", "spatial_correlation_tests.json")
    json.dump(rows, open(out, "w"), indent=1)
    for r in rows:
        print(f"{r['comparison'][:72]:72s} n={r['n']:<6} effect={r['effect']:+.4f} P={r['p']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
