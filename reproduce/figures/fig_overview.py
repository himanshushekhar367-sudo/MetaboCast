"""Figure 1 (design), the within-section benchmark on both paired cohorts and the comparison with
transcriptome-only metabolic inference (supplementary figures; numbers in reproduce/manifest.py)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import BLUE, GOLD, GREY, LIGHTBLUE, ORANGE, DESIUM, SMA_GROUPS, J, letter, res, save


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.1)); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.set_axis_off()

    def box(x, y, w, h, col, title, lines):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.8", fc="white", ec=col, lw=2.2))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=8, fontweight="bold")
        ax.text(x + w / 2, y + h - (10.2 if "\n" in title else 7.0), "\n".join(lines), ha="center", va="top", fontsize=6.6,
                linespacing=1.25 if len(lines) > 4 else 1.35)

    def arrow(x0, y0, x1, y1, col):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
    box(1, 24, 21, 18, "#2F5F98", "Paired training", ["spatial transcriptomics", "+ MSI (DESI or MALDI)", "", "Query: transcriptome only",
                                                       "(Visium, HD, Xenium, ...)"])
    box(26.5, 24, 22, 18, "#4C9BA6", "Spatial features", ["HVG → PCA (100 PCs)", "diffusion over the spot graph",
                                                         "[P, ÂP, Â²P]", "optional: KEGG pathways,", "H&E, protein, cell types"])
    box(53, 24, 19, 18, "#7B6BA8", "Model", ["rank-normalised targets", "ridge + extremely", "randomized trees", "", "uncertainty widths"])
    box(76.5, 24, 22.5, 18, "#B5413A", "Predicted ion maps", ["one value per m/z channel", "and spot (2,086 here)", "+ uncertainty width",
                                                          "+ reliability tier", "+ putative candidates"])
    arrow(22.4, 33, 26.1, 33, "#2F5F98"); arrow(48.9, 33, 52.6, 33, "#4C9BA6"); arrow(72.4, 33, 76.1, 33, "#7B6BA8")
    box(12, 1, 34, 18, "#C49A3A", "Validation (paired data)", ["DESIUM (DESI) and SMA (MALDI)", "spatial blocks; held-out sections,",
                                                              "patients and animals", "baselines: coordinates, joint NMF,",
                                                              "KEGG features, scFEA, MGFEA"])
    box(54, 1, 34, 18, "#3B8B5A", "Application (no MSI)", ["transcriptome-only HCC and GBM", "ions ranked by transfer evidence",
                                                           "", "predictions are hypotheses", "for targeted MSI or MS/MS"])
    arrow(58, 23.6, 34, 19.6, "#C49A3A"); arrow(67, 23.6, 70, 19.6, "#3B8B5A")
    save(fig, "Figure1")
    return {}


COHORTS = [("DESIUM", "DESIUM\n(DESI, tumour)", DESIUM, "within_desium"),
           ("DHB", "SMA-DHB\n(lipids)", ["SMA_DHB_V11L12-038_A1", "SMA_DHB_V11L12-038_B1"], "within_sma"),
           ("9-AA", "SMA-9AA\n(metabolites)", ["SMA_9AA_V11L12-038_D1"], "within_sma"),
           ("FMP", "SMA-FMP\n(neurotransmitters)", SMA_GROUPS["fmp_mouse"] + SMA_GROUPS["fmp_human"], "within_sma")]
S8_METHODS = [("MetaboCast", "MetaboCast", BLUE), ("MetaboCast (ridge only)", "ridge only", "#5C8BC2"),
              ("KEGG pathway features + supervised regressor", "KEGG features + regressor", GOLD), ("joint-NMF", "joint NMF", "#C0603A"),
              ("no diffusion", "no diffusion", LIGHTBLUE), ("single-gene OLS", "single-gene OLS", "#BBBBBB")]


def s8():
    rs = {k: J(res(f"{k}.json")) for k in ("within_desium", "within_sma")}
    fig = plt.figure(figsize=(7.2, 2.9)); ax = fig.add_axes([0.07, 0.30, 0.6, 0.63]); w = 0.135; out = {}
    for ci, (key, lab, secs, src) in enumerate(COHORTS):
        secs = [s for s in secs if s in rs[src]]
        out[key] = {}
        for mi, (m, mlab, col) in enumerate(S8_METHODS):
            v = np.array([rs[src][s]["scores"].get(m, np.nan) for s in secs], float); v = v[np.isfinite(v)]
            x = ci + (mi - 2.5) * w
            if not len(v):
                ax.text(x, 0.005, "n/a", ha="center", fontsize=4.5, rotation=90); continue
            mu, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0)
            out[key][mlab] = (float(mu), float(sd), int(len(v)))
            ax.bar(x, mu, w, yerr=sd if sd else None, color=col, lw=0, ecolor="k",
                   error_kw=dict(lw=0.5, capsize=1.5), label=mlab if ci == 0 else None)
    ax.set_xticks(range(len(COHORTS)))
    ax.set_xticklabels([f"{c[1]}\nn={len([s for s in c[2] if s in rs[c[3]]])}" for c in COHORTS], fontsize=6); ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("within-section median Spearman")
    ax.legend(frameon=False, fontsize=5.5, ncol=6, loc="upper center", bbox_to_anchor=(0.62, -0.33))
    letter(ax, "a", x=-0.1, y=1.0)
    ax = fig.add_axes([0.76, 0.30, 0.22, 0.63]); gains = []
    for key, lab, secs, src in COHORTS:
        secs = [s for s in secs if s in rs[src]]
        gains.append(float(np.mean([rs[src][s]["scores"]["MetaboCast"] - rs[src][s]["scores"]["no diffusion"] for s in secs])))
    ax.bar(range(len(gains)), gains, color=BLUE)
    for i, v in enumerate(gains):
        ax.text(i, v + 0.003, f"{v:+.3f}", ha="center", fontsize=5.8)
    ax.set_xticks(range(len(gains))); ax.set_xticklabels([c[0] for c in COHORTS], fontsize=6); ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Δ median Spearman\n(MetaboCast − no diffusion)")
    letter(ax, "b", x=-0.4, y=1.0)
    out["diffusion_gain"] = dict(zip([c[0] for c in COHORTS], gains))
    save(fig, "SuppFig_SMA_within")
    return out


def s9():
    T = pd.read_csv(res("baselines", "baseline_comparison_per_ion.csv"))
    rel = T.metabocast_within >= 0.30
    fig = plt.figure(figsize=(7.2, 3.0))
    ax = fig.add_axes([0.07, 0.17, 0.36, 0.73])
    cols = ["metabocast_within", "metabocast_loso", "scFEA_r"] + (["MGFEA_r"] if T.MGFEA_r.notna().any() else [])
    labs = ["MetaboCast\nwithin-section", "MetaboCast\nleave-one-\nsection-out", "scFEA\n(no MSI)", "MGFEA\n(no MSI)"][:len(cols)]
    colr = [BLUE, LIGHTBLUE, GOLD, ORANGE]
    rng = np.random.default_rng(0)
    for i, c in enumerate(cols):
        v = T[c].dropna().values
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="k"))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=8, color=colr[i], lw=0, alpha=0.8, zorder=3)
    ax.axhline(0, color="grey", lw=0.6, ls="--"); ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylabel("per-ion Spearman vs measured MSI"); ax.set_title(f"{len(T)} DESIUM ions matched to scFEA compounds")
    letter(ax, "a", x=-0.2, y=1.03)
    ax = fig.add_axes([0.56, 0.17, 0.4, 0.73])
    ax.scatter(T.scFEA_r[~rel], T.metabocast_loso[~rel], s=12, color=GREY, label="within-section r < 0.30")
    ax.scatter(T.scFEA_r[rel], T.metabocast_loso[rel], s=14, color=BLUE, label="within-section r ≥ 0.30")
    hi = max(0.35, float(np.nanmax(T.metabocast_loso)) + 0.05)
    lim = [-0.25, hi]; ax.plot(lim, lim, color="k", lw=0.6, ls=":"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    top = T[rel].nlargest(5, "metabocast_loso")             # the five best-transferred of the blue ions
    multi = T.annotation.value_counts()                      # compounds that match more than one channel get the adduct
    dup = [multi[a_] > 1 for a_ in top.annotation]
    placed = []
    for (_, r), d in zip(top.iterrows(), dup):
        crowded = any(abs(r.scFEA_r - x) < 0.12 and abs(r.metabocast_loso - y) < 0.03 for x, y in placed)
        name = r.annotation[0].lower() + r.annotation[1:] if not r.annotation[:2].isupper() else r.annotation
        ax.annotate(f"{name} {r.adduct}".replace("-", "−") if d else name, (r.scFEA_r, r.metabocast_loso), fontsize=5.5,
                    xytext=(-4, -2) if crowded else (3, 2), ha="right" if crowded else "left", va="top" if crowded else "bottom",
                    textcoords="offset points")
        placed.append((r.scFEA_r, r.metabocast_loso))
    ax.set_xlabel("scFEA estimate vs measured (Spearman)"); ax.set_ylabel("MetaboCast held-out section vs measured")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right"); ax.set_title("Transfer without the section's MSI")
    letter(ax, "b", x=-0.2, y=1.03)
    save(fig, "SuppFig_flux")
    return dict(n=int(len(T)), n_reliable=int(rel.sum()))


def run():
    return {"Figure1": fig1(), "SuppFig_SMA_within": s8(), "SuppFig_flux": s9()}
