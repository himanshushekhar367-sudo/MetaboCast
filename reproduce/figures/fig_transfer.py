"""Figures 5 and 6 and the transfer supplementary figures (SMA neurotransmitter maps, transcriptome harmonisation):
held-out-section and held-out-patient transfer (reproduce/loso_benchmark.py, reproduce/loso_baselines.py,
reproduce/coverage_outer.py, reproduce/histology_ablation.py, reproduce/histology_only_ablation.py,
reproduce/per_ion_reliability.py, reproduce/harmonization.py) and the HCC contrasts of the cross-cancer run."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

from scipy.spatial import cKDTree

from common import (BLUE, GOLD, GREY, LIGHTBLUE, ORANGE, RED, DESIUM, SHORT, SMA_GROUPS, NIGRA, OUT, J, ann_table, fmt_r,
                    letter, loso_medians, map_key, pretty, res, save, spotmap, spotmap_masked)

TRANSFER_MIN, WITHIN_MIN = 0.20, 0.30
TIER_COL = {"transfer-supported": BLUE, "within-section only": GOLD, "low": "#C8C8C8"}


def sma_loso(tag=""):
    return {g: loso_medians(f"sma/loso_{g}{tag}") for g in SMA_GROUPS}


def within_all_detected(name, sections):
    w = J(res(f"{name}.json"))
    return {s: w[s]["full_median"] for s in sections if s in w}


def transfer_methods():
    """Per-section held-out accuracy (detection-matched median Spearman) of MetaboCast and the paired-supervision
    baselines under the same leave-one-section-out design, and MetaboCast with whole patients held out."""
    loso = loso_medians("loso_desium")
    cov = J(res("coverage", "coverage_outer.json"))["per_section"]
    out = [("MetaboCast,\nsection\nheld out", [loso[s] for s in DESIUM], BLUE),
           ("MetaboCast,\npatient\nheld out", [cov[s]["lopo_median_r"] for s in DESIUM], LIGHTBLUE)]
    b = J(res("loso_baselines.json"))["per_section"] if os.path.exists(res("loso_baselines.json")) else {}
    if all(s in b for s in DESIUM):
        for key, lab, col in [("graph-free", "no\ndiffusion", GREY),
                              ("KEGG features", "KEGG\nfeatures +\nregressor", GOLD),
                              ("joint NMF", "joint\nNMF", "#C0603A"), ("single gene", "single-\ngene\nOLS", "#BBBBBB")]:
            out.append((lab, [b[s][key] for s in DESIUM], col))
    return out


def tier(w, t):
    if np.isfinite(t) and t >= TRANSFER_MIN:
        return "transfer-supported"
    if np.isfinite(w) and w >= WITHIN_MIN:
        return "within-section only"
    return "low" if (np.isfinite(w) or np.isfinite(t)) else "uncalibrated"


def fig5():
    fig = plt.figure(figsize=(7.2, 5.6)); summ = {}
    # a: transfer of MetaboCast and paired-supervision baselines
    ax = fig.add_axes([0.075, 0.60, 0.46, 0.33]); rng = np.random.default_rng(2)
    meths = transfer_methods(); allv = []
    for i, (lab, v, col) in enumerate(meths):
        v = np.asarray(v, float); allv += list(v)
        ax.bar(i, v.mean(), 0.68, color=col, zorder=2)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color="k", lw=0.6, capsize=1.8, zorder=3)
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=6, color="k", lw=0, zorder=4)
        summ[lab.replace("\n", " ")] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), per_section=dict(zip(DESIUM, v.tolist())))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(meths))); ax.set_xticklabels([m[0] for m in meths], fontsize=5.6, linespacing=1.0)
    ax.set_ylim(min(-0.02, np.floor((min(allv) - 0.01) * 50) / 50), max(0.12, max(allv) + 0.02))
    ax.set_ylabel("held-out median Spearman\n(ions detected in >20% of spots)")
    ax.set_title("Transfer to unseen DESIUM sections (7) and patients (6)")
    letter(ax, "a", x=-0.17, y=1.03)
    # b: histology control on the top-1,000 panel
    ax = fig.add_axes([0.645, 0.60, 0.335, 0.33])
    hl = pd.read_csv(res("ablations", "histology_loso.csv")); ho = pd.read_csv(res("ablations", "histology_only_desium.csv"))
    secs = [s for s in DESIUM if s in set(hl["sample"])]
    tx = hl[hl.model == "MetaboCast"].set_index("sample").median_spearman.reindex(secs).to_numpy()
    both = hl[hl.model == "MetaboCast+H&E"].set_index("sample").median_spearman.reindex(secs).to_numpy()
    he = ho[(ho.model == "H&E-only") & (ho.test == "LOSO (cross-section)")].set_index("section").median_spearman.reindex(secs).to_numpy()
    shuf = ho[ho.model == "H&E-only(shuffled)"].set_index("section").median_spearman.reindex(secs).to_numpy()
    M = np.vstack([tx, he, both])
    for j in range(M.shape[1]):
        ax.plot(range(3), M[:, j], color=GREY, lw=0.7, marker="o", ms=2.5, zorder=2)
    ax.plot(range(3), M.mean(1), color=BLUE, lw=1.8, marker="o", ms=4, zorder=3)
    for i, v in enumerate(M.mean(1)):
        ax.text(i, v + 0.012, f"{v:.3f}", fontsize=6, ha="center", va="bottom", color=BLUE, fontweight="bold")
    ax.axhline(np.mean(shuf), color=RED, lw=0.7, ls="--")
    ax.text(2.25, np.mean(shuf) + 0.006, "H&E shuffled", fontsize=5.4, color=RED, ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels(["transcriptome\nonly", "H&E only\n(13 patch\nfeatures)", "transcriptome\n+ H&E"], fontsize=5.8)
    ax.set_xlim(-0.3, 2.4); ax.set_ylabel("held-out median Spearman (top-1,000 ions)")
    p_both, p_he = wilcoxon(both, tx).pvalue, wilcoxon(he, tx).pvalue        # exact two-sided, six sections
    ax.set_title(f"Adding H&E: transfer higher in {int((both > tx).sum())}/{len(secs)} (P = {p_both:.3f})\n"
                 f"H&E only: higher in {int((he > tx).sum())}/{len(secs)} (P = {p_he:.2f})", fontsize=6.6)
    letter(ax, "b", x=-0.21, y=1.03)
    summ["histology"] = dict(sections=secs, transcriptome=tx.tolist(), he_only=he.tolist(), combined=both.tolist(),
                             shuffled=shuf.tolist(), means=M.mean(1).tolist(), p_combined=float(p_both), p_he_only=float(p_he))
    # c: within-section versus transfer accuracy per ion: the two reliability tiers
    a = pd.read_csv(res("desium_per_ion_accuracy.csv"))
    w, t = a.within_section_r.to_numpy(float), a.leave_one_section_out_r.to_numpy(float)
    ok = np.isfinite(w) & np.isfinite(t)
    tiers = np.array([tier(x, y) for x, y in zip(w, t)])
    ax = fig.add_axes([0.075, 0.08, 0.36, 0.40])
    for k in ["low", "within-section only", "transfer-supported"]:
        m = ok & (tiers == k)
        ax.scatter(w[m], t[m], s=2.5, color=TIER_COL[k], lw=0, rasterized=True,
                   label=f"{k} ({int((tiers == k).sum()):,})")
    ax.axvline(WITHIN_MIN, color="k", lw=0.6, ls="--"); ax.axhline(TRANSFER_MIN, color="k", lw=0.6, ls="--")
    # labels of transfer-supported ions sit in the empty upper-left quadrant with leader lines; the others beside the point
    lab = {"glutathione": (306.0756, (4, 2), "left"), "ascorbate": (175.0228, (4, 2), "left"),
           "palmitate": (255.2311, (4, -2), "left"), "stearate": (283.2625, (5, 4), "left"),
           "linoleate": (279.2329, (0, -11), "center"), "urate": (167.0207, (0.04, 0.385), "left"),
           "hexose": (179.055, (0.04, 0.345), "left"), "taurine": (124.0072, (0.04, 0.305), "left"),
           "DHA": (327.2314, (0.04, 0.265), "left"), "arachidonate": (303.2323, (0.04, 0.225), "left")}
    A = ann_table(); amz = A.mz.to_numpy(float)
    for name, (mz0, off, ha) in lab.items():
        j = int(np.argmin(np.abs(a.mz.to_numpy(float) - mz0)))
        k = int(np.argmin(np.abs(amz - a.mz.iloc[j])))
        name = name + ("†" if abs(amz[k] - a.mz.iloc[j]) < 1e-3 and bool(A.ambiguous.iloc[k]) else "")  # >1 candidate
        if ok[j]:
            ax.scatter(w[j], t[j], s=14, facecolor="none", edgecolor="k", lw=0.6, zorder=4)
            if isinstance(off[0], float):
                ax.annotate(name, (w[j], t[j]), xytext=off, textcoords="data", fontsize=5.3, va="center",
                            arrowprops=dict(arrowstyle="-", lw=0.35, color="0.35", shrinkA=1, shrinkB=2.5))
            else:
                ax.annotate(name, (w[j], t[j]), xytext=off, textcoords="offset points", fontsize=5.3, ha=ha, zorder=5,
                            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.75))
    ax.set_xlabel("within-section r (spatial-block CV, mean of 7 sections)")
    ax.set_ylabel("held-out-section r (mean of 7)")
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", markerscale=2.5, handletextpad=0.2)
    ax.set_title("Per-ion accuracy defines the reliability tiers (2,086 channels)", fontsize=6.6)
    letter(ax, "c", x=-0.19, y=1.03)
    summ["tiers"] = {k: int((tiers == k).sum()) for k in list(TIER_COL) + ["uncalibrated"]}
    summ["tiers_both"] = int(((t >= TRANSFER_MIN) & (w >= WITHIN_MIN)).sum())
    # d: HCC tumour - normal for annotated ions, by tier
    d = pd.read_csv(res("crosscancer", "HCC_NtoT_annotated_by_patient.csv"))
    d["label"] = d.annotation.apply(nm_label) + "  " + d.mz.map(lambda m: f"{m:.2f}")
    g = d.groupby("label").agg(mean=("delta_NtoT", "mean"), tier=("tier", "first"),
                               up=("delta_NtoT", lambda s: int((s > 0).sum()))).sort_values("mean")
    sel = pd.concat([g.head(8), g.tail(8)])              # the eight largest predicted decreases and increases
    ax = fig.add_axes([0.72, 0.08, 0.26, 0.40]); y = np.arange(len(sel))
    for k, (labx, row) in enumerate(sel.iterrows()):
        filled = row.tier == "transfer-supported"; col = RED if row["mean"] > 0 else BLUE
        ax.barh(k, row["mean"], color=col if filled else "white", edgecolor=col, lw=0.8, height=0.65)
        pts = d[d.label == labx].delta_NtoT.values
        ax.scatter(pts, np.full(len(pts), k), s=5, color="black", zorder=3, lw=0)
    ax.set_yticks(y); ax.set_yticklabels([pretty(s) for s in sel.index], fontsize=5.4); ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("predicted Δ log intensity (tumour − normal)")
    ax.set_title("HCC-1–4, transcriptome only (no MSI):\nfilled, DESIUM transfer r ≥ 0.20; open, within-section only",
                 fontsize=6.2)
    ax.legend(handles=[Patch(facecolor="#888888", edgecolor="#888888", label="transfer-supported"),
                       Patch(facecolor="white", edgecolor="#888888", label="within-section only")],
              frameon=False, fontsize=5.2, loc="lower right")
    letter(ax, "d", x=-1.05, y=1.03)
    save(fig, "Figure5"); g.to_csv(f"{OUT}/Figure5d_source.csv")
    summ["hcc"] = dict(n_ions=int(d.mz.nunique()), by_tier=d.drop_duplicates("mz").tier.value_counts().to_dict())
    return summ


def nm_label(a):
    """Name of the nearest candidate; † when the channel has more than one candidate (all are listed in the tables)."""
    parts = [p.strip() for p in a.split(";")]
    return pretty(parts[0].split(" [")[0]) + ("†" if len(parts) > 1 else "")


def fig6():
    desium = loso_medians("loso_desium"); S = sma_loso(); A = loso_medians("sma/loso_fmp_mouse_by_animal")
    wd = within_all_detected("within_desium", DESIUM)
    ws = within_all_detected("within_sma", [s for g in SMA_GROUPS.values() for s in g])
    fm = SMA_GROUPS["fmp_mouse"]
    rows = [("DESI-MSI · breast/lung tumour · human\n7 sections, 6 patients", [desium[s] for s in DESIUM], [wd[s] for s in DESIUM], "#555555"),
            ("MALDI-DHB lipids · striatum · mouse\n2 sections, 1 animal", [S["dhb"][s] for s in SMA_GROUPS["dhb"]],
             [ws[s] for s in SMA_GROUPS["dhb"] if s in ws], ORANGE),
            ("MALDI-FMP · striatum · mouse\n3 sections, 3 animals", [S["fmp_mouse"][s] for s in fm if s not in NIGRA],
             [ws[s] for s in fm if s not in NIGRA and s in ws], "#56B4E9"),
            ("MALDI-FMP · substantia nigra · mouse\n3 sections, 3 animals", [S["fmp_mouse"][s] for s in fm if s in NIGRA],
             [ws[s] for s in fm if s in NIGRA and s in ws], "#0072B2"),
            ("MALDI-FMP · striatum · human\n4 sections, 1 donor", [S["fmp_human"][s] for s in SMA_GROUPS["fmp_human"]],
             [ws[s] for s in SMA_GROUPS["fmp_human"] if s in ws], "#009E73")]
    fig = plt.figure(figsize=(7.2, 3.1)); summ = {}
    ax = fig.add_axes([0.30, 0.15, 0.38, 0.72]); rng = np.random.default_rng(1)
    for i, (lab, a, wref, col) in enumerate(rows[::-1]):
        ax.scatter(a, i + rng.uniform(-0.12, 0.12, len(a)), s=13, color=col, zorder=3, lw=0)
        m = float(np.mean(a)); ax.plot([m, m], [i - 0.25, i + 0.25], color=col, lw=2)
        if wref:
            ax.scatter([np.mean(wref)], [i], marker="D", s=22, facecolor="white", edgecolor=col, zorder=4, lw=1)
        ax.text(max(max(a), np.mean(wref) if wref else 0) + 0.015, i, f"{m:.3f}", va="center", fontsize=6)
        summ[lab.split("\n")[0]] = dict(values=list(map(float, a)), mean=m, within_all_detected=float(np.mean(wref)) if wref else None)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=5.8)
    ax.axvline(0, color="grey", lw=0.6); ax.set_xlim(-0.02, 0.48); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("median Spearman, ions detected in >20% of spots")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="k", label="held-out section"),
                       Line2D([], [], color="k", lw=2, label="cohort mean (value at right)"),
                       Line2D([], [], marker="D", ls="", mfc="white", mec="k", label="within-section reference\n(spatial blocks, same ions)")],
              loc="upper right", frameon=False, fontsize=5.3)
    ax.set_title("Transfer within each cohort (descriptive: cohorts differ in chemistry, platform, species and design)",
                 fontsize=6.5, loc="right")
    letter(ax, "a", x=-0.74, y=1.03)
    ax = fig.add_axes([0.78, 0.15, 0.20, 0.72])
    for s in fm:
        c = "#0072B2" if s in NIGRA else "#56B4E9"
        ax.plot([0, 1], [S["fmp_mouse"][s], A[s]], color=c, lw=0.9, marker="o", ms=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["section\nheld out", "animal\nheld out"]); ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0, 0.4); ax.set_ylabel("FMP transfer (median Spearman)")
    ax.legend(handles=[Line2D([], [], color="#0072B2", marker="o", ms=3, label="substantia nigra"),
                       Line2D([], [], color="#56B4E9", marker="o", ms=3, label="striatum")], frameon=False, fontsize=5.4,
              loc="upper right")
    loso_m = np.mean([S["fmp_mouse"][s] for s in fm]); loao_m = np.mean([A[s] for s in fm])
    ax.set_title(f"Mouse FMP: whole animal held out\nmean {loso_m:.3f} → {loao_m:.3f}", fontsize=6.5)
    letter(ax, "b", x=-0.45, y=1.03)
    save(fig, "Figure6")
    summ["fmp_mouse_loso_vs_animal"] = {s: (float(S["fmp_mouse"][s]), float(A[s])) for s in fm}
    summ["fmp_mouse_means"] = (float(loso_m), float(loao_m))
    return summ


def supp_neurotransmitter(ion="FMP557"):
    Z = np.load(res("sma/loso_fmp_mouse_maps.npz")); S = sma_loso(); desium = loso_medians("loso_desium")
    T = pd.read_csv(res("sma", "sma_registration_audit_tic.csv.gz"))      # every in-tissue spot, paired or not
    fig = plt.figure(figsize=(7.2, 3.0)); out = {}
    for k, (sec, lab) in enumerate([("SMA_FMP_V11T16-085_A1", "Substantia nigra\n(mPD1)"),
                                    ("SMA_FMP_V11L12-109_A1", "Striatum\n(mPD1)")]):
        co = Z[f"{sec}__{ion}__coords"]; meas = Z[f"{sec}__{ion}__measured"]; pred = Z[f"{sec}__{ion}__predicted"]
        r = float(np.ravel(Z[f"{sec}__{ion}__r"])[0]); out[sec] = r
        allxy = T.loc[T.section == sec, ["x", "y"]].to_numpy(float)
        dist, nn = cKDTree(co).query(allxy); hit = dist < 1                 # analysed spots; the rest are unpaired
        for j, (v, t) in enumerate([(meas, "measured"), (pred, f"predicted, section held out (r = {fmt_r(r, sign=True)})")]):
            ax = fig.add_axes([0.08 + j * 0.25, 0.52 - k * 0.48, 0.23, 0.42]); lo, hi = np.percentile(v, [1, 99])
            full = np.full(len(allxy), np.nan); full[hit] = np.asarray(v, float)[nn[hit]]
            spotmap_masked(ax, allxy[:, 0], -allxy[:, 1], full, lo, hi)
            ax.set_title(t, fontsize=6)
        fig.text(0.02, 0.73 - k * 0.48, lab, rotation=90, va="center", ha="center", fontsize=6.5, fontweight="bold",
                 color="#0072B2" if k == 0 else "#56B4E9")
    map_key(fig, [0.21, -0.02, 0.22, 0.016], orientation="horizontal")
    fm = SMA_GROUPS["fmp_mouse"]
    bars = [("DESI tumour\n(mixed, human)", np.mean([desium[s] for s in DESIUM]), "#7F7F7F"),
            ("FMP striatum\n(mouse)", np.mean([S["fmp_mouse"][s] for s in fm if s not in NIGRA]), "#56B4E9"),
            ("FMP nigra\n(mouse)", np.mean([S["fmp_mouse"][s] for s in fm if s in NIGRA]), "#0072B2"),
            ("FMP striatum\n(human)", np.mean(list(S["fmp_human"].values())), "#009E73"),
            ("DHB lipids\n(mouse)", np.mean(list(S["dhb"].values())), ORANGE)]
    ax = fig.add_axes([0.68, 0.12, 0.3, 0.8])
    for i, (lab, a, c) in enumerate(bars):
        ax.barh(i, a, color=c, height=0.6)
        ax.text(a + 0.005, i, f"{a:.3f}", va="center", fontsize=6.3)
    ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars], fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("held-out-section median Spearman"); ax.set_title(f"Transfer by cohort (m/z {ion[3:]}.2 maps at left)", fontsize=6.6)
    save(fig, "SuppFig_neurotransmitter")
    out["bars"] = {b[0].replace("\n", " "): float(b[1]) for b in bars}
    return out


def s1():
    h = J(res("harmonization.json"))
    main = [("per-fold PCA", "per-fold PCA (default)", BLUE), ("pooled PCA", "pooled PCA", GREY),
            ("pooled + Harmony", "pooled PCA + Harmony", LIGHTBLUE)]
    fig = plt.figure(figsize=(7.2, 2.5))
    ax = fig.add_axes([0.07, 0.2, 0.58, 0.68]); x = np.arange(7); w = 0.26
    for k, (key, lab, c) in enumerate(main):
        ax.bar(x + (k - 1) * w, [h[key]["per"][s] for s in DESIUM], w, color=c, label=lab)
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in DESIUM], fontsize=6)
    ax.set_ylabel("held-out-section median Spearman"); ax.legend(frameon=False, fontsize=5.8, ncol=3, loc="upper right")
    letter(ax, "a", x=-0.1, y=1.02)
    ax = fig.add_axes([0.74, 0.2, 0.25, 0.68])
    order = ["per-fold PCA", "pooled PCA", "pooled + Harmony", "centred PCA", "z-scored features"]
    labs = ["per-fold\nPCA", "pooled\nPCA", "pooled +\nHarmony", "centred\nPCA", "z-scored\nfeatures"]
    mu = [h[k]["overall"][0] for k in order]; sd = [h[k]["overall"][1] for k in order]
    ax.bar(range(5), mu, yerr=sd, color=[BLUE, GREY, LIGHTBLUE, GOLD, "#8C7BB3"], error_kw=dict(lw=0.6, capsize=1.5))
    for i, v in enumerate(mu):
        ax.text(i, v + sd[i] + 0.004, f"{v:.3f}", ha="center", fontsize=5.6)
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=5.4); ax.set_ylabel("mean ± s.d. over 7 sections")
    letter(ax, "b", x=-0.35, y=1.02)
    save(fig, "SuppFig_harmony")
    return {k: h[k]["overall"] for k in order}


def run():
    return {"Figure5": fig5(), "Figure6": fig6(), "SuppFig_neurotransmitter": supp_neurotransmitter(), "SuppFig_harmony": s1()}
