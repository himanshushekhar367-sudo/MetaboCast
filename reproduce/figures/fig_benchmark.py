"""Figures 2, 3 and 4 and the within-section supplementary figures (example maps, best- and worst-predicted
annotated ions, maps on a shared scale): reproduce/within_benchmark.py, reproduce/within_controls.py and
reproduce/coverage_outer.py."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from common import (BLUE, GOLD, GREY, LIGHTBLUE, RED, TEAL, DESIUM, LIPID, OUT, SHORT, J, ann_table, fmt_r, is_polar,
                    ion_label, ion_title, letter, map_key, per_ion_mean, res, save, spotmap, within, wilcoxon_p)
from manifest import figure_stem                      # noqa: E402  (reproduce/ is on the path via common)

js, per = within("within_desium")
SECS = [s for s in DESIUM if s in js]
MZ = per[("mz", SECS[0])]
KEGG = "KEGG pathway features + supervised regressor"
COORD = "coordinates only (tuned kNN)"
METH = [("MetaboCast", "MetaboCast", BLUE), ("MetaboCast + KEGG", "MetaboCast + KEGG", LIGHTBLUE),
        ("MetaboCast (ridge only)", "ridge only", "#5C8BC2"), ("no diffusion", "no diffusion", GREY),
        (KEGG, "KEGG features + regressor", GOLD), ("joint-NMF", "joint NMF", "#C0603A"),
        (COORD, "coordinates only (kNN)", "#7A8F3A"), ("single-gene OLS", "single-gene OLS", "#BBBBBB")]
MAPSEC = "LC_091"     # map panels: the lung section in which EPA, xanthine and fumarate are all recovered
MAPS = np.load(res("within_desium_maps.npz"))
CONF = np.load(res("within_desium_conformal.npz"))
CTRL = J(res("within_controls.json"))
COV = J(res("coverage", "coverage_outer.json"))
ANN = ann_table()
MAP_IONS = {"EPA": 301.2173, "Xanthine": 151.0261, "Fumarate": 115.0037, "Tryptophan": 203.0826, "GSH": 306.0765,
            "DHA": 327.2329}


def sec_scores(method):
    if method == COORD:
        return np.array([CTRL[s]["pooled"]["coordinates only"]["full_panel"] for s in SECS])
    return np.array([js[s]["scores"].get(method, np.nan) for s in SECS])


def fig2():
    fig = plt.figure(figsize=(7.2, 2.9))
    ax = fig.add_axes([0.065, 0.30, 0.38, 0.60]); rng = np.random.default_rng(0)
    allv = []
    for i, (m, lab, col) in enumerate(METH):
        v = sec_scores(m); mu, sd = np.nanmean(v), np.nanstd(v, ddof=1); allv += list(v)
        ax.bar(i, mu, 0.68, color=col, lw=0, zorder=2)
        ax.errorbar(i, mu, yerr=sd, color="k", lw=0.6, capsize=1.8, zorder=3)
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=6, color="k", lw=0, zorder=4)
    ax.set_xticks(range(len(METH)))
    ax.set_xticklabels([m[1] for m in METH], fontsize=5.8, rotation=35, ha="right", rotation_mode="anchor")
    lo = np.floor((min(allv) - 0.02) * 20) / 20                      # keep every section visible
    ax.set_ylabel("within-section median Spearman\n(top-1,000 ions)"); ax.set_ylim(lo, 0.5); ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Seven DESIUM sections, spatial-block CV")
    letter(ax, "a", x=-0.17, y=1.02)
    ax = fig.add_axes([0.52, 0.30, 0.20, 0.60])
    cols = [("joint-NMF", "joint\nNMF"), (COORD, "coord.\nonly"), ("no diffusion", "no\ndiffusion"),
            ("MetaboCast", "Metabo-\nCast")]
    M = np.vstack([sec_scores(c) for c, _ in cols])
    for j in range(M.shape[1]):
        ax.plot(range(4), M[:, j], color=GREY, lw=0.7, marker="o", ms=2.5, zorder=2)
    ax.plot(range(4), M.mean(1), color=BLUE, lw=1.8, marker="o", ms=4, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in cols], fontsize=5.8); ax.set_xlim(-0.3, 3.3)
    p = {c: wilcoxon_p(M[3], M[k]) for k, (c, _) in enumerate(cols[:3])}
    wins = {c: int((M[3] > M[k]).sum()) for k, (c, _) in enumerate(cols[:3])}
    ax.set_title(f"MetaboCast vs joint NMF {wins['joint-NMF']}/7, P={p['joint-NMF']:.3f}\n"
                 f"vs coordinates {wins[COORD]}/7, P={p[COORD]:.3f}\n"
                 f"vs no diffusion {wins['no diffusion']}/7, P={p['no diffusion']:.3f}", fontsize=5.6)
    ax.set_ylabel("median Spearman")
    letter(ax, "b", x=-0.38, y=1.02)
    s = "BC_515_Section_1"; r = per[("MetaboCast", s)]; r = r[np.isfinite(r)]
    ax = fig.add_axes([0.80, 0.30, 0.19, 0.60])
    ax.hist(r, bins=np.linspace(-0.3, 1.0, 40), color=BLUE, alpha=0.9)
    ax.axvline(np.median(r), color=RED, lw=0.8)
    ax.set_xlabel("per-ion Spearman r"); ax.set_ylabel("ions")
    ax.set_title(f"BC 515-1, all {len(r):,} detected ions\nmedian {np.median(r):.3f}; {100 * np.mean(r > 0.3):.0f}% > 0.3", fontsize=6)
    letter(ax, "c", x=-0.38, y=1.02)
    save(fig, "Figure2")
    summ = CTRL["_summary"]
    return dict(median=float(np.median(r)), mean=float(np.mean(r)), gt03=float(np.mean(r > 0.3)),
                gt05=float(np.mean(r > 0.5)), n=len(r), p=p, wins=wins, ylim_low=float(lo),
                method_means={m: float(np.nanmean(sec_scores(m))) for m, _, _ in METH},
                method_sd={m: float(np.nanstd(sec_scores(m), ddof=1)) for m, _, _ in METH},
                coords_all_detected=summ["pooled_mean_over_sections"]["coordinates only"]["all_detected"],
                metabocast_all_detected=summ["pooled_mean_over_sections"]["MetaboCast"]["all_detected"],
                per_block=summ["per_block_mean_over_folds"])


def mpair(fig, rect, s, nm, show_titles=True, shared=False, key=False):
    """Measured and held-out predicted map of one ion; each map on its own 1st-99th percentile scale, or both on
    the measured map's scale (shared=True, log1p intensity)."""
    co = CONF[f"{s}__coords"]; meas = MAPS[f"{s}__{nm}__measured"]; pred = MAPS[f"{s}__{nm}__predicted"]
    r = float(MAPS[f"{s}__{nm}__r"]); x0, y0, w, h = rect
    lo_m, hi_m = np.percentile(meas, [1, 99])
    for k, (v, lab) in enumerate([(meas, "measured"), (pred, f"predicted (r = {fmt_r(r)})")]):
        ax = fig.add_axes([x0 + k * w / 2, y0, w / 2 * 0.96, h])
        lo, hi = (lo_m, hi_m) if shared else np.percentile(v, [1, 99])
        pc = spotmap(ax, co[:, 0], -co[:, 1], v, lo, hi)
        if show_titles:
            ax.set_title(lab, fontsize=6.3)
    if key:
        cb = fig.colorbar(pc, cax=fig.add_axes([x0 + w * 0.99, y0 + h * 0.15, 0.008, h * 0.7]))
        cb.ax.tick_params(labelsize=5.2, length=1.5); cb.outline.set_linewidth(0.4)
        cb.set_label("log1p intensity", fontsize=5.4, labelpad=1)
    return r


def class_table():
    """Per-ion mean within-section Spearman (over the sections in which the ion was scored) for every channel with a
    single-class putative candidate."""
    rmean, rsd, nsec = per_ion_mean(per, "MetaboCast", SECS, MZ)
    idx = {round(m, 4): j for j, m in enumerate(MZ)}
    a = ANN[ANN.cls.notna()].copy()
    j = [idx.get(round(m, 4)) for m in a.mz]
    a["r"] = [rmean[i] if i is not None else np.nan for i in j]
    a["sd"] = [rsd[i] if i is not None else np.nan for i in j]
    a["n"] = [nsec[i] if i is not None else 0 for i in j]
    return a.dropna(subset=["r"])


def class_medians():
    return class_table().groupby("cls").r.agg(["median", "count"]).query("count >= 2").sort_values("median")


def fig3():
    fig = plt.figure(figsize=(7.2, 5.0)); rr = {}
    for i, nm in enumerate(["EPA", "Xanthine", "Fumarate"]):
        rr[nm] = mpair(fig, [0.01 + i * 0.315, 0.71, 0.30, 0.19], MAPSEC, nm)
        fig.text(0.01 + i * 0.315 + 0.15, 0.94, ion_title(MAP_IONS[nm], ANN), ha="center", va="bottom", fontsize=6.5,
                 fontweight="bold")
        fig.text(0.01 + i * 0.315, 0.985, "abc"[i], fontsize=9, fontweight="bold", va="top")
    map_key(fig, [0.955, 0.735, 0.009, 0.14])
    # d: putative exact-mass candidates by class (negative mode)
    a = ANN[ANN.neg]
    vc = a.cls.fillna("candidates in >1 class").value_counts()
    single = vc[vc == 1]                                   # classes with one channel are pooled into one bar
    vc = vc[vc > 1].sort_values()
    if len(single):
        vc = pd.concat([pd.Series({f"{len(single)} other classes (1 each)": int(single.sum())}), vc])
    ax = fig.add_axes([0.14, 0.08, 0.19, 0.54])
    ax.barh(vc.index, vc.values, color=[GREY if c.startswith("candidates") else ("#CFCFCF" if "other classes" in c
                                        else (BLUE if c in LIPID else GOLD)) for c in vc.index], height=0.7)
    for y, v in enumerate(vc.values):
        ax.text(v + 0.3, y, str(v), va="center", fontsize=6)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.set_xlabel("m/z channels"); ax.set_title(f"Putative exact-mass candidates\n{len(a)} of {len(ANN):,} channels (negative mode)", fontsize=6.6)
    letter(ax, "d", x=-0.68, y=1.02)
    # e: within-section accuracy by putative class
    cp = class_medians()
    ax = fig.add_axes([0.53, 0.08, 0.44, 0.54])
    ax.barh(cp.index, cp["median"], color=[BLUE if c in LIPID else GOLD for c in cp.index], height=0.7)
    for y, (m, n) in enumerate(zip(cp["median"], cp["count"])):
        ax.text(max(m, 0) + 0.008, y, f"{fmt_r(m)} (n={n})", va="center", fontsize=6.2)
    ax.set_xlim(min(0, cp["median"].min() - 0.02), 0.85); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("median per-ion Spearman r\n(each ion: mean over the DESIUM sections where it was scored)")
    ax.set_title("Within-section predictability by putative class\n(single-class channels; classes with ≥2 channels)", fontsize=6.6)
    ax.legend(handles=[Patch(color=BLUE, label="lipid"), Patch(color=GOLD, label="polar")], loc="lower right", frameon=False)
    letter(ax, "e", x=-0.32, y=1.02)
    save(fig, "Figure3")
    cp.to_csv(f"{OUT}/Figure3e_source.csv")
    return dict(section=MAPSEC, r=rr, classes=cp.reset_index().to_dict("records"), n_single_class=int(ANN.cls.notna().sum()),
                annotation=dict(n_candidates=int(len(a)), n_multi=int((a.n_candidates > 1).sum()),
                                n_single_class=int(a.cls.notna().sum()), classes=vc.to_dict()))


def fig4():
    neg_cls = {round(m, 4): c for m, c in zip(ANN.mz, ANN.cls) if isinstance(c, str)}
    cov, rel, rr, cls = [], [], [], []
    for s in SECS:
        d = CONF[f"{s}__dets"]
        cov.append(CONF[f"{s}__coverage"]); rel.append(CONF[f"{s}__relwidth"]); rr.append(per[("MetaboCast", s)][d])
        cls += [neg_cls.get(round(m, 4), "") for m in MZ[d]]
    cov, rel, rr, cls = np.concatenate(cov), np.concatenate(rel), np.concatenate(rr), np.array(cls)
    ok = np.isfinite(rr) & np.isfinite(rel)
    rho = spearmanr(rel[ok], rr[ok])[0]
    fig = plt.figure(figsize=(7.2, 7.0))
    ax = fig.add_axes([0.07, 0.75, 0.24, 0.20])
    ax.hist(cov, bins=np.linspace(0.4, 1.0, 31), color=BLUE)
    ax.axvline(0.9, color="k", lw=0.8, ls="--"); ax.axvline(np.nanmean(cov), color=RED, lw=1)
    ax.set_xlabel("coverage of the 90% interval (per ion and section)"); ax.set_ylabel("ion–section pairs")
    ax.set_title(f"within section: mean coverage {np.nanmean(cov):.2f} (red)\nnominal 0.90 (dashed)"); letter(ax, "a", x=-0.3, y=1.03)
    ax = fig.add_axes([0.40, 0.75, 0.25, 0.20])
    ax.hexbin(rel[ok], rr[ok], gridsize=40, cmap="Blues", mincnt=1, bins="log", extent=(0, np.nanpercentile(rel, 99), -0.2, 1))
    ax.set_xlabel("relative interval width (width / s.d.)"); ax.set_ylabel("per-ion Spearman r")
    ax.set_title(f"wider intervals, lower accuracy\n(Spearman ρ = {fmt_r(rho)})"); letter(ax, "b", x=-0.28, y=1.03)
    ax = fig.add_axes([0.75, 0.75, 0.2, 0.20])
    lip, pol = np.isin(cls, list(LIPID)), np.array([is_polar(c) for c in cls])
    vals = [np.nanmean(rel[lip]), np.nanmean(rel[pol])]
    ax.bar([0, 1], vals, color=[BLUE, GOLD], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"lipid\n(n={lip.sum()})", f"polar\n(n={pol.sum()})"]); ax.set_ylabel("mean relative width")
    ax.set_xlabel("ion–section pairs (putative class)", fontsize=6)
    ax.set_title("lipids get narrower intervals"); letter(ax, "c", x=-0.4, y=1.03)
    # d: example maps (fold models) with their interval widths (calibration-split models)
    ex = {}
    for k, (nm, lab) in enumerate([("Xanthine", "narrow interval"), ("Tryptophan", "wide interval")]):
        d = CONF[f"{MAPSEC}__dets"]; j = int(np.argmin(np.abs(MZ[d] - float(MAPS[f"{MAPSEC}__{nm}__mz"]))))
        relw = float(CONF[f"{MAPSEC}__relwidth"][j])
        r = mpair(fig, [0.05 + k * 0.48, 0.43, 0.44, 0.19], MAPSEC, nm)
        fig.text(0.05 + k * 0.48 + 0.22, 0.645, f"{ion_title(MAP_IONS[nm], ANN).replace(chr(10), ', ')}: {lab}, relative width {relw:.2f}",
                 ha="center", fontsize=6.2)
        ex[nm] = (r, relw)
    map_key(fig, [0.40, 0.405, 0.2, 0.008], orientation="horizontal")
    fig.text(0.0, 0.66, "d", fontsize=9, fontweight="bold")
    # e: released widths on fully held-out patients
    ax = fig.add_axes([0.07, 0.065, 0.88, 0.25])
    ps = COV["per_section"]; x = np.arange(len(SECS))
    cf = [ps[s]["coverage_crossfit"] for s in SECS]; sp = [ps[s]["coverage_split_mean"] for s in SECS]
    rng_lo = [min(ps[s]["coverage_split_by_calibration_patient"].values()) for s in SECS]
    rng_hi = [max(ps[s]["coverage_split_by_calibration_patient"].values()) for s in SECS]
    ax.vlines(x + 0.12, rng_lo, rng_hi, color=GOLD, lw=0.8)
    ax.scatter(x - 0.12, cf, s=18, color=BLUE, zorder=3, label=f"cross-fitted widths (released procedure), mean {COV['mean_coverage_crossfit_sections']:.2f}")
    ax.scatter(x + 0.12, sp, s=18, marker="D", color=GOLD, zorder=3,
               label=f"split calibration on one patient (range over patients), mean {COV['mean_coverage_split_sections']:.2f}")
    ax.axhline(0.9, color="k", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in SECS], fontsize=6); ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("coverage on the held-out patient")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    ax.set_title("90% widths on patients excluded from fitting, tuning and calibration")
    letter(ax, "e", x=-0.06, y=1.03)
    save(fig, "Figure4")
    return dict(coverage=float(np.nanmean(cov)), rho=float(rho), relw_lipid=vals[0], relw_polar=vals[1],
                n_lipid=int(lip.sum()), n_polar=int(pol.sum()), examples=ex, section=MAPSEC, n_pairs=int(ok.sum()),
                outer=dict(crossfit=dict(zip(SECS, cf)), split=dict(zip(SECS, sp)),
                           mean_crossfit=COV["mean_coverage_crossfit_sections"], mean_split=COV["mean_coverage_split_sections"],
                           mean_crossfit_patients=COV["mean_coverage_crossfit_patients"],
                           mean_split_patients=COV["mean_coverage_split_patients"]))


def supp_maps():
    fig = plt.figure(figsize=(6.0, 6.6)); rr = {}
    for i, (nm, note) in enumerate([("EPA", "lipid, well predicted"), ("Tryptophan", "polar, not recovered"),
                                    ("Fumarate", "polar, recovered")]):
        rr[nm] = mpair(fig, [0.06, 0.68 - i * 0.32, 0.84, 0.26], MAPSEC, nm)
        fig.text(0.02, 0.81 - i * 0.32, f"{ion_title(MAP_IONS[nm], ANN).replace(chr(10), ', ')}\n{note}", rotation=90,
                 va="center", ha="center", fontsize=6.0)
        fig.text(0.02, 0.97 - i * 0.32, "abc"[i], fontsize=9, fontweight="bold")
    map_key(fig, [0.925, 0.42, 0.012, 0.2])
    save(fig, "SuppFig_maps")
    return rr


def supp_shared_scale():
    """Measured and predicted maps on one log1p scale per ion: shows how much predicted contrast is compressed."""
    names = ["EPA", "DHA", "GSH", "Xanthine", "Fumarate", "Tryptophan"]
    fig = plt.figure(figsize=(7.2, 6.6)); out = {}
    for i, nm in enumerate(names):
        col, row = i % 2, i // 2
        x0, y0 = 0.03 + col * 0.49, 0.70 - row * 0.32
        r = mpair(fig, [x0, y0, 0.40, 0.21], MAPSEC, nm, shared=True, key=True)
        fig.text(x0 + 0.20, y0 + 0.245, ion_title(MAP_IONS[nm], ANN).replace("\n", ", "), ha="center", fontsize=6.3, fontweight="bold")
        meas = MAPS[f"{MAPSEC}__{nm}__measured"]; pred = MAPS[f"{MAPSEC}__{nm}__predicted"]
        out[nm] = dict(r=r, sd_ratio=float(np.std(pred) / max(np.std(meas), 1e-9)))
    save(fig, "SuppFig_maps_shared_scale")
    return out


def supp_best_worst():
    t = class_table()
    t = t[t.n >= 2]
    t = pd.concat([t.sort_values("r", ascending=False).head(12), t.sort_values("r").head(8).sort_values("r", ascending=False)])
    t["lab"] = [ion_label(row) for row in t.itertuples()]
    fig, ax = plt.subplots(figsize=(4.4, 4.0)); y = np.arange(len(t))[::-1]
    ax.barh(y, t.r, xerr=t.sd, color=[BLUE if c in LIPID else GOLD for c in t.cls], height=0.7,
            error_kw=dict(lw=0.5, capsize=1.2))
    ax.set_yticks(y); ax.set_yticklabels(t.lab, fontsize=5.6); ax.axhline(7.5, color="grey", ls=":", lw=0.7); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("within-section mean Spearman r (± s.d. across DESIUM sections)")
    ax.set_title("Putative identities: m/z, nearest candidate, adduct", fontsize=6.5)
    ax.legend(handles=[Patch(color=BLUE, label="lipid"), Patch(color=GOLD, label="polar")], frameon=False, loc="lower right")
    save(fig, "SuppFig_best_worst")
    t[["mz", "annotation", "adduct", "candidates", "cls", "r", "sd", "n"]].to_csv(f"{OUT}/{figure_stem('SuppFig_best_worst')}_source.csv", index=False)
    return t[["mz", "annotation", "adduct", "r"]].to_dict("records")


def run():
    # keys are manifest keys (reproduce/manifest.py), so figure_summary.json names items independently of numbering
    return {"Figure2": fig2(), "Figure3": fig3(), "Figure4": fig4(), "SuppFig_maps": supp_maps(),
            "SuppFig_maps_shared_scale": supp_shared_scale(), "SuppFig_best_worst": supp_best_worst()}
