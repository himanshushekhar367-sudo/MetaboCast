"""Robustness figures SuppFig_predictability, SuppFig_spatial_controls and SuppFig_robustness (numbers: reproduce/manifest.py):
analyses from reproduce/rigor_*.py and the representation screen (reproduce/model_selection)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import BLUE, GOLD, GREY, LIGHTBLUE, RED, TEAL, J, letter, res, save


def R(name):
    return J(res("rigor", f"{name}.json"))


FAMILY = [("MetaboCast", "MetaboCast (ensemble)"), ("MetaboCast (ridge only)", "MetaboCast (ridge)"), ("Ridge (alpha 200)", "ridge, α = 200"),
          ("ElasticNet", "elastic net"), ("RandomForest", "random forest"), ("GradBoost(HGB)", "gradient boosting"), ("MLP", "MLP")]


def ed4():
    rc, rs, rb, ru = R("rigor_ceiling"), R("rigor_splits"), R("rigor_baselines"), R("rigor_uncertainty")
    fig = plt.figure(figsize=(7.2, 4.6)); out = {}
    ax = fig.add_axes([0.155, 0.58, 0.17, 0.34])
    fam = rc["family"]; order = [k for k, _ in FAMILY if k in fam]; lab = dict(FAMILY)
    y = np.arange(len(order))[::-1]
    ax.barh(y, [fam[k][0] for k in order], xerr=[fam[k][1] for k in order], height=0.7,
            color=[BLUE, LIGHTBLUE] + [GREY] * (len(order) - 2), error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in order], fontsize=5.8); ax.set_xlabel("median r")
    ax.set_title(f"Learner family\n({rc['n_metab']} ions, 2 sections)"); letter(ax, "a", x=-0.85, y=1.04)
    out["family"] = {k: fam[k] for k in order}
    ax = fig.add_axes([0.40, 0.58, 0.25, 0.34]); h = rs["hops"]; hk = sorted(h, key=int)
    ax.plot([int(k) for k in hk], [h[k][0] for k in hk], color=BLUE, marker="o", ms=4, lw=1.5)
    ax.axvline(2, color=RED, ls="--", lw=0.8); ax.set_xlabel("diffusion hops"); ax.set_ylabel("median r"); ax.set_xticks(range(5))
    ax.set_title("Diffusion depth: two hops suffice"); letter(ax, "b", x=-0.3, y=1.04); out["hops"] = {k: h[k][0] for k in hk}
    ax = fig.add_axes([0.73, 0.58, 0.25, 0.34]); sp = rs["splits"]
    vals = [sp["random_spot"][0], sp["spatial_block"][0], rb["leave_region_out"][0], sp["LOSO"][0]]
    errs = [sp["random_spot"][1], sp["spatial_block"][1], rb["leave_region_out"][1], sp["LOSO"][1]]
    ax.bar(range(4), vals, yerr=errs, color=[RED, BLUE, TEAL, GOLD], error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(range(4)); ax.set_xticklabels(["random\nspot", "spatial\nblock", "leave-\nregion-out", "LOSO"], fontsize=6)
    ax.text(0.5, max(vals) * 0.95, f"inflation {vals[0] - vals[1]:+.2f}", color=RED, fontsize=6)
    ax.set_ylabel("median r"); ax.set_title("Spatial-leakage audit"); letter(ax, "c", x=-0.3, y=1.04)
    out["splits"] = dict(zip(["random", "block", "region", "loso"], vals))
    ax = fig.add_axes([0.07, 0.08, 0.25, 0.34]); ab = np.array(rb["abundance"])
    ax.plot(ab[:, 0], ab[:, 1], color="#3B8B5A", marker="o", ms=4, lw=1.5)
    ax.set_xlabel("metabolite abundance (mean log, quintile centre)"); ax.set_ylabel("median r")
    ax.set_title("Accuracy rises with abundance"); letter(ax, "d", x=-0.3, y=1.04); out["abundance"] = ab[:, 1].tolist()
    ax = fig.add_axes([0.40, 0.08, 0.25, 0.34])
    r3 = J(res("model_selection", "r3_loso_desium.json"))["mean"]; r2 = J(res("model_selection", "loso2_desium.json"))["mean"]
    rep = [("PCA-100\n(default)", r3["pca|yR|aG"]), ("PCA-200", r3["pca200|yR|aG"]), ("NMF-100", r2["nmf|yR|aG"]),
           ("PCA +\nenzyme", r3["pca+enz|yR|aG"]), ("section\nz-genes", r3["secz|yR|aG"]), ("section\nrank-genes", r3["secrank|yR|aG"])]
    ax.bar(range(len(rep)), [v for _, v in rep], color=[BLUE] + [GREY] * (len(rep) - 1))
    for i, (_, v) in enumerate(rep):
        ax.text(i, v + 0.0008, f"{v:.3f}", ha="center", fontsize=5.4)
    ax.set_xticks(range(len(rep))); ax.set_xticklabels([r.replace("\n", " ") for r, _ in rep], fontsize=5.6, rotation=35, ha="right",
                                                        rotation_mode="anchor"); ax.set_ylabel("LOSO median r (DESIUM)")
    ax.set_title("Representation (ridge, rank targets)"); letter(ax, "e", x=-0.3, y=1.04); out["representation"] = {k: v for k, v in rep}
    ax = fig.add_axes([0.73, 0.08, 0.25, 0.34])
    cv = [ru["coverage_id"], ru["coverage_ood"]]
    ax.bar([0, 1], cv, 0.55, color=[BLUE, RED])
    for x, v in enumerate(cv):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=6)
    ax.axhline(0.9, color="k", ls="--", lw=0.8); ax.set_ylim(0, 1.1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["in-distribution", f"new section\n({ru['held'].replace('_', ' ')})"])
    ax.set_ylabel("90% interval coverage")
    ax.set_title(f"Calibration on a new section\n(epistemic uncertainty ×{ru['epistemic_ratio']:.1f})"); letter(ax, "f", x=-0.3, y=1.04)
    out["calibration"] = dict(id=ru["coverage_id"], ood=ru["coverage_ood"], ratio=ru["epistemic_ratio"])
    save(fig, "SuppFig_robustness")
    return out


def ed5():
    rp, rsh, rg = R("rigor_predictability"), R("rigor_shift"), R("rigor_gcn")
    fig = plt.figure(figsize=(7.2, 4.6)); out = {}
    ax = fig.add_axes([0.07, 0.58, 0.25, 0.32])
    x, y = np.array(rp["scatter"]["moran"][0]), np.array(rp["scatter"]["moran"][1])
    ax.scatter(x, y, s=1.5, color=BLUE, alpha=0.3, lw=0, rasterized=True)
    ax.set_xlabel("Moran's I (spatial autocorrelation)"); ax.set_ylabel("predictability (r)")
    ax.set_title(f"Predictability tracks spatial structure\nρ = {rp['univariate']['moran']:+.2f}".replace("-", "−")); letter(ax, "a", x=-0.3, y=1.04)
    ax = fig.add_axes([0.42, 0.58, 0.22, 0.32]); pm = rp["permutation"]
    lab = {"moran": "Moran's I", "cv": "coefficient of\nvariation", "abundance": "abundance", "detection": "detection rate",
           "chemical group": "chemical group"}
    ks = rp["ranking"][::-1]                                   # most important on top
    mean = np.array([pm[k]["loso_mean"] for k in ks])
    lo, hi = np.array([pm[k]["loso_min"] for k in ks]), np.array([pm[k]["loso_max"] for k in ks])
    ax.barh(range(len(ks)), mean, color=[BLUE if k == "moran" else GREY for k in ks])
    ax.errorbar(mean, range(len(ks)), xerr=[mean - lo, hi - mean], fmt="none", ecolor="k", lw=0.6, capsize=1.5)
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(0, 2.5)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([lab[k] for k in ks], fontsize=5.8)
    ax.set_xlabel("decrease in held-out R² when permuted\n(mean and range over held-out sections)", fontsize=5.8)
    ax.set_title(f"Ranking (held-out R² = {rp['cv_r2']:.2f},\nleave-one-section-out)", fontsize=6.4); letter(ax, "b", x=-0.62, y=1.04)
    out["predictability"] = dict(r2=rp["cv_r2"], r2_ion_folds=rp["cv_r2_ion_folds"], rho_moran=rp["univariate"]["moran"],
                                 ranking=rp["ranking"], permutation={k: pm[k]["loso_mean"] for k in rp["ranking"]})
    ax = fig.add_axes([0.74, 0.58, 0.24, 0.32]); sl = rsh["shift_loso"]
    xs = [r[3] for r in sl]; ys = [r[1] for r in sl]; cc = [BLUE if r[0].startswith("BC") else GOLD for r in sl]
    ax.scatter(xs, ys, c=cc, s=20); ax.set_xlabel("metabolite distribution shift"); ax.set_ylabel("LOSO median r")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="breast"), Line2D([], [], marker="o", ls="", color=GOLD, label="lung")],
              frameon=False, fontsize=5.5, loc="upper right", handletextpad=0.2)
    rho_t = f"{rsh['spearman_metab_loso']:+.2f}".replace("-", "−")
    ax.set_title(f"Transfer vs metabolite shift\nρ = {rho_t} (n=7, n.s.)"); letter(ax, "c", x=-0.32, y=1.04)
    out["shift"] = dict(rho_metab=rsh["spearman_metab_loso"], rho_embed=rsh["spearman_embed_loso"], loso=ys)
    ax = fig.add_axes([0.07, 0.08, 0.25, 0.32]); ec = np.array(rsh["epistemic_calibration"])
    ax.bar(range(4), ec[:, 1], 0.6, color=BLUE)
    for i, v in enumerate(ec[:, 1]):
        ax.text(i, v - 0.07, f"{v:.2f}", ha="center", fontsize=5.8, color="white")
    ax.axhline(0.9, color="k", ls="--", lw=0.8); ax.set_ylim(0, 1.1)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4\nhigh"]); ax.set_xlabel("epistemic uncertainty quartile")
    ax.set_ylabel("90% interval coverage"); ax.set_title("Coverage by epistemic quartile"); letter(ax, "d", x=-0.3, y=1.04)
    out["epistemic_coverage"] = ec[:, 1].tolist()
    ax = fig.add_axes([0.42, 0.08, 0.22, 0.32]); ks = list(rg)
    ax.bar(range(len(ks)), [rg[k][0] for k in ks], yerr=[rg[k][1] for k in ks], color=[BLUE, GREY], error_kw=dict(lw=0.6, capsize=2))
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(["fixed 2-hop\n(MetaboCast)", "learned\nGCN"]); ax.set_ylabel("median r")
    ax.set_title("Simple prior vs learned GCN"); letter(ax, "e", x=-0.45, y=1.04); out["gcn"] = {k: rg[k] for k in ks}
    save(fig, "SuppFig_predictability")
    return out


def ed6():
    d = R("rigor_spatial_control")["conditions"]
    labs = ["intact graph\n(diffusion)", "no diffusion", "permuted\ncoordinates", "random graph\nedges", "gene–metabolite\npairing shuffled"]
    v = list(d.values())
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.bar(range(5), [x[0] for x in v], yerr=[x[1] for x in v], color=[BLUE, GREY, GOLD, RED, "#555555"], error_kw=dict(lw=0.6, capsize=2))
    ax.axhline(v[1][0], color="grey", ls=":", lw=0.8); ax.axhline(0, color="k", lw=0.5)
    for i, x in enumerate(v):
        ax.text(i, max(x[0], 0) + x[1] + 0.006, f"{x[0]:.3f}", ha="center", fontsize=6)
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=6); ax.set_ylabel("median Spearman (mean ± s.d.)")
    save(fig, "SuppFig_spatial_controls")
    return {k: x for k, x in zip(d, v)}


def run():
    return {"SuppFig_robustness": ed4(), "SuppFig_predictability": ed5(), "SuppFig_spatial_controls": ed6()}
