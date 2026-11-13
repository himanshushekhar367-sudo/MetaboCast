"""Figure 7 and the HCC and GBM supplementary figures: one cross-cancer prediction run on transcriptome-only HCC and
GBM sections (reproduce/crosscancer/regenerate_crosscancer.py and derive_figure_sources.py). Every value is a
prediction; ions are marked by their DESIUM accuracy tier (transfer-supported: held-out-section r >= 0.20;
within-section only: within-section r >= 0.30 with weaker transfer)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from common import BLUE, CATEGORICAL, RED, OUT, ann_table, fmt_r, letter, pretty, res, save, spotmap
from manifest import figure_stem                      # noqa: E402  (reproduce/ is on the path via common)

TIER_MARK = {"transfer-supported": "", "within-section only": " ◦", "low": " ×"}
TIER_NOTE = "◦ within-section only (DESIUM transfer r < 0.20); unmarked: transfer-supported"


def XS(name):
    return res("crosscancer", name)


def short(name):
    return pretty(name.replace("(GSH)", ""))


def curated_adduct(mz_values):
    """Adduct of the curated channels shown by name ([M-H]- for every one), checked against the annotation table."""
    A = ann_table(); amz = A.mz.to_numpy(float)
    adds = {A.adduct.iloc[int(np.argmin(np.abs(amz - float(m))))] for m in mz_values}
    assert adds == {"[M-H]-"}, adds
    return "[M−H]−"


def mark(name, ambiguous, tier):
    """Display name with the marks used in every HCC/GBM panel: † more than one exact-mass candidate on the channel,
    ◦ within-section only."""
    return short(name) + ("†" if bool(ambiguous) else "") + TIER_MARK.get(tier, "")


def regmaps(fig, d, col, rect, vmin, vmax):
    x0, y0, w, h = rect; regs = [("N", "N · normal"), ("L", "L · leading edge"), ("P", "P · PVTT"), ("T", "T · tumour")]
    for i, (r, lab) in enumerate(regs):
        ax = fig.add_axes([x0 + i * w / 4, y0, w / 4 * 0.94, h]); s = d[d.region == r]
        sc = spotmap(ax, s.x, -s.y, s[col], vmin, vmax)
        ax.set_title(lab, fontsize=6.5, color=RED if r == "P" else "black")
    return sc


def fig7():
    maps = pd.read_csv(XS("HCC2_maps.csv")); s9 = pd.read_csv(XS("S9_and_Fig7c_source.csv"))
    s10 = pd.read_csv(XS("S10_metabolite_x_protein.csv"))
    nich = pd.read_csv(XS("Fig7e_niches.csv")); nf = pd.read_csv(XS("Fig7f_metabolite_by_niche.csv"))
    ADD = curated_adduct(list(s9.mz) + list(s10.mz) + list(nf.mz))
    fig = plt.figure(figsize=(7.2, 8.6))
    lo, hi = np.percentile(maps.Arachidonate, [1, 99])
    sc = regmaps(fig, maps, "Arachidonate", [0.02, 0.795, 0.86, 0.15], lo, hi)
    cax = fig.add_axes([0.90, 0.82, 0.012, 0.11]); cb = fig.colorbar(sc, cax=cax)
    cb.set_label("predicted log1p", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    fig.text(0.02, 0.985, "a", fontsize=9, fontweight="bold")
    fig.text(0.06, 0.985, "HCC-2 (transcriptome only): predicted arachidonate, m/z 303.232 [M−H]−, a transfer-supported ion",
             fontsize=7.2)
    # b: HCC-2 normal -> PVTT, the metabolites of panel c
    ax = fig.add_axes([0.17, 0.50, 0.28, 0.23])
    b = s9.set_index("metabolite"); order = list(b.index)
    v = b["HCC-2 N->P"].to_numpy(float); y = np.arange(len(order))[::-1]
    for k, (m, val) in enumerate(zip(order, v)):
        col = RED if val > 0 else BLUE; filled = b.loc[m, "tier"] == "transfer-supported"
        ax.barh(y[k], val, color=col if filled else "white", edgecolor=col, lw=0.8, height=0.7)
    ax.set_yticks(y); ax.set_yticklabels([mark(m, b.loc[m, "ambiguous_annotation"], b.loc[m, "tier"]) for m in order], fontsize=5.8)
    ax.axvline(0, color="k", lw=0.6)
    lim = max(0.05, float(np.abs(v).max()) * 1.4)
    for k, val in zip(y, v):
        ax.text(val + (0.03 * lim if val > 0 else -0.03 * lim), k, fmt_r(val, sign=True), va="center",
                ha="left" if val > 0 else "right", fontsize=5.3)
    ax.set_xlim(-lim, lim); ax.set_xlabel("Δ predicted log intensity, normal → PVTT")
    ax.set_title(f"HCC-2: predicted normal→PVTT change ({ADD})\n(filled, transfer-supported; open, within-section only)",
                 fontsize=6.4)
    letter(ax, "b", x=-0.62, y=1.03)
    # c: cross-patient contrasts
    ax = fig.add_axes([0.60, 0.50, 0.30, 0.23])
    cols = ["HCC-1 N->T", "HCC-2 N->T", "HCC-3 N->T", "HCC-4 N->T", "HCC-2 N->P"]
    M = b[cols].to_numpy(float)
    vmax = max(0.05, float(np.nanmax(np.abs(M))))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt_r(M[i, j], sign=True), ha="center", va="center", fontsize=4.8,
                    color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([mark(m, b.loc[m, "ambiguous_annotation"], b.loc[m, "tier"]) for m in order],
                                                         fontsize=5.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["HCC-1\nN→T", "HCC-2\nN→T", "HCC-3\nN→T", "HCC-4\nN→T", "HCC-2\nN→P"], fontsize=5.6)
    ax.axvline(3.5, color="k", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=5.5); cb.set_label("Δ predicted log", fontsize=6)
    ax.set_title(f"Predicted tumour − normal contrast ({ADD})\n(one contrast per patient; not replicates)", fontsize=6.4)
    letter(ax, "c", x=-0.42, y=1.03)
    # d: GBM protein co-localisation (exploratory)
    ax = fig.add_axes([0.15, 0.08, 0.38, 0.32])
    meta = ["predicted_metabolite", "mz", "within_section_r", "transfer_r", "tier", "ambiguous_annotation"]
    prot = [c for c in s10.columns if c not in meta]
    M = s10[prot].to_numpy(float)
    rn = [mark(m, a, t) for m, a, t in zip(s10.predicted_metabolite, s10.ambiguous_annotation, s10.tier)]
    vmax = max(0.5, float(np.nanmax(np.abs(M))))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(rn))); ax.set_yticklabels(rn, fontsize=5.8)
    ax.set_xticks(range(len(prot))); ax.set_xticklabels([p.replace("_2", " (2)") for p in prot], rotation=90, fontsize=5.2)
    names = list(s10.predicted_metabolite)
    if "N-acetylaspartate" in names:
        k = names.index("N-acetylaspartate"); ax.add_patch(Rectangle((-0.5, k - 0.5), len(prot), 1, fill=False, ec=RED, lw=1.2))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015); cb.ax.tick_params(labelsize=5.5); cb.set_label("Spearman ρ", fontsize=6)
    ax.set_title(f"GBM CytAssist (exploratory): predicted metabolite ({ADD}) × measured protein\n"
                 "boxed: N-acetylaspartate, a possible failure signal", fontsize=6.4)
    letter(ax, "d", x=-0.32, y=1.03)
    # e, f: GBM niches (exploratory)
    ax = fig.add_axes([0.68, 0.245, 0.17, 0.16])
    pal = dict(zip(["n0", "n1", "n2", "n3", "n4"], CATEGORICAL))
    spotmap(ax, nich.pxl_col, -nich.pxl_row, colors=nich.niche.map(pal))
    ax.set_title("GBM Visium (exploratory): niches of the\npredicted transfer-supported channels (k=5)", fontsize=6.2)
    ax.legend(handles=[Patch(color=c, label=n) for n, c in pal.items()], frameon=False, fontsize=5.5, loc="upper left",
              bbox_to_anchor=(1.0, 1.0), handlelength=1.0, handleheight=1.0, handletextpad=0.4)
    letter(ax, "e", x=-0.456, y=1.12)
    ax = fig.add_axes([0.76, 0.02, 0.15, 0.17])
    nm = [mark(m, a, t) for m, a, t in zip(nf.metabolite, nf.ambiguous_annotation, nf.tier)]
    M = nf[["n0", "n1", "n2", "n3", "n4"]].to_numpy(float); vmax = max(1.0, float(np.nanmax(np.abs(M))))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm, fontsize=5.5); ax.set_xticks(range(5)); ax.set_xticklabels(list(pal), fontsize=5.8)
    for j, c in enumerate(pal.values()):                     # niche colour swatch under each column label
        ax.add_patch(Rectangle((j - 0.3, len(nm) - 0.5 + 0.12), 0.6, 0.38, color=c, clip_on=False, lw=0))
    ax.tick_params(axis="x", length=0, pad=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03); cb.ax.tick_params(labelsize=5.5); cb.set_label("mean z", fontsize=6)
    ax.set_title(f"Predicted metabolite ({ADD}) per niche", fontsize=6.6); letter(ax, "f", x=-1.05, y=1.05)
    fig.text(0.02, 0.003, TIER_NOTE + "; † more than one candidate", fontsize=5.6)
    save(fig, "Figure7")
    return dict(hcc2_NtoP=b["HCC-2 N->P"].round(3).to_dict(), tiers=b.tier.to_dict())


def s5():
    d = pd.read_csv(XS("HCC2_maps.csv")); tr = pd.read_csv(XS("HCC2_trajectory.csv")); s8 = pd.read_csv(XS("S8_supported_annotated.csv"))
    s9 = pd.read_csv(XS("S9_and_Fig7c_source.csv")).set_index("metabolite")
    fig = plt.figure(figsize=(7.2, 6.2))
    for k, (col, met, y0) in enumerate([("GSH", "Glutathione(GSH)", 0.70), ("Hexose", "Hexose", 0.42)]):
        ADD = curated_adduct([s9.loc[met, "mz"]])
        lab = f"predicted {short(met)} {ADD}" + ("†" if s9.loc[met, "ambiguous_annotation"] else "") + TIER_MARK.get(s9.loc[met, "tier"], "")
        lo, hi = np.percentile(d[col], [1, 99])
        sc = regmaps(fig, d, col, [0.04, y0, 0.84, 0.24], lo, hi)
        cax = fig.add_axes([0.9, y0 + 0.03, 0.012, 0.16]); cb = fig.colorbar(sc, cax=cax); cb.ax.tick_params(labelsize=5.5)
        fig.text(0.01, y0 + 0.12, lab, rotation=90, va="center", fontsize=6.5); fig.text(0.01, y0 + 0.255, "ab"[k], fontsize=9, fontweight="bold")
    top = s8.reindex(s8.delta_NtoP_log.abs().sort_values(ascending=False).index).head(12).sort_values("delta_NtoP_log")
    ax = fig.add_axes([0.2, 0.08, 0.27, 0.27])

    def ionlab(a):
        name, add = a.split(";")[0].strip().rsplit(" [", 1)
        return f"{pretty(name)} [{add.replace('-', '−')}"
    lab = [ionlab(a) + ("†" if amb else "") + TIER_MARK.get(t, "")
           for a, amb, t in zip(top.annotation, top.ambiguous_annotation, top.tier)]
    for k, (val, t) in enumerate(zip(top.delta_NtoP_log, top.tier)):
        col = RED if val > 0 else BLUE
        ax.barh(k, val, color=col if t == "transfer-supported" else "white", edgecolor=col, lw=0.8, height=0.7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(lab, fontsize=5.4); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Δ predicted log intensity, normal → PVTT")
    ax.set_title("Largest predicted normal→PVTT shifts\n(annotated ions, two upper tiers)", fontsize=6.4)
    letter(ax, "c", x=-0.8, y=1.03)
    ax = fig.add_axes([0.6, 0.08, 0.36, 0.27])
    for ion, c in zip(["Arachidonate", "DHA", "Hexose", "GSH", "Palmitate", "Inosine"],
                      ["#2ca02c", "#ff7f0e", "#8c564b", "#1f77b4", "#9467bd", "#d62728"]):
        t = tr[tr.ion == ion].set_index("region").reindex(["N", "L", "P", "T"])
        nm = {"GSH": "glutathione", "DHA": "DHA"}.get(ion, ion.lower())
        ax.plot(range(4), t.delta_vs_N, marker="o", ms=3, color=c, lw=1,
                label=nm + (" ◦" if ion in ("GSH", "Palmitate") else ""))
    ax.axhline(0, color="grey", lw=0.5); ax.set_xticks(range(4)); ax.set_xticklabels(["normal", "leading\nedge", "PVTT", "tumour"])
    ax.set_ylabel("Δ predicted vs normal (log)"); ax.legend(frameon=False, fontsize=5.5, ncol=2)
    ax.set_title(f"Predicted profile across HCC-2 regions ({curated_adduct(s9.mz)})"); letter(ax, "d", x=-0.25, y=1.03)
    fig.text(0.02, -0.01, TIER_NOTE + "; † more than one candidate", fontsize=5.6)
    save(fig, "SuppFig_HCC2_subsections")


def s6():
    """a, tumour- minus normal-typed spots pooled over each patient's sections; b, the same contrast inside the
    leading-edge section, which holds hundreds of spots of both types in every patient."""
    s9 = pd.read_csv(XS("S9_and_Fig7c_source.csv")).set_index("metabolite"); tiers = s9.tier
    pooled = pd.read_csv(XS("HCC_TypeLevel_TumorCell_vs_NormalCell.csv")).pivot(index="metabolite", columns="patient", values="delta")
    edge_t = pd.read_csv(XS("HCC_TypeLevel_LeadingEdge.csv"))
    edge = edge_t.pivot(index="metabolite", columns="patient", values="delta")
    order = [m for m in tiers.index if m in pooled.index]
    pooled, edge = pooled.reindex(order), edge.reindex(order)[pooled.columns]
    vmax = max(0.05, float(np.nanmax(np.abs(np.r_[pooled.values.ravel(), edge.values.ravel()]))))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.9), gridspec_kw=dict(wspace=0.08))
    n_edge = edge_t.drop_duplicates("patient").set_index("patient")
    for k, (ax, P, ttl) in enumerate([(axes[0], pooled, "pooled over each patient's sections"),
                                      (axes[1], edge, "within the leading-edge section")]):
        im = ax.imshow(P.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                ax.text(j, i, fmt_r(P.values[i, j], sign=True, nd=3), ha="center", va="center", fontsize=5.2,
                        color="white" if abs(P.values[i, j]) > 0.65 * vmax else "black")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([mark(o, s9.loc[o, "ambiguous_annotation"], tiers[o]) for o in order] if k == 0 else [], fontsize=6)
        ax.set_xticks(range(P.shape[1]))
        ax.set_xticklabels(P.columns if k == 0 else [f"{p}\n{int(n_edge.loc[p, 'n_tumour'])}/{int(n_edge.loc[p, 'n_normal'])}"
                                                     for p in P.columns], fontsize=6)
        ax.set_title(ttl, fontsize=6.6); letter(ax, "ab"[k], x=-0.05 if k else -0.55, y=1.04)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Δ predicted log (tumour − normal spots)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    fig.suptitle(f"Per-spot histological type contrast ({curated_adduct(s9.loc[order, 'mz'])}); "
                 "b: tumour/normal spots in each leading-edge section", fontsize=6.6, y=0.995)
    save(fig, "SuppFig_HCC_typelevel")
    out = pd.concat({"pooled": pooled, "leading_edge": edge}, axis=1)
    out.to_csv(f"{OUT}/{figure_stem('SuppFig_HCC_typelevel')}_source.csv")
    return out


def s7():
    g = pd.read_csv(XS("GBM_Visium_maps.csv")); s8 = pd.read_csv(XS("S8_full_all_ions.csv"))
    want = {"Arachidonate(FA20:4)": 303.2323, "DHA(FA22:6)": 327.2314, "Hexose": 179.055, "Glutathione(GSH)": 306.0756}
    fig = plt.figure(figsize=(7.2, 2.5)); rr = {}
    for i, (c, m) in enumerate(want.items()):
        row = s8.loc[(s8.mz - m).abs().idxmin()]; rr[c] = (float(row.within_section_r), float(row.transfer_r))
        ax = fig.add_axes([0.01 + i * 0.245, 0.02, 0.21, 0.78]); lo, hi = np.percentile(g[c], [1, 99])
        sc = spotmap(ax, g.x, -g.y, g[c], lo, hi)
        ax.set_title(f"predicted {short(c)} {curated_adduct([row.mz])}" + ("†" if row.ambiguous_annotation else "")
                     + f"{TIER_MARK.get(row.tier, '')}\n(DESIUM r: within {row.within_section_r:.2f}, "
                     f"transfer {row.transfer_r:.2f})", fontsize=6.0)
        cax = fig.add_axes([0.01 + i * 0.245 + 0.205, 0.15, 0.007, 0.5]); cb = fig.colorbar(sc, cax=cax); cb.ax.tick_params(labelsize=5)
        letter(ax, "abcd"[i], x=-0.02, y=1.14)
    save(fig, "SuppFig_GBM_maps")
    return rr


def ed3():
    g = pd.read_csv(XS("HCC_GSH_per_patient.csv")); g["delta"] = g.mean_T - g.mean_N
    pooled_spot = (g.mean_T * g.n_T).sum() / g.n_T.sum() - (g.mean_N * g.n_N).sum() / g.n_N.sum()
    fig = plt.figure(figsize=(5.0, 2.4))
    ax = fig.add_axes([0.1, 0.15, 0.35, 0.72])
    ys = g.mean_T.values.copy(); order = np.argsort(ys); span = float(np.ptp(np.r_[g.mean_N, g.mean_T])) or 1.0
    lab_y = ys.copy()
    for a_, b_ in zip(order[:-1], order[1:]):                       # keep patient labels at least 6% of the range apart
        if lab_y[b_] - lab_y[a_] < 0.06 * span:
            lab_y[b_] = lab_y[a_] + 0.06 * span
    for k, (_, r) in enumerate(g.iterrows()):
        c = RED if r.delta > 0 else BLUE
        ax.plot([0, 1], [r.mean_N, r.mean_T], color=c, lw=1); ax.scatter([0, 1], [r.mean_N, r.mean_T], color=c, s=12, zorder=3)
        ax.text(1.05, lab_y[k], r.patient, fontsize=5.8, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "tumour"]); ax.set_xlim(-0.3, 1.45); ax.set_ylabel("mean predicted GSH (log)")
    ax.set_title(f"{int((g.delta > 0).sum())}/4 patients: predicted tumour > normal", fontsize=6.8); letter(ax, "a", x=-0.3, y=1.03)
    ax = fig.add_axes([0.6, 0.15, 0.37, 0.72])
    ax.bar(g.patient, g.delta, color=[RED if v > 0 else BLUE for v in g.delta]); ax.axhline(0, color="k", lw=0.6)
    rng_ = float(np.abs(g.delta).max()) or 1.0; ax.set_ylim(min(0, g.delta.min()) - 0.25 * rng_, max(0, g.delta.max()) + 0.25 * rng_)
    for i, v in enumerate(g.delta):
        ax.text(i, v + (0.06 * rng_ if v > 0 else -0.06 * rng_), fmt_r(v, sign=True, nd=3), ha="center",
                va="bottom" if v > 0 else "top", fontsize=6)
    ax.set_ylabel("Δ predicted GSH (tumour − normal)")
    ax.set_title(f"mean of patients {fmt_r(g.delta.mean(), sign=True, nd=3)}; spot-pooled {fmt_r(pooled_spot, sign=True, nd=3)}",
                 fontsize=6.8)
    letter(ax, "b", x=-0.3, y=1.03)
    save(fig, "SuppFig_GSH_patients")
    return dict(per_patient=g.delta.round(3).tolist(), mean=float(g.delta.mean()), pooled=float(pooled_spot), n_up=int((g.delta > 0).sum()))


def run():
    out = {"Figure7": fig7(), "SuppFig_GBM_maps": s7(), "SuppFig_GSH_patients": ed3()}
    s5(); tl = s6(); out["SuppFig_HCC_typelevel"] = {b: tl[b].round(3).to_dict() for b in ("pooled", "leading_edge")}
    return out
