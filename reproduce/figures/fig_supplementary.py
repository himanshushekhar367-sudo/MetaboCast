"""Model-independent supplementary figures: spatial cross-validation blocks (SuppFig_cv_folds) and SMA registration
audit (SuppFig_SMA_registration). Numbers: reproduce/manifest.py."""
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from common import BLUE, CATEGORICAL, ORANGE, letter, res, save, spotmap_masked


def s2(section="BC_525"):
    """Cross-validation blocks of one section. Each spot is drawn as a separate dot, so that single spots stay visible,
    and each block is numbered at its most central spot."""
    co = np.load(res("within_desium_conformal.npz"))[f"{section}__coords"]
    folds = KMeans(5, n_init=3, random_state=0).fit_predict(co)          # as in reproduce/within_benchmark.py
    counts = [int((folds == f).sum()) for f in range(5)]
    xy = np.c_[co[:, 0], -co[:, 1]]
    pitch = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))       # centre-to-centre spot distance
    fig = plt.figure(figsize=(6.5, 4.6))
    ax = fig.add_axes([0.0, 0.0, 0.74, 0.92])
    ax.add_collection(EllipseCollection(widths=0.72 * pitch, heights=0.72 * pitch, angles=0, units="xy", offsets=xy,
                                        offset_transform=ax.transData, facecolors=[CATEGORICAL[f] for f in folds],
                                        linewidths=0))
    pad = 1.2 * pitch
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.set_axis_off()
    for f in range(5):
        p = xy[folds == f]; c = p[np.argmin(((p - p.mean(0)) ** 2).sum(1))]
        ax.text(c[0], c[1], str(f + 1), ha="center", va="center", fontsize=11, fontweight="bold", color="#1A1A1A",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    fig.text(0.0, 0.975, f"{section.replace('_', ' ')}: five spatial blocks", fontsize=8.5, ha="left", va="top")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=CATEGORICAL[f],
                               markeredgewidth=0, label=f"block {f + 1} ({counts[f]} spots)") for f in range(5)],
               loc="upper left", bbox_to_anchor=(0.755, 0.86), frameon=False, fontsize=7.5, handletextpad=0.4,
               labelspacing=0.7)
    save(fig, "SuppFig_cv_folds")
    return {f"block {f + 1}": counts[f] for f in range(5)}


def s3():
    """SMA registration: total ion count under the published spot-pixel pairing, agreement with it of registration by
    tissue geometry (outcome-free) and of the coupling-chosen orientation (outcome-dependent), and the post-hoc coupling
    check."""
    au = pd.read_csv(res("sma", "sma_registration_audit.csv")); tic = pd.read_csv(res("sma", "sma_registration_audit_tic.csv.gz"))
    q = pd.read_csv(res("sma_coregistration_qc.csv")).set_index("section").reindex(au.section)
    fig = plt.figure(figsize=(7.2, 7.4))
    for i, r in au.reset_index(drop=True).iterrows():
        row, col = divmod(i, 5)
        ax = fig.add_axes([0.01 + col * 0.198, 0.74 - row * 0.20, 0.18, 0.17])
        t = tic[tic.section == r.section]; paired = t.log_tic.notna()
        lo, hi = np.percentile(t.log_tic[paired], [1, 99])
        spotmap_masked(ax, t.x, -t.y, t.log_tic, lo, hi)           # one tiling; unpaired spots grey
        ax.set_title(f"{r.section.split('_', 2)[2]}\n{r.matrix} · {r.subject} · {r.brain_area.replace('SubNigra', 'nigra').lower()}", fontsize=5.4)
    fig.text(0.01, 0.955, "a", fontsize=9, fontweight="bold")
    fig.text(0.04, 0.955, "MALDI total ion count of the pixel paired with each spot (published pairing); grey, in-tissue spots without a pair",
             fontsize=6.5)
    ax = fig.add_axes([0.08, 0.06, 0.40, 0.26]); y = np.arange(len(au))[::-1]; h = 0.38
    ax.barh(y + h / 2, au.geometry_within_1px, h, color=BLUE, label="tissue-geometry registration")
    if "coupling_within_1px" in au:
        ax.barh(y - h / 2, au.coupling_within_1px, h, color=ORANGE, label="orientation chosen by coupling")
    ax.set_yticks(y); ax.set_yticklabels([f"{s_.split('_', 2)[2]} ({m})" for s_, m in zip(au.section, au.matrix)], fontsize=5.4)
    ax.set_xlim(0, 1); ax.set_xlabel("spots within one pixel of the published pixel")
    ax.legend(frameon=False, fontsize=5.6, loc="center right", bbox_to_anchor=(1.0, 0.42))
    letter(ax, "b", x=-0.32, y=1.04)
    ax = fig.add_axes([0.62, 0.06, 0.35, 0.26])
    ax.barh(y, q.within_section_coupling, color="#4C9BA6", height=0.66, zorder=2)
    ax.scatter(q.shuffled_null, y, marker="D", s=12, color="#222222", zorder=3, label="shuffled-spot null")
    for yi, v in zip(y, q.within_section_coupling):
        ax.text(v + 0.006, yi, f"{v:.2f}", va="center", fontsize=5.4)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.axvline(0, color="#BBBBBB", lw=0.8)
    ax.set_xlabel("coupling after registration (30-PC ridge, median Spearman)")
    ax.legend(frameon=False, fontsize=5.6, loc="lower right"); letter(ax, "c", x=-0.08, y=1.08)
    save(fig, "SuppFig_SMA_registration")
    return dict(geometry_within_1px=au.set_index("section").geometry_within_1px.round(3).to_dict(),
                coupling_within_1px=(au.set_index("section").coupling_within_1px.round(3).to_dict()
                                     if "coupling_within_1px" in au else None),
                coupling=q.within_section_coupling.round(3).to_dict())


def run():
    return {"SuppFig_cv_folds": s2(), "SuppFig_SMA_registration": s3()}
