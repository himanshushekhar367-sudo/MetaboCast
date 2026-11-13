"""README overview image (docs/overview.png): transcriptome-only spatial data in, predicted metabolite maps with per-ion
reliability out (predictions for the glioblastoma Visium section of the manuscript, two transfer-supported ions).

Usage: python docs/make_overview.py docs/overview.png"""
import json, os, sys
os.environ.setdefault("METABOCAST_FIGURES", os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "figures"))
import numpy as np, pandas as pd                                     # noqa: E402
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch       # noqa: E402
from common import spotmap                                           # noqa: E402

R = os.path.join(ROOT, "reproduce", "results", "crosscancer")
g = pd.read_csv(f"{R}/GBM_Visium_maps.csv")
s = json.load(open(f"{R}/crosscancer_summary.json"))
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9})
BLUE, GREY = "#2F5F98", "#5f6368"
INPUT_CMAP, LO_HI = "Greens", (0.30, 0.80)
fig = plt.figure(figsize=(10.5, 3.4), dpi=200)

# left: the input (each spot carries its own expression profile)
ax = fig.add_axes([0.01, 0.10, 0.22, 0.70])
rng = np.random.default_rng(0)
spotmap(ax, g.x, -g.y, rng.uniform(*LO_HI, len(g)), 0, 1, cmap=INPUT_CMAP)
fig.text(0.12, 0.91, "Spatial transcriptomics", ha="center", fontsize=11, fontweight="bold")
fig.text(0.12, 0.84, "genes + spot or cell coordinates", ha="center", fontsize=8.5, color=GREY)
fig.text(0.12, 0.03, "Visium, Visium HD, Slide-seq, Stereo-seq;\nXenium, CosMx, MERSCOPE as pseudo-spots",
         ha="center", va="bottom", fontsize=7.2, color=GREY, linespacing=1.4)

# middle: the model
box = FancyBboxPatch((0.315, 0.17), 0.30, 0.66, boxstyle="round,pad=0.012,rounding_size=0.03", fc=BLUE, ec="none",
                     transform=fig.transFigure)
fig.patches.append(box)
fig.text(0.465, 0.72, "MetaboCast", ha="center", color="white", fontsize=15, fontweight="bold")
lines = ["prepare_query: coordinates, gene IDs, normalisation",
         "PCs + diffusion features [P, ÂP, Â²P]",
         "ridge + extremely randomized trees",
         "trained on paired MSI + spatial transcriptomics"]
for i, t in enumerate(lines):
    fig.text(0.465, 0.59 - i * 0.105, t, ha="center", color="#dfe8f4", fontsize=8.3)
for x0, x1 in [(0.245, 0.30), (0.63, 0.685)]:
    fig.patches.append(FancyArrowPatch((x0, 0.50), (x1, 0.50), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=14, lw=2, color="#444444"))

# right: predicted maps with their measured reliability on held-out DESI-MSI sections
fig.text(0.85, 0.91, "Predicted spatial metabolome", ha="center", fontsize=11, fontweight="bold")
fig.text(0.85, 0.84, "transcriptome-only glioblastoma (10x Visium)", ha="center", fontsize=8.5, color=GREY)
for k, (col, lab) in enumerate([("Arachidonate(FA20:4)", "arachidonate"), ("DHA(FA22:6)", "DHA")]):
    ax = fig.add_axes([0.70 + k * 0.15, 0.22, 0.14, 0.54])
    v = g[col].values; lo, hi = np.percentile(v, [2, 98])
    spotmap(ax, g.x, -g.y, v, lo, hi)
    ax.set_title(lab, fontsize=9, pad=2)
    rw, rl = s["ion_r_within"][col], s["ion_r_loso"][col]
    ax.text(0.5, -0.06, f"r = {rw:.2f} within section\nr = {rl:.2f} on unseen sections\n{s['ion_tier'][col]}", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.8, color=GREY, linespacing=1.3)
fig.text(0.85, 0.01, "r: accuracy of the putative ion on paired DESI-MSI sections;\nevery m/z channel gets its own values and tier",
         ha="center", va="bottom", fontsize=6.6, color=GREY, linespacing=1.35)
fig.savefig(sys.argv[1], dpi=200, facecolor="white")
print("saved", sys.argv[1])
