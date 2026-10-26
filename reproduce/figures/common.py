"""Shared style, locations and loaders for the manuscript figures.

Results are read from reproduce/results (override with METABOCAST_RESULTS); figures are written to
METABOCAST_FIGURES (default: figures/ in the working directory) as PNG (300 dpi) and PDF.
Spatial maps draw every spot as its own gap-free tile (spotmap) on the viridis scale."""
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib import font_manager                 # noqa: E402
from matplotlib.cm import ScalarMappable            # noqa: E402
from matplotlib.collections import PolyCollection   # noqa: E402
from matplotlib.colors import Normalize             # noqa: E402
from scipy.spatial import Voronoi, cKDTree          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import sys                                           # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
from manifest import figure_stem                     # noqa: E402
RES = os.environ.get("METABOCAST_RESULTS", os.path.join(os.path.dirname(HERE), "results"))
OUT = os.environ.get("METABOCAST_FIGURES", "figures")
os.makedirs(OUT, exist_ok=True)

fams = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Arial" if "Arial" in fams else ("Liberation Sans" if "Liberation Sans" in fams else "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, GOLD, RED, GREY, ORANGE, TEAL = "#2F5F98", "#C49A3A", "#B5413A", "#9AA0A6", "#D55E00", "#4C9BA6"
LIGHTBLUE, DARK = "#7BA7D9", "#3A3A3A"
MAPCMAP = "viridis"                                   # continuous spatial maps
CATEGORICAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9"]   # categorical maps (Okabe-Ito subset)
DESIUM =["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]
SHORT = {"BC_515_Section_1": "BC 515-1", "BC_515_Section_2": "BC 515-2", "BC_525": "BC 525", "BC_823": "BC 823",
         "LC_091": "LC 091", "LC_170": "LC 170", "LC_276": "LC 276"}
SMA_GROUPS = {"dhb": ["SMA_DHB_V11L12-038_A1", "SMA_DHB_V11L12-038_B1"],
              "fmp_mouse": ["SMA_FMP_V11L12-109_A1", "SMA_FMP_V11L12-109_B1", "SMA_FMP_V11L12-109_C1",
                            "SMA_FMP_V11T16-085_A1", "SMA_FMP_V11T16-085_B1", "SMA_FMP_V11T16-085_C1"],
              "fmp_human": ["SMA_FMP_V11T17-102_A1", "SMA_FMP_V11T17-102_B1", "SMA_FMP_V11T17-102_C1", "SMA_FMP_V11T17-102_D1"]}
NIGRA = {"SMA_FMP_V11T16-085_A1", "SMA_FMP_V11T16-085_B1", "SMA_FMP_V11T16-085_C1"}
# chemical groups of single-class putative candidates (the same split as reproduce/rigor_predictability.py): lipids are
# the hydrophobic classes; every other annotated class is polar, including nucleotides, nucleosides and nucleobases,
# carnitines, choline and its phosphodiesters (class "lipid headgroup"), cofactors, polyamines and vitamins
LIPID = {"sphingolipid", "phospholipid", "fatty acid", "lysophospholipid", "glycerolipid", "sterol"}


def is_polar(c):
    return isinstance(c, str) and c not in LIPID and c not in ("", "ambiguous")


def res(*parts):
    return os.path.join(RES, *parts)


def letter(ax, s, x=-0.08, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    """Write <stem>.png (300 dpi) and <stem>.pdf; the stem carries the item's number from reproduce/manifest.py."""
    stem = figure_stem(name)
    fig.savefig(os.path.join(OUT, f"{stem}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, f"{stem}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fmt_r(r, sign=False, nd=2):
    """Number for a panel title or cell: nd decimals, true minus sign and no signed zero."""
    s = f"{r:+.{nd}f}" if sign else f"{r:.{nd}f}"
    zero = f"{0:.{nd}f}"
    return re.sub(rf"^[+-]{re.escape(zero)}$", zero, s).replace("-", "−")


FA_COMMON = {"FA16:0": "palmitate", "FA18:0": "stearate", "FA18:1": "oleate", "FA18:2": "linoleate",
             "FA20:4": "arachidonate", "FA22:6": "DHA", "FA20:5": "EPA"}


def pretty(name):
    """Metabolite name for display: 'Glutathione(GSH)' -> 'glutathione (GSH)', 'N-Acetylaspartate' ->
    'N-acetylaspartate'; fatty acids named in the text show the common name with the shorthand ('FA20:4(AA)' ->
    'arachidonate (FA20:4)'); other lipid shorthand such as 'PE38:4' or 'Cer d18:1/16:0' keeps its form."""
    m = re.fullmatch(r"(FA\d+:\d+)(?:\(\w+\))?", name)
    if m and m.group(1) in FA_COMMON:
        return f"{FA_COMMON[m.group(1)]} ({m.group(1)})"
    name = re.sub(r"(?<=\w)\((?=[A-Z])", " (", name)
    name = re.sub(r"^(?!Cer\b)([A-Z])(?=[a-z])", lambda m: m.group(1).lower(), name)
    return re.sub(r"^N-([A-Z])(?=[a-z])", lambda m: "N-" + m.group(1).lower(), name)


def _clip(subject, clipper):
    """Sutherland-Hodgman intersection of a convex polygon with a convex, counter-clockwise clip polygon."""
    out = [tuple(p) for p in subject]
    for a, b in zip(clipper, np.roll(clipper, -1, axis=0)):
        if not out:
            break
        inp, out = out, []
        ex, ey = b[0] - a[0], b[1] - a[1]
        s = inp[-1]
        fs = ex * (s[1] - a[1]) - ey * (s[0] - a[0])
        for e in inp:
            fe = ex * (e[1] - a[1]) - ey * (e[0] - a[0])
            if (fe >= 0) != (fs >= 0):
                t = fs / (fs - fe)
                out.append((s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])))
            if fe >= 0:
                out.append(e)
            s, fs = e, fe
    return np.array(out)


_CELLS = {}


def spot_cells(xy, box=None):
    """One polygon per spot: its Voronoi cell, so that neighbouring spots meet without gaps. Cells at the tissue
    edge (or beside a hole) are trimmed to the spot's own neighbourhood, a disc of the lattice cell radius,
    so they do not spill outwards; with box=(x0, x1, y0, y1) every cell is cut to that rectangle instead."""
    xy = np.ascontiguousarray(xy, dtype=float)
    key = (xy.shape, xy.tobytes(), box)
    if key in _CELLS:
        return _CELLS[key]
    d = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))           # spot pitch
    lo, hi = xy.min(0) - 50 * d, xy.max(0) + 50 * d                      # far frame keeps every real cell bounded
    t = np.linspace(0, 1, 9)
    frame = np.unique(np.r_[np.c_[lo[0] + t * (hi[0] - lo[0]), np.full(9, lo[1])],
                            np.c_[lo[0] + t * (hi[0] - lo[0]), np.full(9, hi[1])],
                            np.c_[np.full(9, lo[0]), lo[1] + t * (hi[1] - lo[1])],
                            np.c_[np.full(9, hi[0]), lo[1] + t * (hi[1] - lo[1])]], axis=0)
    vor = Voronoi(np.r_[xy, frame])
    ang = np.linspace(0, 2 * np.pi, 33)[:-1]
    disc = (d / np.sqrt(3)) * np.c_[np.cos(ang), np.sin(ang)]             # circumradius of a hexagonal cell
    rect = None if box is None else np.array([[box[0], box[2]], [box[1], box[2]], [box[1], box[3]], [box[0], box[3]]])
    cells = []
    for i, c in enumerate(xy):
        v = vor.vertices[vor.regions[vor.point_region[i]]]
        v = v[np.argsort(np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0]))]
        if rect is not None:
            v = _clip(v, rect)
        elif np.hypot(v[:, 0] - c[0], v[:, 1] - c[1]).max() > 0.75 * d:
            v = _clip(v, c + disc)
        cells.append(v)
    _CELLS[key] = cells
    return cells


def spotmap(ax, x, y, values=None, vmin=None, vmax=None, cmap=MAPCMAP, colors=None, box=None):
    """Spatial map with one filled tile per spot, coloured by value (cmap, vmin-vmax) or by explicit colours."""
    pc = PolyCollection(spot_cells(np.c_[np.asarray(x, float), np.asarray(y, float)], box),
                        edgecolors="face", linewidths=0.3, rasterized=True)
    if colors is not None:
        pc.set_facecolor(list(colors))
    else:
        pc.set_array(np.asarray(values, float)); pc.set_cmap(cmap); pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.autoscale_view(); ax.set_aspect("equal"); ax.set_axis_off()
    return pc


def spotmap_masked(ax, x, y, values, vmin, vmax, cmap=MAPCMAP, missing="#E3E3E3"):
    """spotmap over every spot in one tiling, so that neighbouring tiles meet; spots without a value (NaN) are drawn
    in light grey instead of being left out."""
    v = np.asarray(values, float); cm = plt.get_cmap(cmap); norm = Normalize(vmin, vmax)
    return spotmap(ax, x, y, colors=[cm(norm(t)) if np.isfinite(t) else missing for t in v])


def map_key(fig, rect, orientation="vertical", label="scaled per map"):
    """Low-to-high key for maps that are each scaled to their own 1st-99th percentile."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), MAPCMAP), cax=cax, orientation=orientation)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.8, length=0, pad=1.5)
    cb.set_label(label, fontsize=5.8, labelpad=2)
    return cb


def J(path):
    return json.load(open(path))


def within(name="within_desium"):
    """Per-section scores and per-ion arrays keyed (method, section); m/z per section under ("mz", section)."""
    js = J(res(f"{name}.json")); Z = np.load(res(f"{name}_per_ion.npz"))
    per = {}
    for k in Z.files:
        m, s = k.rsplit("__", 1)
        per[(m, s)] = Z[k]
    return js, per


def ann_table():
    """Putative exact-mass candidates of the DESIUM m/z axis (reproduce/annotate_ions.py, negative-mode adducts).
    neg: the channel has at least one candidate; cls: its chemical class when all candidates share one (else NaN)."""
    a = pd.read_csv(res("desium_ion_annotation.csv"))
    a["neg"] = a.n_candidates > 0
    a.loc[~a.neg, "annotation"] = np.nan
    a["cls"] = a["class"].where(a.neg & (a["class"] != "ambiguous"))
    return a


def ion_label(row, mz=True):
    """'175.023 ascorbate [M−H]−' style label for a putative exact-mass candidate (nearest candidate first);
    † marks a channel with more than one candidate."""
    name = pretty(str(row.annotation).split(";")[0].strip())
    add = str(row.adduct).replace("-", "−")
    s = f"{name} {add}{'†' if bool(row.ambiguous) else ''}"
    return f"{row.mz:.3f} {s}" if mz else s


def ion_title(mz, ann=None):
    """Two-line map title for the channel nearest to mz: 'putative <name> <adduct>' over 'm/z <value>'."""
    a = ann_table() if ann is None else ann
    row = a.iloc[int(np.argmin(np.abs(a.mz.to_numpy(float) - float(mz))))]
    if abs(row.mz - float(mz)) > 0.01 or not bool(row.neg):
        return f"m/z {float(mz):.3f}"
    return f"putative {ion_label(row, mz=False)}\nm/z {row.mz:.3f}"


def per_ion_mean(per, method, sections, mz_ref):
    """Mean, s.d. and count over sections of a method's per-ion Spearman (NaN where never detected)."""
    import warnings
    M = np.vstack([per[(method, s)] for s in sections if (method, s) in per])
    with warnings.catch_warnings():                      # channels never detected stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(M, axis=0), np.nanstd(M, axis=0), np.sum(np.isfinite(M), axis=0)


def loso_medians(name):
    return {s: v["median"] for s, v in J(res(f"{name}.json"))["per_section"].items()}


def wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    return float(wilcoxon(d, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
