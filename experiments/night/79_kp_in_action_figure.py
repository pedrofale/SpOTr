"""Figure: SpOTr in action on KP-Tracer spatial data (KP-Spatial Slide-seq puck M18_13, seed-0 split; 2026-10-13).

The KP equivalent of figures/petracer_in_action.pdf (75_in_action_figure.py), from the arrays and couplings that
78_kp_spotr.py writes:
  a  the tree of the dissociated cells, coloured by lineage (the tree's first greedy splits, 44_kpspatial_prep.py);
  b  the spatial slide as SpOTr sees it: spot expression (first principal component) and the 20% anchor spots;
  c  each spot's dominant lineage among its own beads (ground truth);
  d  Tangram (expression only);  e  SpOTr (learned per-clade model).
The trees are not time-calibrated, so the tree is drawn by the depth of each common ancestor in tree units.  The
section is wider than tall and is rotated a quarter turn for display, so the layout matches the PEtracer figure.
Usage: python 79_kp_in_action_figure.py [out.pdf] [--png] [--no-slide]
       (default: SpOTr-overleaf/figures/kp_in_action.pdf, or kp_in_action_noslide.pdf)
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SLIDE = "--no-slide" not in sys.argv
OUT = Path(ARGS[0]) if ARGS else HERE.parents[2] / "SpOTr-overleaf" / "figures" / (
    "kp_in_action.pdf" if SLIDE else "kp_in_action_noslide.pdf")
PUCK, SEED = "M18_13", 0

z = np.load(HERE.parent / "kpspatial" / f"kpspotr_{PUCK}_s{SEED}.npz", allow_pickle=True)
J = json.load(open(HERE / "spotr_out" / f"kp_{PUCK}_s{SEED}.json"))
T = {"tangram": np.load(HERE / "spotr_out" / f"T_KP-{PUCK}_s{SEED}_tangram.npy").astype(float),
     "spotr": np.load(HERE / "spotr_out" / f"T_KP-{PUCK}_s{SEED}_per-clade.npy").astype(float)}
use, lin3, clade_lin = z["use"], z["lin3"], z["clade_lin"]
n = len(use); Nspot = z["Nspot"]; m = len(Nspot)
lin = lin3[use]
has = Nspot.sum(1) > 0
true_lin = (Nspot @ np.eye(4)[clade_lin]).argmax(1)
inf = {k: (TT.T @ np.eye(4)[lin]).argmax(1) for k, TT in T.items()}
R = J["results"]
xy = np.c_[z["spots"][:, 1], -z["spots"][:, 0]]                 # a quarter turn, for a tall panel
pitch = float(z["pitch"]); pad = pitch * 0.9

# ---------------------------------------------------------------- style (as 75_in_action_figure.py)
plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "cm", "pdf.fonttype": 42,
                     "axes.unicode_minus": False, "font.size": 9.5})
INK, INK2, BACKBONE = "#111111", "#4a4a4a", "#5a5a5a"
LIN_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"]
EMPTY = "#f1f0ec"
FIG_W, FIG_H = 911 / 72, 5.7
W_MAP = 2.12
if not SLIDE:
    FIG_W = FIG_W - W_MAP - 0.24
fig = plt.figure(figsize=(FIG_W, FIG_H))
L_TREE, W_TREE, BOT, H = 0.36, 2.45, 1.16, 3.9


def frac(x0, w, y0=BOT, h=H):
    return [x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H]


def spot_map(ax, colours, title, letter, sub=None, marks=None):
    rects = [Rectangle((x - pitch * 0.46, y - pitch * 0.46), pitch * 0.92, pitch * 0.92) for x, y in xy]
    ax.add_collection(PatchCollection(rects, facecolors=colours, edgecolors="none"))
    if marks is not None:
        ax.scatter(xy[marks, 0], xy[marks, 1], s=6, color=INK, lw=0, zorder=4)
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box"); ax.set_anchor("N"); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=18 if sub else 6)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8.8, color=INK2, va="bottom", style="italic")
    ax.text(-0.04, 1.0, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")


def scale_bar(ax):
    x0, y0 = xy[:, 0].min(), xy[:, 1].min() - pitch * 1.3       # below the lowest row of spots
    ax.plot([x0, x0 + 500], [y0, y0], color=INK, lw=2, solid_capstyle="butt", clip_on=False)
    ax.text(x0 + 250, y0 - pitch * 0.35, "500 µm", ha="center", va="top", fontsize=8.5)


# a: the tree, by depth of each common ancestor (the trees are not time-calibrated)
Lt = z["Lt"].astype(float)[np.ix_(use, use)]
top = float(Lt.max())
U = top - Lt; np.fill_diagonal(U, 0.0)                         # an ultrametric: same topology, leaves aligned
Z = linkage(squareform(U, checks=False), "average")
node_lin = {i: lin[i] for i in range(n)}
for k, (a_, b_, h_, _) in enumerate(Z):
    la, lb = node_lin[int(a_)], node_lin[int(b_)]
    node_lin[n + k] = la if la == lb else -1
ax = fig.add_axes(frac(L_TREE, W_TREE))
with plt.rc_context({"lines.linewidth": 0.45}):
    dg = dendrogram(Z, orientation="left", no_labels=True, ax=ax,
                    link_color_func=lambda k: LIN_COL[node_lin[k]] if node_lin[k] >= 0 else BACKBONE)
ax.set_xlim(top + 4, -2)
ticks = [d for d in [0, 30, 60, 90, 120] if d <= top]
ax.set_xticks([top - d for d in ticks]); ax.set_xticklabels([str(d) for d in ticks])
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.set_yticks([])
ax.set_xlabel("depth of the common ancestor\n(tree units; not calibrated in time)", fontsize=9.5, linespacing=1.3)
ax.set_title(f"Tree of the {n:,} dissociated cells", loc="left", fontsize=11, fontweight="bold", pad=18)
ax.text(0, 1.012, f"KP-Tracer, Slide-seq puck {PUCK.replace('_', '-')}; no positions", transform=ax.transAxes,
        fontsize=8.8, color=INK2, va="bottom", style="italic")
ax.text(-0.03, 1.0, "a", transform=ax.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")
leaves = np.array(dg["leaves"])
strip = fig.add_axes(frac(L_TREE + W_TREE + 0.04, 0.09))
strip.imshow(np.array([matplotlib.colors.to_rgb(LIN_COL[lin[i]]) for i in leaves])[:, None, :], aspect="auto",
             origin="lower", extent=(0, 1, 0, 10 * n), interpolation="nearest")
strip.set_ylim(ax.get_ylim()); strip.axis("off")

# b: the slide as SpOTr sees it
x_b = L_TREE + W_TREE + 0.5
if SLIDE:
    Xst = z["Xst_tg"].astype(float); Xc = Xst - Xst.mean(0)
    pc1 = np.linalg.svd(Xc, full_matrices=False)[0][:, 0]
    pc1 = pc1 if np.corrcoef(pc1, Xst.sum(1))[0, 1] >= 0 else -pc1
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("slide", ["#eef3f7", "#9fb3c6", "#34495e"])
    v = (pc1 - np.percentile(pc1, 2)) / (np.percentile(pc1, 98) - np.percentile(pc1, 2))
    ax_b = fig.add_axes(frac(x_b, W_MAP))
    spot_map(ax_b, [ramp(np.clip(t, 0, 1)) for t in v], "Spatial slide", "b", sub="no lineage; dots: the 20% anchor spots",
             marks=z["J"])
    scale_bar(ax_b)
    cax = fig.add_axes(frac(x_b + W_MAP * 0.52, W_MAP * 0.42, y0=BOT - 0.36, h=0.09))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=ramp), cax=cax, orientation="horizontal", ticks=[])
    cb.outline.set_visible(False)
    cax.text(-0.03, 0.5, "low", transform=cax.transAxes, ha="right", va="center", fontsize=8, color=INK2)
    cax.text(1.03, 0.5, "high", transform=cax.transAxes, ha="left", va="center", fontsize=8, color=INK2)
    cax.text(0.5, -0.6, "expression PC1", transform=cax.transAxes, ha="center", va="top", fontsize=8, color=INK2)

# c, d, e: true and inferred lineages
GAP = 0.24
x_c = x_b + W_MAP + GAP if SLIDE else x_b
LET = "cde" if SLIDE else "bcd"
ax_c = fig.add_axes(frac(x_c, W_MAP))
spot_map(ax_c, [LIN_COL[k] if h else EMPTY for k, h in zip(true_lin, has)], "True lineages", LET[0],
         sub="lineage barcodes read in situ")
if not SLIDE:
    scale_bar(ax_c)
ax_d = fig.add_axes(frac(x_c + W_MAP + GAP, W_MAP))
spot_map(ax_d, [LIN_COL[k] if h else EMPTY for k, h in zip(inf["tangram"], has)], "Tangram", LET[1], sub="expression only")
ax_e = fig.add_axes(frac(x_c + 2 * (W_MAP + GAP), W_MAP))
spot_map(ax_e, [LIN_COL[k] if h else EMPTY for k, h in zip(inf["spotr"], has)], "SpOTr", LET[2],
         sub="expression + tree + anchor")

# legend and the numbers
cl_counts = [int(np.sum(clade_lin == r)) for r in range(4)]
labels = [f"lineage {r + 1} ({cl_counts[r]} clade{'s' if cl_counts[r] != 1 else ''})" for r in range(3)] + ["small lineage"]
fig.legend([Rectangle((0, 0), 1, 1, color=c) for c in LIN_COL], labels, loc="lower left",
           bbox_to_anchor=((x_c - 0.05) / FIG_W, 0.68 / FIG_H), ncol=4, frameon=False, fontsize=9, handlelength=1.1,
           handletextpad=0.45, columnspacing=1.3)
pc = lambda v: f"{100 * v:.0f}%"
fig.text((x_b if SLIDE else x_c - 0.05) / FIG_W, 0.07 / FIG_H,
         f"{J['held_out']} spots outside the anchor, dominant lineage right: SpOTr {pc(R['spotr']['lineage'])}, Tangram "
         f"{pc(R['tangram']['lineage'])}, expression-only OT {pc(R['expression']['lineage'])}, largest lineage {pc(R['largest']['lineage'])}.\n"
         f"Dominant clade of the {J['K']} right: {pc(R['spotr']['clade'])}, {pc(R['tangram']['clade'])}, {pc(R['expression']['clade'])}, "
         f"{pc(R['largest']['clade'])}.\n"
         f"Median cell placement error: SpOTr {R['spotr']['err']:.0f} µm, FGW {R['fgw']['err']:.0f} µm, Tangram "
         f"{R['tangram']['err']:.0f} µm, expression-only OT {R['expression']['err']:.0f} µm.",
         fontsize=8.5, color=INK2, ha="left", va="bottom", style="italic", linespacing=1.4)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
if "--png" in sys.argv:
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("wrote", OUT)
