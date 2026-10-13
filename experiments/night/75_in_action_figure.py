"""Figure: SpOTr in action on a real section (PEtracer M3-4, seed-0 split; 2026-10-13).

Four panels at the width of figures/overview.pdf (911 pt), so the two sit together in the paper:
  a  the tree of the dissociated cells, coloured by major lineage (the tree cut at 32 days before sampling);
  b  the spatial slide as SpOTr sees it: spot expression (first principal component), no lineage, and the 20% of
     spots whose clade counts are the anchor;
  c  each spot's dominant lineage, from the in-situ lineage readout (ground truth);
  d  the same for Tangram, the expression-only mapper (73_tangram.py settings: cells mode, density term on);
  e  each spot's dominant lineage among the cells SpOTr places there (learned per-clade model, 70_spotr.py).
Lineages, not the 18 clades, are coloured because a map can only carry about three colours that stay
distinguishable for colour-blind readers (dataviz all-pairs check); the eight small lineages are grey.
Agreement is scored on the spots outside the anchor and also reported at the level of the 18 clades.
Usage: python 75_in_action_figure.py [out.pdf] [--png]   (default: SpOTr-overleaf/figures/petracer_in_action.pdf;
       --png also writes a 200-dpi preview; the numbers go to spotr_out/in_action_numbers.json)
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
POSTER = "--poster" in sys.argv                              # --poster: no slide, no numbers, type x2, 39.4 cm wide
SLIDE = "--no-slide" not in sys.argv and not POSTER          # --no-slide: drop the spatial-slide panel
FS = 2.0 if POSTER else 1.0
fs = lambda x: x * FS
OUT = Path(ARGS[0]) if ARGS else HERE.parents[2] / "SpOTr-overleaf" / ("poster" if POSTER else "figures") / (
    "petracer_in_action_poster.pdf" if POSTER else "petracer_in_action.pdf" if SLIDE else "petracer_in_action_noslide.pdf")
SEC, SEED, ARM, CUT = "M3-4", 0, "per-clade", 32.0


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


m70 = mod("m70", "70_spotr.py"); m60 = mod("m60", "60_anchor_learn.py")
X = m70.data_real(SEC, SEED)
z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
o = m60.obs_clade()
names = sorted(set(o.loc[np.asarray(z["cell_names"])]) | set(o.loc[np.asarray(z["spot_cell_names"])]))
f_T = HERE / "spotr_out" / f"T_{SEC}_s{SEED}_{ARM}.npy"
if not f_T.exists():                                       # the learned coupling; 70_spotr.py saves it
    m70.job_real((SEC, SEED, ARM))
T = np.load(f_T).astype(float)
import ot
n, m = T.shape
Texp = ot.emd(np.ones(n) / n, X["q"], X["M"])              # expression-only exact OT, for the comparison numbers
m73 = mod("m73", "73_tangram.py")                          # Tangram on the same cells and spots, same seed
n_all = len(z["C_tree"]); use = np.random.default_rng(SEED).permutation(n_all)[n_all // 2:]
f_tg = HERE / "spotr_out" / f"T_{SEC}_s{SEED}_tangram.npy"   # saved: Tangram is not bit-reproducible on the GPU,
if f_tg.exists():                                          # so every version of the figure reads one run
    Ttg = np.load(f_tg).astype(float)
else:
    Ttg = m73.tangram(np.asarray(z["X_sc"], float)[use], np.asarray(z["X_st"], float), X["q"], SEED)
    np.save(f_tg, Ttg.astype(np.float32))

# ---------------------------------------------------------------- lineages: the tree cut at CUT days
Z = linkage(squareform(X["AGE"], checks=False), "average")  # AGE is ultrametric (days), so this is the tree itself
grp = fcluster(Z, CUT, "distance") - 1
order = np.argsort(-np.bincount(grp))
top3 = order[:3]
lin = np.full(n, 3)                                         # 0, 1, 2 = the three largest lineages; 3 = the rest
for r, g in enumerate(top3):
    lin[grp == g] = r
clade_lin = np.array([np.bincount(lin[X["cl"] == c], minlength=4).argmax() if (X["cl"] == c).any() else 3
                      for c in range(X["K"])])
assert all(len(np.unique(lin[X["cl"] == c])) <= 1 for c in range(X["K"])), "a clade straddles two lineages"
lin_clades = [[names[c] for c in range(X["K"]) if clade_lin[c] == r and (X["cl"] == c).any()] for r in range(4)]

A = np.eye(4)[clade_lin]                                    # clade -> lineage, applied to the spot-side truth
has = X["Nspot"].sum(1) > 0
held = np.setdiff1d(np.where(has)[0], X["J"])
true_lin = (X["Nspot"] @ A).argmax(1)
inf_lin = (T.T @ np.eye(4)[lin]).argmax(1)
exp_lin = (Texp.T @ np.eye(4)[lin]).argmax(1)
tg_lin = (Ttg.T @ np.eye(4)[lin]).argmax(1)
true_cl = X["Nspot"].argmax(1); Y = np.eye(X["K"])[X["cl"]]
inf_cl, exp_cl, tg_cl = (T.T @ Y).argmax(1), (Texp.T @ Y).argmax(1), (Ttg.T @ Y).argmax(1)
acc = lambda a, b: float(np.mean(a[held] == b[held]))
NUM = dict(section=SEC, seed=SEED, arm=ARM, cells=int(n), spots=int(m), anchor_spots=int(len(X["J"])),
           held_out_spots=int(len(held)), lineage_cut_days=CUT, lineage_clades=lin_clades,
           lineage_cells=np.bincount(lin, minlength=4).tolist(),
           lineage_agree=dict(spotr=acc(inf_lin, true_lin), tangram=acc(tg_lin, true_lin), expression=acc(exp_lin, true_lin),
                              largest=acc(np.full(m, np.bincount(true_lin[has], minlength=4).argmax()), true_lin)),
           clade_agree=dict(spotr=acc(inf_cl, true_cl), tangram=acc(tg_cl, true_cl), expression=acc(exp_cl, true_cl),
                            largest=acc(np.full(m, np.bincount(X["cl"], minlength=X["K"]).argmax()), true_cl)),
           placement_error_um=dict(spotr=m70.err(T, X), tangram=m70.err(Ttg, X), expression=m70.err(Texp, X),
                                   fgw=min([r for r in json.load(open(HERE / "spotr_out" / "real.json"))
                                            if r["arm"] == "path-fgw" and r["sec"] == SEC and r["seed"] == SEED][0]["path"],
                                           key=lambda q: q["nll_val"])["err"]))
print(json.dumps(NUM, indent=1))

# ---------------------------------------------------------------- style
plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "cm", "pdf.fonttype": 42,
                     "axes.unicode_minus": False, "font.size": 9.5 * FS})
INK, INK2, BACKBONE = "#111111", "#4a4a4a", "#5a5a5a"
LIN_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"]     # documented slots 1-3 (pass all-pairs) + grey "other"
EMPTY = "#f1f0ec"
FIG_W = 911 / 72
FIG_H = 5.7
if not SLIDE:                                               # same scale, one panel fewer
    FIG_W = FIG_W - 2.12 - 0.24
W_MAP = 2.12                                                # inches per map panel
L_TREE, W_TREE, BOT, H = 0.36, 2.45, 1.16, 3.9
GAP = 0.24
if POSTER:                                                  # two template columns: 39.4 cm
    FIG_W, FIG_H, W_MAP, GAP = 39.38 / 2.54, 7.3, 2.95, 0.35
    L_TREE, W_TREE, BOT, H = 0.45, 4.2, 1.05, 5.35
fig = plt.figure(figsize=(FIG_W, FIG_H))
xy = X["spots"]; pitch = float(z["pitch"]); pad = pitch * 0.9


def frac(x0, w, y0=BOT, h=H):
    return [x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H]


def spot_map(ax, colours, title, letter, sub=None, marks=None):
    rects = [Rectangle((x - pitch * 0.46, y - pitch * 0.46), pitch * 0.92, pitch * 0.92) for x, y in xy]
    ax.add_collection(PatchCollection(rects, facecolors=colours, edgecolors="none"))
    if marks is not None:
        ax.scatter(xy[marks, 0], xy[marks, 1], s=7, color=INK, lw=0, zorder=4)
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box"); ax.set_anchor("N"); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=fs(11), fontweight="bold", pad=fs(18) if sub else fs(6))
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=fs(8.8), color=INK2, va="bottom", style="italic")
    if not POSTER:
        ax.text(-0.04, 1.0, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")


# a: the tree, branches coloured by lineage below the cut, backbone above it
ax = fig.add_axes(frac(L_TREE, W_TREE))
node_lin = {}
for i in range(n):
    node_lin[i] = lin[i]
for k, (a_, b_, h_, _) in enumerate(Z):
    la, lb = node_lin[int(a_)], node_lin[int(b_)]
    node_lin[n + k] = la if (la == lb and h_ <= CUT) else -1
col_of = lambda k: LIN_COL[node_lin[k]] if node_lin[k] >= 0 else BACKBONE
with plt.rc_context({"lines.linewidth": 0.45 * (1.4 if POSTER else 1.0)}):
    dg = dendrogram(Z, orientation="left", no_labels=True, ax=ax, link_color_func=col_of)
ax.set_xlim(36.5, -0.5)
ax.axvline(CUT, color=INK2, lw=0.9, ls=(0, (3, 2.5)), zorder=0)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.set_yticks([]); ax.set_xticks([35, 30, 20, 10, 0])
ax.set_xlabel(f"days before sampling\n(dashed: lineages cut at {CUT:.0f} days)", fontsize=fs(9.5), linespacing=1.3)
ax.set_title(f"Lineage tree, {n:,} cells" if POSTER else f"Tree of the {n:,} dissociated cells", loc="left",
             fontsize=fs(11), fontweight="bold", pad=fs(18))
ax.text(0, 1.012, "dissociated: no positions" if POSTER else "scRNA-seq with lineage barcodes, no positions",
        transform=ax.transAxes, fontsize=fs(8.8), color=INK2, va="bottom", style="italic")
if not POSTER:
    ax.text(-0.03, 1.0, "a", transform=ax.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")
leaves = np.array(dg["leaves"])
strip = fig.add_axes(frac(L_TREE + W_TREE + 0.04, 0.09))
strip.imshow(np.array([matplotlib.colors.to_rgb(LIN_COL[lin[i]]) for i in leaves])[:, None, :], aspect="auto",
             origin="lower", extent=(0, 1, 0, 10 * n), interpolation="nearest")
strip.set_ylim(ax.get_ylim()); strip.axis("off")

# b: the slide as SpOTr sees it
Xst = np.asarray(z["X_st"], float); Xc = Xst - Xst.mean(0)
pc1 = np.linalg.svd(Xc, full_matrices=False)[0][:, 0]
pc1 = pc1 if np.corrcoef(pc1, Xst.sum(1))[0, 1] >= 0 else -pc1
ramp = matplotlib.colors.LinearSegmentedColormap.from_list("slide", ["#eef3f7", "#9fb3c6", "#34495e"])
v = (pc1 - np.percentile(pc1, 2)) / (np.percentile(pc1, 98) - np.percentile(pc1, 2))
x_b = L_TREE + W_TREE + 0.5
if SLIDE:
    ax_b = fig.add_axes(frac(x_b, W_MAP))
    spot_map(ax_b, [ramp(np.clip(t, 0, 1)) for t in v], "Spatial slide", "b", sub="no lineage; dots: the 20% anchor spots",
             marks=X["J"])
    x0, y0 = xy[:, 0].min(), xy[:, 1].min() - pad * 0.55
    ax_b.plot([x0, x0 + 500], [y0, y0], color=INK, lw=2, solid_capstyle="butt")
    ax_b.text(x0 + 250, y0 - pitch * 0.35, "500 µm", ha="center", va="top", fontsize=8.5)
    cax = fig.add_axes(frac(x_b + W_MAP * 0.52, W_MAP * 0.42, y0=BOT - 0.36, h=0.09))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=ramp), cax=cax, orientation="horizontal", ticks=[])
    cb.outline.set_visible(False)
    cax.text(-0.03, 0.5, "low", transform=cax.transAxes, ha="right", va="center", fontsize=8, color=INK2)
    cax.text(1.03, 0.5, "high", transform=cax.transAxes, ha="left", va="center", fontsize=8, color=INK2)
    cax.text(0.5, -0.6, "expression PC1", transform=cax.transAxes, ha="center", va="top", fontsize=8, color=INK2)

# c, d: true and inferred lineages
x_c = x_b + W_MAP + GAP if SLIDE else x_b
LET = "cde" if SLIDE else "bcd"
ax_c = fig.add_axes(frac(x_c, W_MAP))
spot_map(ax_c, [LIN_COL[k] if h else EMPTY for k, h in zip(true_lin, has)], "True lineages", LET[0],
         sub="in-situ lineage readout")
if not SLIDE:                                               # the scale bar lived in the slide panel
    x0, y0 = xy[:, 0].min(), xy[:, 1].min() - pad * 0.55
    ax_c.plot([x0, x0 + 500], [y0, y0], color=INK, lw=2, solid_capstyle="butt", clip_on=False)
    ax_c.text(x0 + 250, y0 - pitch * 0.35, "500 µm", ha="center", va="top", fontsize=fs(8.5))
ax_d = fig.add_axes(frac(x_c + W_MAP + GAP, W_MAP))
spot_map(ax_d, [LIN_COL[k] if h else EMPTY for k, h in zip(tg_lin, has)], "Tangram", LET[1],
         sub="expression only")
ax_e = fig.add_axes(frac(x_c + 2 * (W_MAP + GAP), W_MAP))
spot_map(ax_e, [LIN_COL[k] if h else EMPTY for k, h in zip(inf_lin, has)], "SpOTr", LET[2],
         sub="expression + tree + anchor")

# legend and the numbers, under c and d
la = NUM["lineage_agree"]; ca = NUM["clade_agree"]
handles = [Rectangle((0, 0), 1, 1, color=c) for c in LIN_COL]
labels = [f"clades {', '.join(sorted(cs, key=int))}" for cs in lin_clades[:3]] + [f"{len(set(grp)) - 3} smaller lineages"]
fig.legend(handles, labels, loc="lower left", bbox_to_anchor=((x_c - 0.05) / FIG_W, (0.05 if POSTER else 0.68) / FIG_H),
           ncol=4, frameon=False, fontsize=fs(9), handlelength=1.1, handletextpad=0.45, columnspacing=1.3)
pe = NUM["placement_error_um"]
if not POSTER:
  fig.text((x_b if SLIDE else x_c - 0.05) / FIG_W, 0.07 / FIG_H,
           f"{len(held)} spots outside the anchor, dominant lineage right: SpOTr {100 * la['spotr']:.0f}%, Tangram "
         f"{100 * la['tangram']:.0f}%, expression-only OT {100 * la['expression']:.0f}%, largest lineage {100 * la['largest']:.0f}%.\n"
         f"Dominant clade of the 18 right: {100 * ca['spotr']:.0f}%, {100 * ca['tangram']:.0f}%, {100 * ca['expression']:.0f}%, "
         f"{100 * ca['largest']:.0f}%.\n"
         f"Median cell placement error: SpOTr {pe['spotr']:.0f} µm, FGW {pe['fgw']:.0f} µm, Tangram {pe['tangram']:.0f} µm, "
         f"expression-only OT {pe['expression']:.0f} µm.",
         fontsize=8.5, color=INK2, ha="left", va="bottom", style="italic", linespacing=1.4)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
if "--png" in sys.argv:
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
json.dump(NUM, open(HERE / "spotr_out" / "in_action_numbers.json", "w"), indent=1)
print("wrote", OUT)
