"""Three panels for one PEtracer split (M3-4, seed 0): the dissociated cells' tree coloured by clade; the spatial
spots coloured by their TRUE dominant clade (from the spot-side cells); and the spots coloured by the dominant
clade the METHOD assigns to them.  Method: law-based tree term, tree weight learned from the anchor
(20% of spots labelled; run M3-4_s0_c20_a-only of 61_anchor_fast.py), mapped with the exact solver (55_).
The 8 largest clades get the 8 palette colours; the 10 smaller ones are grey.
Usage: python 63_clone_maps.py <out_dir>
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("m60", HERE / "60_anchor_learn.py")
m60 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m60)

SEC, SEED, TAG = "M3-4", 0, "M3-4_s0_c20_a-only"
P = m60.prep(SEC, SEED)
z = np.load(HERE / "anchor_fast" / f"{TAG}.npz")
n, m = P["M"].shape
T = m60.m55.solve(P["M"], z["W"] * z["sc"], z["WPhi"] * z["sc"], P["D"], P["D"] ** 2,
                  np.ones(n) / n, P["q"], float(z["a"]))
Texp = m60.m55.solve(P["M"], np.zeros((n, n)), np.zeros((n, n)), P["D"], P["D"] ** 2, np.ones(n) / n, P["q"], 0.0)

K = P["K"]; Y = np.eye(K)[P["cl"]]
true_dom = P["Nspot"].argmax(1); has = P["Nspot"].sum(1) > 0
inf_dom = (T.T @ Y).argmax(1); exp_dom = (Texp.T @ Y).argmax(1)
sizes = np.bincount(P["cl"], minlength=K); top = np.argsort(-sizes)[:8]
names = sorted(set(m60.obs_clade().loc[np.asarray(np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)["cell_names"])]) |
               set(m60.obs_clade().loc[np.asarray(np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)["spot_cell_names"])]))
am = T.argmax(1)
stats = dict(
    spots=int(has.sum()), cells=int(n),
    agree_method=float((inf_dom[has] == true_dom[has]).mean()),
    agree_expression=float((exp_dom[has] == true_dom[has]).mean()),
    agree_largest_clade=float((true_dom[has] == np.argmax(sizes)).mean()),
    err_method=float(np.median(np.linalg.norm(P["xy_st"][am] - P["xy_true"], axis=1))),
    err_expression=float(np.median(np.linalg.norm(P["xy_st"][Texp.argmax(1)] - P["xy_true"], axis=1))),
    tree_weight=float(z["a"]), top8_share=float(sizes[top].sum() / sizes.sum()),
    top8_names=[names[k] for k in top])
json.dump(stats, open(OUT / "clone_map_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

pitch = float(np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)["pitch"])
THEMES = {
    "light": dict(surf="#fcfcfb", ink="#0b0b0b", ink2="#52514e", other="#c9c8c2", empty="#efeeea", line="#8a8984",
                  s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]),
    "dark": dict(surf="#1a1a19", ink="#f0efec", ink2="#c3c2b7", other="#55554f", empty="#262624", line="#8f8e87",
                 s=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]),
}

Z = linkage(squareform(2.0 * P["AGE"], checks=False), "average")
for th, t in THEMES.items():
    col = {k: t["s"][i] for i, k in enumerate(top)}
    c_of = lambda k: col.get(int(k), t["other"])
    plt.rcParams.update({"figure.facecolor": t["surf"], "axes.facecolor": t["surf"], "savefig.facecolor": t["surf"],
                         "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"],
                         "ytick.color": t["ink2"], "font.family": "DejaVu Sans", "font.size": 10.5})
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 0.06, 1.25, 1.25], wspace=0.06)
    ax_t = fig.add_subplot(gs[0, 0]); ax_s = fig.add_subplot(gs[0, 1])
    with plt.rc_context({"lines.linewidth": 0.35}):
        dg = dendrogram(Z, orientation="left", no_labels=True, ax=ax_t, color_threshold=0, above_threshold_color=t["line"])
    ax_t.set_xlim(72, 0); ax_t.set_xlabel("Divergence (patristic distance, days)")
    for sp in ["top", "right", "left"]:
        ax_t.spines[sp].set_visible(False)
    ax_t.spines["bottom"].set_color(t["ink2"]); ax_t.set_yticks([])
    ax_t.set_title(f"Tree of the {n} dissociated cells", loc="left", fontweight="bold", fontsize=11.5)
    leaves = np.array(dg["leaves"])
    strip = np.array([matplotlib.colors.to_rgb(c_of(P["cl"][i])) for i in leaves])[:, None, :]
    ax_s.imshow(strip, aspect="auto", origin="lower", extent=(0, 1, 0, 10 * len(leaves)))
    ax_s.set_ylim(ax_t.get_ylim()); ax_s.axis("off")

    xy = P["xy_st"]; pad = pitch * 0.8
    for gi, (lab, dom) in enumerate([("Spatial data, true clone labels", true_dom),
                                     ("Spatial data, clone labels inferred by SpOTr", inf_dom)]):
        ax = fig.add_subplot(gs[0, 2 + gi])
        rects = [Rectangle((x - pitch * 0.47, y - pitch * 0.47), pitch * 0.94, pitch * 0.94) for x, y in xy]
        fc = [c_of(d) if (gi == 1 or h) else t["empty"] for d, h in zip(dom, has)]
        ax.add_collection(PatchCollection(rects, facecolors=fc, edgecolors="none"))
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(lab, loc="left", fontweight="bold", fontsize=11.5)
        if gi == 0:
            x0, y0 = xy[:, 0].min(), xy[:, 1].min() - pad * 0.6
            ax.plot([x0, x0 + 500], [y0, y0], color=t["ink2"], lw=2); ax.text(x0 + 250, y0 - pitch * 0.5, "500 µm", ha="center", va="top", color=t["ink2"], fontsize=9)
    handles = [Rectangle((0, 0), 1, 1, color=t["s"][i]) for i in range(8)] + [Rectangle((0, 0), 1, 1, color=t["other"])]
    labels = [f"Clade {names[k]}" for k in top] + ["10 smaller clades"]
    fig.legend(handles, labels, loc="lower center", ncol=9, frameon=False, fontsize=9.5, bbox_to_anchor=(0.56, -0.02),
               handlelength=1.1, columnspacing=1.2)
    fig.savefig(OUT / f"clone_map_{th}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
