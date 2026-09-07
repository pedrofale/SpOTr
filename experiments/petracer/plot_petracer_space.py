"""The PEtracer spatial data: cells coloured by clade.

18 clades is well beyond what a categorical palette can separate, so identity is carried by
DIRECT LABELS at each clade's centroid, with colour as a secondary grouping aid, and an
unambiguous small-multiples panel where each clade is shown on its own.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
d = np.load("petracer_xy.npz", allow_pickle=True)
xy, lab, clades, D = d["xy"], d["lab"].astype(str), d["clades"].astype(str), d["D"]
scale = np.ptp(xy, axis=0).max()
sec = DBSCAN(eps=0.02 * scale, min_samples=20).fit_predict(xy)
NAME = {0: "M3-1", 1: "M3-2", 2: "M3-4"}
SECS = [0, 1, 2]

# order clades by the phylogeny so neighbouring colours are related clades
order = leaves_list(linkage(squareform(D, checks=False), "average", optimal_ordering=True))
ORD = [clades[i] for i in order]
cmap = plt.get_cmap("tab20")
COL = {c: cmap(i % 20) for i, c in enumerate(ORD)}

def frame(ax, P):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(GRID)
    m = 0.04 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    ax.set_xlim(P[:, 0].min() - m, P[:, 0].max() + m)
    ax.set_ylim(P[:, 1].min() - m, P[:, 1].max() + m)

# ---------- figure 1: overview ----------
fig = plt.figure(figsize=(15.2, 5.9))
gs = fig.add_gridspec(1, 3, wspace=0.06, top=0.86, bottom=0.04, left=0.02, right=0.98)
for k, s in enumerate(SECS):
    m = sec == s; P, L = xy[m], lab[m]
    ax = fig.add_subplot(gs[0, k])
    ax.scatter(P[:, 0], P[:, 1], s=7, c=[COL[c] for c in L], linewidth=0)
    for c in ORD:
        q = L == c
        if q.sum() < 20: continue
        cx, cy = np.median(P[q, 0]), np.median(P[q, 1])
        ax.text(cx, cy, c, fontsize=11, fontweight="bold", ha="center", va="center",
                color="white", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.8, foreground=INK)])
    frame(ax, P)
    ax.set_title(f"{NAME[s]}   {m.sum()} cells   {len(set(L))} clades",
                 fontsize=11.5, color=INK, pad=8)
fig.suptitle("PEtracer M3: cells in space, coloured by clade (number = clade id, at its centroid)",
             fontsize=13.5, y=0.955, color=INK)
fig.savefig("petracer_space_overview.png", dpi=155, facecolor="white")
print("wrote petracer_space_overview.png")

# ---------- figure 2: small multiples, split in two so panels are legible ----------
for half, cols in enumerate([ORD[:9], ORD[9:]]):
    fig, axes = plt.subplots(3, 9, figsize=(16.4, 6.4))
    for r, s_ in enumerate(SECS):
        m = sec == s_; P, L = xy[m], lab[m]
        for cix, c in enumerate(cols):
            ax = axes[r, cix]
            ax.scatter(P[:, 0], P[:, 1], s=2.2, c=GRID, linewidth=0)
            q = L == c
            if q.sum():
                ax.scatter(P[q, 0], P[q, 1], s=5.5, c=[COL[c]], linewidth=0)
            frame(ax, P)
            if r == 0:
                ax.set_title(f"clade {c}", fontsize=11, color=INK, pad=5)
            if cix == 0:
                ax.set_ylabel(NAME[s_], fontsize=12, color=INK, labelpad=8)
            ax.text(0.04, 0.03, f"n={int(q.sum())}", transform=ax.transAxes, fontsize=8.5,
                    color=INK2, ha="left", va="bottom")
    fig.suptitle(f"Each clade on its own  ({half*9+1}\u2013{half*9+len(cols)} of 18, ordered by "
                 "position in the phylogeny).  Grey = all cells in that section.",
                 fontsize=13, y=0.985, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"petracer_space_perclade_{half+1}.png", dpi=150, facecolor="white")
    print(f"wrote petracer_space_perclade_{half+1}.png")
