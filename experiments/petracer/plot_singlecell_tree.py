"""The single-cell phylogeny, and the two distance matrices ordered by it."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, cophenet
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
SEC = "M3-4"
p = np.load(f"petracer_{SEC}.npz", allow_pickle=True)
CT = np.asarray(p["C_tree"], float)
xy = np.asarray(p["xy_sc"], float)
cc = np.asarray(p["cell_clade"]).astype(str)
n = len(cc)

# clade colours, ordered by the clade-level phylogeny (same mapping as the spatial plots)
dd = np.load("petracer_xy.npz", allow_pickle=True)
Dcl, clades = np.asarray(dd["D"], float), dd["clades"].astype(str)
ordc = leaves_list(linkage(squareform(Dcl, checks=False), "average", optimal_ordering=True))
PHYLO = [clades[i] for i in ordc]
cmap = plt.get_cmap("tab20")
COL = {c: cmap(i % 20) for i, c in enumerate(PHYLO)}

# the single-cell tree: linkage reproduces C_tree exactly (cophenetic r = 1.0000)
Z = linkage(squareform(CT, checks=False), "average")
coph = pearsonr(cophenet(Z), squareform(CT, checks=False))[0]
print(f"cophenetic r of the linkage vs the measured patristic matrix: {coph:.4f}")

# colour each branch by clade when its whole subtree is one clade, else grey
desc = {i: {cc[i]} for i in range(n)}
lcol = {}
for i, (x, y, h, _) in enumerate(Z):
    x, y = int(x), int(y)
    s = desc[x] | desc[y]
    desc[n + i] = s
    lcol[n + i] = matplotlib.colors.to_hex(COL[next(iter(s))]) if len(s) == 1 else GRID

fig = plt.figure(figsize=(15.0, 9.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.05, 1.75], hspace=0.16, wspace=0.10,
                      top=0.905, bottom=0.045, left=0.058, right=0.975)

# --- the tree ---
ax = fig.add_subplot(gs[0, :])
dn = dendrogram(Z, ax=ax, no_labels=True, color_threshold=None,
                link_color_func=lambda k: lcol.get(k, GRID))
order = dn["leaves"]
ax.set_ylabel("patristic distance from the root", fontsize=10)
ax.set_xticks([])
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title(f"Single-cell phylogeny, {n} cells of section {SEC}   "
             f"(branch coloured by clade where the whole subtree is one clade, grey above)",
             fontsize=11.5, color=INK, pad=10)

# --- clade strip under the leaves ---
axs = fig.add_subplot(gs[1, :])
strip = np.array([matplotlib.colors.to_rgb(COL[cc[i]]) for i in order])[None, :, :]
axs.imshow(strip, aspect="auto", interpolation="nearest")
axs.set_xticks([]); axs.set_yticks([])
axs.set_ylabel("clade", rotation=0, ha="right", va="center", fontsize=9, color=INK2, labelpad=8)
for sp in axs.spines.values(): sp.set_visible(False)

# --- the two matrices, with clade colour strips on both edges ---
from matplotlib.gridspec import GridSpecFromSubplotSpec
CTo = CT[np.ix_(order, order)]
CSo = squareform(pdist(xy))[np.ix_(order, order)]
strip_rgb = np.array([matplotlib.colors.to_rgb(COL[cc[i]]) for i in order])
lab_ord = cc[order]

# contiguous clade blocks along the leaf order, for direct labels
blocks = []
st = 0
for i in range(1, n + 1):
    if i == n or lab_ord[i] != lab_ord[st]:
        blocks.append((lab_ord[st], st, i)); st = i
    
for k, (M, lab) in enumerate([(CTo, "tree (patristic) distance"),
                              (CSo, "spatial distance (μm)")]):
    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[2, k],
                                  height_ratios=[0.035, 1], width_ratios=[0.035, 1, 0.045],
                                  hspace=0.012, wspace=0.012)
    axt = fig.add_subplot(sub[0, 1])
    axt.imshow(strip_rgb[None, :, :], aspect="auto", interpolation="nearest")
    for c, s0, s1 in blocks:
        if s1 - s0 >= n * 0.035:
            axt.text((s0 + s1) / 2, 0, c, ha="center", va="center", fontsize=7.5,
                     color="white", fontweight="bold",
                     path_effects=[pe.withStroke(linewidth=2.0, foreground=INK)])
    axt.set_xticks([]); axt.set_yticks([])
    for sp in axt.spines.values(): sp.set_visible(False)
    axt.set_title(f"cell × cell {lab}", fontsize=10.5, color=INK, pad=7)

    axl = fig.add_subplot(sub[1, 0])
    axl.imshow(strip_rgb[:, None, :], aspect="auto", interpolation="nearest")
    axl.set_xticks([]); axl.set_yticks([])
    for sp in axl.spines.values(): sp.set_visible(False)

    axm = fig.add_subplot(sub[1, 1])
    im = axm.imshow(M, cmap="Blues", interpolation="nearest", aspect="auto")
    for _, _, s1 in blocks[:-1]:
        axm.axhline(s1, color="white", lw=0.4); axm.axvline(s1, color="white", lw=0.4)
    axm.set_xticks([]); axm.set_yticks([])

    axc = fig.add_subplot(sub[1, 2])
    cb = fig.colorbar(im, cax=axc); cb.ax.tick_params(labelsize=8)

fig.suptitle(f"PEtracer {SEC}: the single-cell tree, and the two metrics it is compared against",
             fontsize=13.5, y=0.972, color=INK)
fig.text(0.5, 0.028, "Both matrices use the tree's leaf order, so columns line up with the "
         "dendrogram and the clade strip above.", ha="center", fontsize=9.5, color=INK2)
fig.savefig("petracer_singlecell_tree.png", dpi=150, facecolor="white")
print("wrote petracer_singlecell_tree.png")
