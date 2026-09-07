"""The expression distance SpOTr actually uses, next to the tree and spatial metrics.

prepare_ot_inputs does  C_feature = ot.dist(X_sc, X_st) / max  -- POT's ot.dist defaults to
SQUARED EUCLIDEAN. The object the model optimises against is cell x SPOT. The cell x cell
version below uses the identical metric, so it is comparable with C_tree and C_space.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.stats import pearsonr, spearmanr

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
SEC = "M3-4"
p = np.load(f"petracer_{SEC}.npz", allow_pickle=True)
CT = np.asarray(p["C_tree"], float)
X_sc = np.asarray(p["X_sc"], float)
X_st = np.asarray(p["X_st"], float)
xy = np.asarray(p["xy_sc"], float)
cc = np.asarray(p["cell_clade"]).astype(str)
n = len(cc)

dd = np.load("petracer_xy.npz", allow_pickle=True)
Dcl, clades = np.asarray(dd["D"], float), dd["clades"].astype(str)
ordc = leaves_list(linkage(squareform(Dcl, checks=False), "average", optimal_ordering=True))
PHYLO = [clades[i] for i in ordc]
cmap = plt.get_cmap("tab20"); COL = {c: cmap(i % 20) for i, c in enumerate(PHYLO)}

Z = linkage(squareform(CT, checks=False), "average")
order = leaves_list(Z)

CE = squareform(pdist(X_sc, metric="sqeuclidean")); CE = CE / CE.max()   # same metric as ot.dist
CS = squareform(pdist(xy))
M = {"tree (patristic)": CT[np.ix_(order, order)],
     "spatial (μm)": CS[np.ix_(order, order)],
     "expression (sq. Euclidean, the SpOTr feature metric)": CE[np.ix_(order, order)]}

iu = np.triu_indices(n, 1)
same = cc[iu[0]] == cc[iu[1]]
print("cell-pair correlations (all pairs):")
for a, b, la, lb in [(CE, CT, "expression", "tree"), (CE, CS, "expression", "space"),
                     (CT, CS, "tree", "space")]:
    print(f"  {la:>10} vs {lb:<6} pearson {pearsonr(a[iu], b[iu])[0]:+.4f}  "
          f"spearman {spearmanr(a[iu], b[iu])[0]:+.4f}")
print(f"\nexpression distance: within-clade median {np.median(CE[iu][same]):.4f}, "
      f"between-clade median {np.median(CE[iu][~same]):.4f}, "
      f"ratio {np.median(CE[iu][~same])/np.median(CE[iu][same]):.3f}")
print(f"  variance explained by the same/different-clade binary: "
      f"R2 = {pearsonr(CE[iu], same.astype(float))[0]**2:.4f}   "
      f"(tree metric: R2 = {pearsonr(CT[iu], same.astype(float))[0]**2:.4f})")

strip = np.array([matplotlib.colors.to_rgb(COL[cc[i]]) for i in order])
lab_ord = cc[order]
blocks, st = [], 0
for i in range(1, n + 1):
    if i == n or lab_ord[i] != lab_ord[st]:
        blocks.append((lab_ord[st], st, i)); st = i

fig = plt.figure(figsize=(16.4, 6.2))
gs = fig.add_gridspec(1, 3, wspace=0.16, top=0.80, bottom=0.10, left=0.035, right=0.985)
for k, (lab, Mat) in enumerate(M.items()):
    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, k],
                                  height_ratios=[0.045, 1], width_ratios=[0.045, 1, 0.05],
                                  hspace=0.015, wspace=0.015)
    axt = fig.add_subplot(sub[0, 1])
    axt.imshow(strip[None, :, :], aspect="auto", interpolation="nearest")
    for c, s0, s1 in blocks:
        if s1 - s0 >= n * 0.05:
            axt.text((s0 + s1) / 2, 0, c, ha="center", va="center", fontsize=7.5,
                     color="white", fontweight="bold",
                     path_effects=[pe.withStroke(linewidth=2.0, foreground=INK)])
    axt.set_xticks([]); axt.set_yticks([])
    for sp in axt.spines.values(): sp.set_visible(False)
    axt.set_title(f"cell × cell {lab}", fontsize=10.5, color=INK, pad=7)

    axl = fig.add_subplot(sub[1, 0])
    axl.imshow(strip[:, None, :], aspect="auto", interpolation="nearest")
    axl.set_xticks([]); axl.set_yticks([])
    for sp in axl.spines.values(): sp.set_visible(False)

    axm = fig.add_subplot(sub[1, 1])
    lo, hi = np.percentile(Mat[np.triu_indices(n, 1)], [1, 99])
    im = axm.imshow(Mat, cmap="Blues", interpolation="nearest", aspect="auto",
                    vmin=lo, vmax=hi)
    for _, _, s1 in blocks[:-1]:
        axm.axhline(s1, color="white", lw=0.4); axm.axvline(s1, color="white", lw=0.4)
    axm.set_xticks([]); axm.set_yticks([])
    cb = fig.colorbar(im, cax=fig.add_subplot(sub[1, 2])); cb.ax.tick_params(labelsize=8)

fig.suptitle(f"PEtracer {SEC}: the three metrics SpOTr sees, all in the tree's leaf order",
             fontsize=13.5, y=0.925, color=INK)
fig.text(0.5, 0.035, "Clade strips and ordering are identical to the phylogeny figure. Expression uses the "
         "same squared-Euclidean metric prepare_ot_inputs applies. All three panels are colour-scaled to their "
         "own 1st\u201399th percentile, so each one's real variation is visible.",
         ha="center", fontsize=9.5, color=INK2)
fig.savefig("petracer_three_metrics.png", dpi=150, facecolor="white")
print("\nwrote petracer_three_metrics.png")

# the object SpOTr actually optimises against: cell x spot
CF = cdist(X_sc, X_st, metric="sqeuclidean"); CF = CF / CF.max()
fig2, ax = plt.subplots(figsize=(7.6, 5.4))
im = ax.imshow(CF[order, :], cmap="Blues", aspect="auto", interpolation="nearest")
ax.set_xlabel(f"{X_st.shape[0]} spots"); ax.set_ylabel(f"{n} cells, tree leaf order")
ax.set_title("What SpOTr actually optimises against:\nthe cell × spot expression cost C_feature",
             fontsize=11, color=INK)
fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
fig2.tight_layout()
fig2.savefig("petracer_cell_spot_cost.png", dpi=150, facecolor="white")
print("wrote petracer_cell_spot_cost.png")
