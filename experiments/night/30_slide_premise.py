import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm

OUT = "/home/pedro/projects/spotr/phylospatial_mapping/experiments/night/slide_premise.png"
d = np.load("../petracer/petracerFAIR_M3-4.npz", allow_pickle=True)
xy = d["xy_sc"].astype(np.float64)
cl = d["cell_clade"]
C = d["C_tree"].astype(np.float64)
n = xy.shape[0]

# ---- pair quantities (i<j) ----
iu = np.triu_indices(n, 1)
tree = C[iu]
dx = xy[:, 0][:, None] - xy[:, 0][None, :]
dy = xy[:, 1][:, None] - xy[:, 1][None, :]
sp = np.sqrt(dx * dx + dy * dy)[iu]
del dx, dy

same = (cl[:, None] == cl[None, :])[iu]
m_same = sp[same].mean()
m_diff = sp[~same].mean()
ratio = m_same / m_diff
m_all = sp.mean()

CEIL = C.max()
ceil_mask = tree >= CEIL
ceil_frac = ceil_mask.mean()

# 14 equal-width bins strictly below the ceiling + one ceiling point
edges = np.linspace(0.0, CEIL, 15)
centres, means, counts = [], [], []
for k in range(14):
    lo, hi = edges[k], edges[k + 1]
    m = (tree >= lo) & (tree < hi)
    if m.sum() > 0:
        centres.append(0.5 * (lo + hi)); means.append(sp[m].mean()); counts.append(int(m.sum()))
centres.append(CEIL); means.append(sp[ceil_mask].mean()); counts.append(int(ceil_mask.sum()))
centres = np.array(centres); means = np.array(means)

# ---- figure ----
fig = plt.figure(figsize=(13, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 8.5], wspace=0.16,
                      left=0.035, right=0.985, bottom=0.13, top=0.90)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

# ---- Panel A ----
clades = sorted(np.unique(cl), key=lambda s: int(s))
cmap = plt.get_cmap("tab20")
for i, c in enumerate(clades):
    m = cl == c
    axA.scatter(xy[m, 0], xy[m, 1], s=7, color=cmap(i % 20), linewidths=0, alpha=0.95)
axA.set_aspect("equal")
axA.set_xticks([]); axA.set_yticks([])
for s in axA.spines.values():
    s.set_visible(False)
axA.set_title("One tumour section: 2281 cells,\ncoloured by clade", fontsize=11.5, pad=6)

sizes = {c: int((cl == c).sum()) for c in clades}
cent = {c: np.median(xy[cl == c], 0) for c in clades}
top3 = []
for c in sorted(sizes, key=lambda c: -sizes[c]):
    if all(np.hypot(*(cent[c] - cent[k])) > 350 for k in top3):
        top3.append(c)
    if len(top3) == 3:
        break
for c in top3:
    cx, cy = cent[c]
    axA.text(cx, cy, c, fontsize=14, fontweight="bold", color="black",
             ha="center", va="center",
             path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])

axA.text(0.5, -0.035,
         "same-clade pairs sit at %.2fx the distance\nof different-clade pairs" % ratio,
         transform=axA.transAxes, ha="center", va="top", fontsize=10.5)
axA.text(0.02, 0.995, "A", transform=axA.transAxes, fontsize=15,
         fontweight="bold", ha="left", va="top")

# ---- Panel B ----
h = axB.hist2d(tree, sp, bins=[90, 90], cmap="Blues", norm=LogNorm())
cb = fig.colorbar(h[3], ax=axB, pad=0.012, fraction=0.045)
cb.set_label("cell pairs per bin", fontsize=10)
cb.ax.tick_params(labelsize=9)

axB.axhline(m_all, ls=":", lw=2, color="0.25")
axB.text(1.5, m_all + 22, "two random cells (%.0f um)" % m_all,
         fontsize=10, color="0.15", va="bottom")

axB.plot(centres, means, "-o", color="#8B1A1A", lw=3.2, ms=7,
         mec="white", mew=1.2, zorder=5)
axB.text(3.5, 540, "mean spatial distance\nper tree-distance bin",
         fontsize=10.5, color="#8B1A1A", ha="left", va="center")
axB.annotate("ceiling bin:\n%.0f%% of all pairs" % (100 * ceil_frac),
             xy=(CEIL, means[-1]), xytext=(CEIL - 13.0, means[-1] + 480),
             fontsize=10.5, color="#8B1A1A", ha="right", va="center",
             arrowprops=dict(arrowstyle="-", color="#8B1A1A", lw=1.4))

axB.set_xlabel("patristic distance (= 2 x days since common ancestor)", fontsize=11)
axB.set_ylabel("spatial distance (um)", fontsize=11)
axB.set_title("Spatial proximity decays with divergence time", fontsize=11.5, pad=8)
axB.tick_params(labelsize=10)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
axB.grid(alpha=0.25)
axB.set_axisbelow(False)
axB.set_xlim(0, CEIL + 1.2)
axB.text(0.012, 0.975, "B", transform=axB.transAxes, fontsize=15,
         fontweight="bold", ha="left", va="top")

fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")

print("PATH", OUT)
print("n_cells", n)
print("n_pairs", tree.size)
print("mean_same_clade_um", m_same)
print("mean_diff_clade_um", m_diff)
print("ratio", ratio)
print("mean_all_pairs_um", m_all)
print("ceiling_value", CEIL)
print("ceiling_frac", ceil_frac)
print("top3_clades", [(c, sizes[c]) for c in top3])
for cc, mm, nn in zip(centres, means, counts):
    print("BIN %.3f %.4f %d" % (cc, mm, nn))
