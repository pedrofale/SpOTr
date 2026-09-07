"""The simulated data in the same views used for PEtracer: spatial by clade, and the
cell x cell tree / space / expression distance heatmaps with clade colour annotations."""
import sys, numpy as np, ot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform, pdist
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim_varalpha_branches import simulate

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
RAMP = "Blues"
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.titlesize": 10.5})

d = simulate(0)
CT, xy_sc, br = d["C_tree"], d["xy_sc"], d["branch"]
n = d["n"]
K_PER = 9                                   # 9 pseudo-clades per branch -> 18, as in PEtracer

# pseudo-clades: cut each branch's tree into K_PER groups (PEtracer's clades come from the
# recorder; here we cut the simulated tree so the two figures are directly comparable)
clade = np.zeros(n, int)
for b in (0, 1):
    s = np.where(br == b)[0]
    sub = CT[np.ix_(s, s)]
    lab = fcluster(linkage(squareform(sub, checks=False), "average"), K_PER, "maxclust")
    clade[s] = lab - 1 + b * K_PER

# order cells: branch A clades (in tree order) then branch B clades
order = []
for b in (0, 1):
    ks = [k for k in range(b*K_PER, (b+1)*K_PER)]
    cen = np.array([CT[np.ix_(np.where(clade == k)[0], np.where(clade == k)[0])].mean()
                    if (clade == k).sum() > 1 else 0 for k in ks])
    for k in [ks[i] for i in np.argsort(cen)]:
        order += list(np.where(clade == k)[0])
order = np.array(order)

cmap = plt.get_cmap("tab20")
COL = np.array([cmap(k % 20) for k in range(2*K_PER)])

# ---------------- figure A: the simulated tissue ----------------
fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.7))
for k, (b, ttl) in enumerate([
        (0, "Branch A — tree built from position\n(planted TREE-INFORMATIVE)"),
        (1, "Branch B — tree built from noise\n(planted TREE-BLIND)")]):
    s = br == b
    ax[k].scatter(xy_sc[s, 0] + np.random.default_rng(k).normal(0, .012, s.sum()),
                  xy_sc[s, 1] + np.random.default_rng(k+9).normal(0, .012, s.sum()),
                  c=COL[clade[s]], s=13, edgecolors="none")
    ax[k].set_title(ttl); ax[k].set_aspect("equal")
    ax[k].set_xticks([]); ax[k].set_yticks([])
ax[2].scatter(xy_sc[:, 0] + np.random.default_rng(3).normal(0, .012, n),
              xy_sc[:, 1] + np.random.default_rng(4).normal(0, .012, n),
              c=np.where(br == 0, "#2a78d6", "#eb6834"), s=13, edgecolors="none")
ax[2].set_title("Both branches\n(blue = A, orange = B)")
ax[2].set_aspect("equal"); ax[2].set_xticks([]); ax[2].set_yticks([])
fig.suptitle("Simulated tissue, cells coloured by pseudo-clade (9 per branch, cut from the "
             "simulated tree)\nA's clades are spatially contiguous; B's are scattered — "
             "that is the planted effect", fontsize=12, y=0.99, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(HERE / "sim_space.png", dpi=155, facecolor="white")
print("wrote sim_space.png")

# ---------------- figure B: cell x cell distance heatmaps ----------------
CTo = CT[np.ix_(order, order)]
CSo = squareform(pdist(xy_sc[order]))
CEo = ot.dist(d["X_sc"][order], d["X_sc"][order]) ** 0.5
co = clade[order]
bounds = np.where(np.diff(co) != 0)[0] + 1
bbound = int((br[order] == 0).sum())

fig = plt.figure(figsize=(16.0, 5.8))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.95], height_ratios=[0.038, 1],
                      wspace=0.30, hspace=0.03, top=0.74, bottom=0.09, left=0.035, right=0.975)
for k, (M, lab) in enumerate([(CTo, "tree distance"), (CSo, "spatial distance"),
                              (CEo, "expression distance")]):
    strip = fig.add_subplot(gs[0, k])
    strip.imshow(COL[co][None, :, :], aspect="auto")
    strip.set_xticks([]); strip.set_yticks([])
    for sp in strip.spines.values(): sp.set_visible(False)
    strip.set_title(f"cell \u00d7 cell {lab}", fontsize=10.5, pad=7)
    ax = fig.add_subplot(gs[1, k])
    im = ax.imshow(M, cmap=RAMP, interpolation="nearest")
    for b_ in bounds:
        ax.axhline(b_, color="white", lw=0.35); ax.axvline(b_, color="white", lw=0.35)
    ax.axhline(bbound, color=INK, lw=1.3); ax.axvline(bbound, color=INK, lw=1.3)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[:, 3])
iu = np.triu_indices(n, 1)
same = co[iu[0]] == co[iu[1]]
v = CTo[iu]
bins = np.linspace(v.min(), v.max(), 70)
ax.hist(v[same], bins=bins, color="#2a78d6", alpha=0.9, label="within pseudo-clade", log=True)
ax.hist(v[~same], bins=bins, color="#eb6834", alpha=0.75, label="between", log=True)
cv = v[~same].std() / v[~same].mean()
fm = np.mean(v >= v.max() - 1e-9)
ax.set_xlabel("tree distance"); ax.set_ylabel("cell pairs (log scale)")
ax.set_title(f"Simulated metric is GRADED\nbetween-clade CV = {cv:.3f}; "
             f"{fm:.1%} at the max", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Simulated data, cells ordered by branch then pseudo-clade "
             "(black line = branch split; colour strip = pseudo-clade)\n"
             "Top-left quadrant is branch A, bottom-right is branch B.   "
             "PEtracer for comparison: between-clade CV 0.043, 49% of pairs at the maximum",
             fontsize=11.5, y=0.965, color=INK)
fig.savefig(HERE / "sim_tree_vs_space.png", dpi=155, facecolor="white")
print("wrote sim_tree_vs_space.png")
