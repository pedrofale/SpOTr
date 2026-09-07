"""The branching simulator in the same views as PEtracer and the old simulator."""
import sys, numpy as np, ot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sim_branching

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
RAMP = "Blues"
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.titlesize": 10.5})

d = sim_branching.simulate(0, bl_shape=8.0, sat_q=0.5, sigma_expr=10.0)
pos, clade, br = d["pos"], d["clade"], d["branch"]
CT, n = d["C_tree"], d["n"]
xy_sc, xy_st, tspot = d["xy_sc"], d["xy_st"], d["true_spot"]
K = len(np.unique(clade))
cmap = plt.get_cmap("tab20")
COL = np.array([cmap(k % 20) for k in range(K)])

# ---------------- figure A: the simulated tissue ----------------
fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.8))
ax[0].scatter(pos[:, 0], pos[:, 1], c=COL[clade], s=11, edgecolors="none")
ax[0].set_title(f"Cells coloured by clade ({K} clades,\ncut at generation 4)")
ax[1].scatter(pos[:, 0], pos[:, 1], c=np.where(br == 0, "#2a78d6", "#eb6834"),
              s=11, edgecolors="none")
ax[1].set_title("The two main branches\n(the root split)")
rngc = np.random.default_rng(0)
scol = rngc.permutation(plt.get_cmap("tab20b")(np.linspace(0, 1, 20)))
ax[2].scatter(pos[:, 0], pos[:, 1], c=scol[tspot % 20], s=11, edgecolors="none")
ax[2].scatter(xy_st[:, 0], xy_st[:, 1], c="k", s=5, marker="x", lw=.7)
ax[2].set_title(f"The {d['m']} spots (equal-size k-d partition;\nblack × = spot centroid)")
for a in ax:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fig.suptitle("Branching-process simulation: cells divide and daughters displace, so clades hold "
             "territories\nthat overlap heavily — clade coherence 0.66, matching PEtracer's 0.59–0.66",
             fontsize=12, y=0.99, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.87])
fig.savefig(HERE / "sim_branching_space.png", dpi=155, facecolor="white")
print("wrote sim_branching_space.png")

# ---------------- figure B: cell x cell heatmaps ----------------
order = np.argsort(clade + 1000 * br, kind="stable")
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
    strip.set_title(f"cell × cell {lab}", fontsize=10.5, pad=7)
    a = fig.add_subplot(gs[1, k])
    im = a.imshow(M, cmap=RAMP, interpolation="nearest")
    for b_ in bounds:
        a.axhline(b_, color="white", lw=0.35); a.axvline(b_, color="white", lw=0.35)
    a.axhline(bbound, color=INK, lw=1.3); a.axvline(bbound, color=INK, lw=1.3)
    a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=8)

a = fig.add_subplot(gs[:, 3])
iu = np.triu_indices(n, 1)
same = co[iu[0]] == co[iu[1]]
v = CTo[iu]
bins = np.linspace(v.min(), v.max(), 70)
a.hist(v[same], bins=bins, color="#2a78d6", alpha=0.9, label="within clade", log=True)
a.hist(v[~same], bins=bins, color="#eb6834", alpha=0.75, label="between clades", log=True)
cv = v[~same].std() / v[~same].mean()
fm = np.mean(v >= v.max() * 0.999)
a.annotate(f"{fm:.0%} of ALL pairs sit\nat exactly {v.max():.1f}", xy=(v.max(), 4e4),
           xytext=(-120, -4), textcoords="offset points", fontsize=8.5, color=INK,
           arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
a.set_xlabel("tree distance"); a.set_ylabel("cell pairs (log scale)")
a.set_title(f"Saturated, like the real recorder\nbetween-clade CV = {cv:.3f}", fontsize=10.5)
a.legend(frameon=False, fontsize=8.5, loc="upper left")
a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Branching simulation, cells ordered by branch then clade "
             "(black line = root split; colour strip = clade)\n"
             "PEtracer M3-4: between-clade CV 0.043, 49% of pairs at the maximum",
             fontsize=11.5, y=0.965, color=INK)
fig.savefig(HERE / "sim_branching_heatmaps.png", dpi=155, facecolor="white")
print("wrote sim_branching_heatmaps.png")
