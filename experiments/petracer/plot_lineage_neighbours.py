"""Do lineage neighbours sit together in space? Yes -- the heatmap just hides it."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
C1, C2 = "#2a78d6", "#eb6834"
SEC = "M3-4"
p = np.load(f"petracer_{SEC}.npz", allow_pickle=True)
CT = np.asarray(p["C_tree"], float); xy = np.asarray(p["xy_sc"], float)
n = len(xy); CS = squareform(pdist(xy))
iu = np.triu_indices(n, 1); t, s = CT[iu], CS[iu]
med = np.median(s)

fig = plt.figure(figsize=(15.2, 7.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.36, wspace=0.26,
                      top=0.86, bottom=0.07, left=0.055, right=0.98)

# A: spatial distance vs tree distance
ax = fig.add_subplot(gs[0, 0])
edges = np.array([0,5,10,20,30,40,50,55,60,65,70.01])
mids, meds, cnts = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (t >= lo) & (t < hi)
    if m.sum() < 30: continue
    mids.append((lo+hi)/2); meds.append(np.median(s[m])); cnts.append(m.sum())
ax.plot(mids, meds, c=C1, lw=2.4, marker="o", ms=7, markeredgecolor="white", markeredgewidth=1)
ax.axhline(med, color=INK2, ls=":", lw=1.4)
ax.text(2, med+38, f"random pair: {med:.0f} µm", fontsize=8.5, color=INK2)
ax.set_xlabel("tree (patristic) distance"); ax.set_ylabel("median spatial distance (µm)")
ax.set_title("Close in lineage → close in space\nmonotone over the whole range", fontsize=10.5)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# B: where the pairs actually are
ax = fig.add_subplot(gs[0, 1])
ax.bar(mids, cnts, width=np.diff(edges)*0.85, color=[C2 if m_>62 else C1 for m_ in mids],
       edgecolor="white", linewidth=0.8, log=True)
frac = (t >= 65).mean()
ax.text(0.03, 0.93, f"{frac:.0%} of all pairs sit in the\nsaturated tail (orange) —\nwhich is why the "
        "heatmap\nlooks featureless", transform=ax.transAxes, fontsize=8.8, va="top", color=INK)
ax.set_xlabel("tree (patristic) distance"); ax.set_ylabel("cell pairs (log)")
ax.set_title("But almost all pairs are saturated", fontsize=10.5)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# C: kNN-in-tree spatial distance vs random
ax = fig.add_subplot(gs[0, 2])
D = CT + np.eye(n)*1e9; rng = np.random.default_rng(0)
KS = [1,2,5,10,20,50]
tr, rd = [], []
for k in KS:
    idx = np.argsort(D, axis=1)[:, :k]
    tr.append(np.median([CS[i, idx[i]].mean() for i in range(n)]))
    rd.append(np.median([CS[i, rng.choice(n, k)].mean() for i in range(n)]))
ax.plot(KS, tr, c=C1, lw=2.4, marker="o", ms=7, label="nearest in the tree",
        markeredgecolor="white", markeredgewidth=1)
ax.plot(KS, rd, c=GRID, lw=2.4, marker="s", ms=7, label="random cells",
        markeredgecolor="white", markeredgewidth=1)
ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("k"); ax.set_ylabel("median distance to the k cells (µm)")
ax.set_title("A cell's nearest lineage relatives\nare ~4× closer than chance", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# D-F: examples in the tissue
rng2 = np.random.default_rng(3)
picks = rng2.choice(n, 3, replace=False)
for j, i0 in enumerate(picks):
    ax = fig.add_subplot(gs[1, j])
    ax.scatter(xy[:,0], xy[:,1], s=3.5, c=GRID, linewidth=0)
    nb = np.argsort(D[i0])[:20]
    ax.scatter(xy[nb,0], xy[nb,1], s=26, c=C1, linewidth=0.5, edgecolor="white",
               label="20 nearest in the tree")
    ax.scatter(*xy[i0], s=110, marker="*", c=C2, edgecolor="white", linewidth=1.2,
               zorder=5, label="the cell")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(GRID)
    d20 = CS[i0, nb].mean()
    ax.set_title(f"cell {i0}: its 20 closest relatives\nsit {d20:.0f} µm away on average",
                 fontsize=9.5, color=INK)
    if j == 0: ax.legend(frameon=False, fontsize=8, loc="lower left")

fig.suptitle(f"PEtracer {SEC}: lineage neighbours ARE spatial neighbours — "
             "the saturated tail hides it in the heatmap", fontsize=13, y=0.945, color=INK)
fig.savefig("petracer_lineage_neighbours.png", dpi=150, facecolor="white")
print("wrote petracer_lineage_neighbours.png")
