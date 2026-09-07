"""How expression-similar are neighbouring cells -- neighbouring in the TREE, and in SPACE?

Same construction as plot_lineage_neighbours.py, with expression distance as the response.
Expression distance is the SpOTr feature metric: squared Euclidean, normalised by its max.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SEC = "M3-4"
p = np.load(f"petracer_{SEC}.npz", allow_pickle=True)
CT = np.asarray(p["C_tree"], float)
X = np.asarray(p["X_sc"], float)
xy = np.asarray(p["xy_sc"], float)
cc = np.asarray(p["cell_clade"]).astype(str)
n = len(xy)
CS = squareform(pdist(xy))
CE = squareform(pdist(X, metric="sqeuclidean")); CE = CE / CE.max()
iu = np.triu_indices(n, 1)
t, s, e = CT[iu], CS[iu], CE[iu]
same = cc[iu[0]] == cc[iu[1]]
base = np.median(e)
print(f"{SEC}: {n} cells | median expression distance {base:.4f}")

fig = plt.figure(figsize=(15.2, 7.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.38, wspace=0.27,
                      top=0.855, bottom=0.06, left=0.058, right=0.98)

def curve(ax, x, edges, lab, colour, mask=None, marker="o"):
    mids, meds, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if mask is not None: m &= mask
        if m.sum() < 40: continue
        mids.append(np.median(x[m])); meds.append(np.median(e[m])); cnt.append(int(m.sum()))
    ax.plot(mids, meds, c=colour, lw=2.3, marker=marker, ms=6.5, label=lab,
            markeredgecolor="white", markeredgewidth=0.9)
    return mids, meds, cnt

# A: expression vs TREE distance
ax = fig.add_subplot(gs[0, 0])
te = np.array([0,5,10,20,30,40,50,55,60,65,70.01])
_, m1, c1 = curve(ax, t, te, "all pairs", C1)
ax.axhline(base, color=INK2, ls=":", lw=1.4)
ax.text(2, base*1.012, f"random pair: {base:.3f}", fontsize=8.5, color=INK2)
ax.set_xlabel("tree (patristic) distance"); ax.set_ylabel("median expression distance")
ax.set_ylim(0.440, 0.497)
ax.set_title(f"Expression vs LINEAGE distance\nspan {max(m1)-min(m1):.4f}, non-monotone", fontsize=10.5)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)

# B: expression vs SPATIAL distance, split by same/different clade
ax = fig.add_subplot(gs[0, 1])
se = np.array([0,25,50,100,200,400,800,1600,3200])
_, m2, c2 = curve(ax, s, se, "all pairs", C1)
curve(ax, s, se, "same clade", C3, mask=same, marker="s")
curve(ax, s, se, "different clade", C2, mask=~same, marker="D")
ax.axhline(base, color=INK2, ls=":", lw=1.4)
ax.set_xscale("log"); ax.set_xlim(10, 2600)
ax.set_xlabel("spatial distance (µm)"); ax.set_ylabel("median expression distance")
ax.set_ylim(0.440, 0.497)
ax.set_title(f"Expression vs SPATIAL distance\nspan {max(m2)-min(m2):.4f}, monotone", fontsize=10.5)
ax.legend(frameon=False, fontsize=8)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)

# C: kNN in space / in tree / random -> expression distance
ax = fig.add_subplot(gs[0, 2])
Dt = CT + np.eye(n)*1e9; Ds = CS + np.eye(n)*1e9
rng = np.random.default_rng(0); KS = [1,2,5,10,20,50]
for lab, D_, colour, mk in [("nearest in SPACE", Ds, C2, "D"), ("nearest in TREE", Dt, C1, "o")]:
    v = []
    for k in KS:
        idx = np.argsort(D_, axis=1)[:, :k]
        v.append(np.median([CE[i, idx[i]].mean() for i in range(n)]))
    ax.plot(KS, v, c=colour, lw=2.3, marker=mk, ms=6.5, label=lab,
            markeredgecolor="white", markeredgewidth=0.9)
rd = [np.median([CE[i, rng.choice(n, k)].mean() for i in range(n)]) for k in KS]
ax.plot(KS, rd, c=GRID, lw=2.3, marker="s", ms=6.5, label="random cells",
        markeredgecolor="white", markeredgewidth=0.9)
ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("k"); ax.set_ylabel("median expression distance to the k cells")
ax.set_title("Are a cell's neighbours expression-similar?", fontsize=10.5)
ax.legend(frameon=False, fontsize=8)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)

# D-F: tissue coloured by expression distance to one cell
rng2 = np.random.default_rng(3)
for j, i0 in enumerate(rng2.choice(n, 3, replace=False)):
    ax = fig.add_subplot(gs[1, j])
    d0 = CE[i0].copy(); d0[i0] = np.nan
    lo, hi = np.nanpercentile(d0, [2, 98])
    sc_ = ax.scatter(xy[:,0], xy[:,1], s=8, c=d0, cmap="Blues_r", vmin=lo, vmax=hi, linewidth=0)
    ax.scatter(*xy[i0], s=130, marker="*", c=C2, edgecolor="white", linewidth=1.2, zorder=5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.set_title(f"cell {i0} (star): tissue coloured by\nexpression similarity to it", fontsize=9.5)
    cb = fig.colorbar(sc_, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=7.5)
    cb.set_label("expression distance", fontsize=8)

fig.suptitle(f"PEtracer {SEC}: expression similarity against lineage distance and against spatial distance",
             fontsize=13, y=0.945, color=INK)
fig.savefig("petracer_expression_neighbours.png", dpi=150, facecolor="white")
print("wrote petracer_expression_neighbours.png")

# printed summary
print(f"\n  expression distance by TREE distance bin: {['%.4f'%v for v in m1]}")
print(f"  expression distance by SPACE distance bin: {['%.4f'%v for v in m2]}")
for lab, D_ in [("space", Ds), ("tree", Dt)]:
    idx = np.argsort(D_, axis=1)[:, :5]
    v = np.median([CE[i, idx[i]].mean() for i in range(n)])
    print(f"  5 nearest in {lab:>5}: expression distance {v:.4f}  vs random {rd[2]:.4f}  "
          f"ratio {v/rd[2]:.4f}")
