"""PEtracer and the branching simulator in the same views, one figure, matched styling."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sim_branching

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
RAMP = "Blues"
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9})

# ---- PEtracer M3-4
p = np.load(HERE.parent / "petracer" / "petracer_M3-4.npz", allow_pickle=True)
CTp = np.asarray(p["C_tree"], float); xyp = np.asarray(p["xy_sc"], float)
cc = np.asarray(p["cell_clade"]).astype(str)
cl = list(np.asarray(p["clades"]).astype(str))
ordc = leaves_list(linkage(squareform(
    np.asarray(np.load(HERE.parent / "petracer" / "petracer_xy.npz",
                       allow_pickle=True)["D"], float), checks=False), "average"))
phylo = [np.asarray(np.load(HERE.parent / "petracer" / "petracer_xy.npz",
                            allow_pickle=True)["clades"]).astype(str)[i] for i in ordc]
key = {c: i for i, c in enumerate(phylo)}
cip = np.array([key.get(c, 99) for c in cc])
op = np.argsort(cip, kind="stable")

# ---- simulation
d = sim_branching.simulate(0, bl_shape=8.0, sat_q=0.5, sigma_expr=10.0)
CTs, xys, cls = d["C_tree"], d["pos"], d["clade"]
os_ = np.argsort(cls + 1000 * d["branch"], kind="stable")

cmap = plt.get_cmap("tab20")
fig = plt.figure(figsize=(14.4, 9.2))
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], wspace=0.20, hspace=0.28,
                      top=0.88, bottom=0.05, left=0.04, right=0.97)

for row, (name, CT, xy, ci, order, nclade) in enumerate([
        ("PEtracer M3-4", CTp, xyp, cip, op, len(phylo)),
        ("Branching simulation", CTs, xys, cls, os_, len(np.unique(cls)))]):
    COL = np.array([cmap(k % 20) for k in range(max(nclade, 20))])
    a = fig.add_subplot(gs[row, 0])
    a.scatter(xy[:, 0], xy[:, 1], c=COL[ci % len(COL)], s=7, edgecolors="none")
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{name}\ncells coloured by clade ({nclade})", fontsize=10.5)

    co = ci[order]
    bounds = np.where(np.diff(co) != 0)[0] + 1
    for k, (M, lab) in enumerate([(CT[np.ix_(order, order)], "tree distance"),
                                  (squareform(pdist(xy[order])), "spatial distance")]):
        a = fig.add_subplot(gs[row, k + 1])
        im = a.imshow(M, cmap=RAMP, interpolation="nearest")
        for b_ in bounds:
            a.axhline(b_, color="white", lw=0.3); a.axvline(b_, color="white", lw=0.3)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"cell × cell {lab}", fontsize=10.5)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7.5)

fig.suptitle("PEtracer against the branching simulator, same views, cells ordered by clade\n"
             "clade coherence 0.591 vs 0.659   ·   tree–space corr 0.208 vs 0.256   ·   "
             "local concordance 0.511 vs 0.471   ·   pairs at the tree maximum 49% vs 50%",
             fontsize=12, y=0.965, color=INK)
fig.savefig(HERE / "side_by_side.png", dpi=150, facecolor="white")
print("wrote side_by_side.png")
