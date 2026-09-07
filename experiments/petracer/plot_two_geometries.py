"""The two distance matrices, ordered two ways. Should they match?"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform, pdist
HERE = Path(__file__).resolve().parent
INK, GRID = "#0b0b0b", "#d8d8d4"
plt.rcParams.update({"axes.edgecolor": GRID, "text.color": INK, "font.size": 9})

d = np.load(HERE / "petracerFAIR_M3-4.npz", allow_pickle=True)
CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
rng = np.random.default_rng(0)
sub = np.sort(rng.choice(len(cc), 700, replace=False))
CT, xy, cc = CT[np.ix_(sub, sub)], xy[sub], cc[sub]
CS = squareform(pdist(xy))

# order 1: by the tree itself
o_tree = leaves_list(linkage(squareform(CT, checks=False), "average"))
# order 2: by space (average linkage on spatial distance)
o_sp = leaves_list(linkage(squareform(CS, checks=False), "average"))

fig, ax = plt.subplots(2, 2, figsize=(9.6, 9.8))
for r, (o, oname) in enumerate([(o_tree, "cells ordered by the TREE"),
                                (o_sp, "cells ordered by SPACE")]):
    for c, (Mx, mname, cmapn) in enumerate([(CT, "tree distance (days x 2)", "Blues"),
                                            (CS, "spatial distance (µm)", "Oranges")]):
        a = ax[r, c]
        im = a.imshow(Mx[np.ix_(o, o)], cmap=cmapn, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(mname, fontsize=10.5)
        fig.colorbar(im, ax=a, fraction=.046, pad=.03).ax.tick_params(labelsize=7)
    ax[r, 0].set_ylabel(oname, fontsize=11)
fig.suptitle("The same 700 cells, the same two matrices, ordered two ways\n"
             "Each ordering makes ONE matrix block-structured and leaves the other mottled",
             fontsize=12, y=0.975)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(HERE / "two_geometries.png", dpi=150, facecolor="white")

# quantify: can an ultrametric tree live in 2D at all?
from sklearn.manifold import MDS
iu = np.triu_indices(len(CT), 1)
emb = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
          normalized_stress="auto").fit(CT / CT.mean())
dd = squareform(pdist(emb.embedding_))
t = CT[iu]/CT[iu].mean()
print("Can the tree metric be represented in 2-D at all?")
print(f"  best 2-D embedding of the tree: stress = {emb.stress_:.1f}, "
      f"corr with true tree distance = {np.corrcoef(dd[iu], t)[0,1]:.3f}")
ev = np.linalg.eigvalsh(-0.5 * (np.eye(len(CT)) - 1/len(CT)) @ (CT**2) @
                        (np.eye(len(CT)) - 1/len(CT)))[::-1]
ev = ev[ev > 0]
print(f"  variance captured by the top 2 dimensions: {ev[:2].sum()/ev.sum():.1%}")
print(f"  dimensions needed for 90%: {int(np.searchsorted(np.cumsum(ev)/ev.sum(), 0.90))+1}")
print(f"\n  observed corr(tree, space) on these cells: "
      f"{np.corrcoef(t, CS[iu]/CS[iu].mean())[0,1]:.3f}")
print("wrote two_geometries.png")
