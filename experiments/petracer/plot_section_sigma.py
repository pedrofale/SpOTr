"""The M3-4 section: cells by clade (with the motile ones marked), by motility, and by sigma."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.neighbourhood as nb
INK, GRID = "#0b0b0b", "#d8d8d4"
plt.rcParams.update({"axes.edgecolor": GRID, "text.color": INK, "font.size": 9})

d = np.load(HERE / "petracerFAIR_M3-4.npz", allow_pickle=True)
DT = np.asarray(d["C_tree"], float); XY = np.asarray(d["xy_sc"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
X = XY / np.median(np.linalg.norm(XY - np.median(XY, 0), axis=1))
SIG = np.array([1, 1.5, 2, 3, 4, 5, 7, 10, 14, 20, 30, 45, 70])
sig, L = nb.estimate_sigma(DT, X, SIG)
DTd = DT.copy(); np.fill_diagonal(DTd, np.inf)
o = np.argsort(DTd, 1)[:, :15]
cen = XY[o].mean(1)
spr = np.sqrt(((XY[o] - cen[:, None]) ** 2).sum(2).mean(1))
mot = np.linalg.norm(XY - cen, axis=1) / np.maximum(spr, 1e-9)
hi = mot >= np.quantile(mot, 0.75)

# clade colours ordered by the phylogeny so neighbouring colours are related clades
cl = np.unique(cc)
Dc = np.array([[DT[np.ix_(cc == a, cc == b)].mean() for b in cl] for a in cl])
np.fill_diagonal(Dc, 0)
order = leaves_list(linkage(squareform((Dc + Dc.T) / 2, checks=False), "average"))
cmap = plt.get_cmap("tab20")
COL = {cl[i]: cmap(k % 20) for k, i in enumerate(order)}

fig, ax = plt.subplots(1, 3, figsize=(15.4, 5.6))

a = ax[0]
a.scatter(XY[:, 0], XY[:, 1], c=[COL[c] for c in cc], s=9, edgecolors="none")
a.scatter(XY[hi, 0], XY[hi, 1], s=34, facecolors="none", edgecolors="k", lw=.55, alpha=.75)
for c in cl:
    m = cc == c
    if m.sum() >= 60:
        t = a.text(*np.median(XY[m], 0), c, fontsize=9, fontweight="bold", ha="center",
                   va="center", color="white")
        t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="black")])
a.set_title("Cells by clade\nblack rings = the motile quartile", fontsize=11)

a = ax[1]
s = a.scatter(XY[:, 0], XY[:, 1], c=np.clip(mot, 0, 6), s=9, cmap="magma_r",
              edgecolors="none")
a.set_title("Motility\n(distance from its 15 nearest relatives,\nin units of their spread)",
            fontsize=11)
fig.colorbar(s, ax=a, fraction=.046, pad=.03).ax.tick_params(labelsize=7.5)

a = ax[2]
s = a.scatter(XY[:, 0], XY[:, 1], c=np.log10(sig), s=9, cmap="viridis", edgecolors="none")
a.set_title("Inferred bandwidth $\\hat\\sigma$\n(green/yellow = needs a wider neighbourhood)",
            fontsize=11)
cb = fig.colorbar(s, ax=a, fraction=.046, pad=.03)
cb.set_ticks(np.log10([1, 2, 5, 10, 30, 70])); cb.set_ticklabels(["1", "2", "5", "10", "30", "70"])
cb.ax.tick_params(labelsize=7.5)

for a in ax:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fig.suptitle("PEtracer M3-4, 2281 cells: clade membership, motility, and the inferred "
             "neighbourhood bandwidth", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(HERE / "section_sigma.png", dpi=150, facecolor="white")
print("wrote section_sigma.png")
frac = [np.mean(hi[cc == c]) for c in cl if (cc == c).sum() >= 60]
big = [c for c in cl if (cc == c).sum() >= 60]
w = np.argsort(frac)[::-1]
print("clades with the most motile cells: " +
      ", ".join(f"{big[i]} ({frac[i]:.0%})" for i in w[:4]))
print("clades with the fewest:            " +
      ", ".join(f"{big[i]} ({frac[i]:.0%})" for i in w[-4:]))
