"""Three clades, one per row: where they sit, how motile, and what bandwidth they need."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
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
cen15 = XY[o].mean(1)
spr15 = np.sqrt(((XY[o] - cen15[:, None]) ** 2).sum(2).mean(1))
mot = np.linalg.norm(XY - cen15, axis=1) / np.maximum(spr15, 1e-9)
thr = np.quantile(mot, 0.75)

CLADES = ["10", "2", "4"]           # most motile, middling, least motile
fig, ax = plt.subplots(len(CLADES), 3, figsize=(12.4, 4.35 * len(CLADES)))
for r, c in enumerate(CLADES):
    m = cc == c
    ctr = np.median(XY[m], 0)
    hi = m & (mot >= thr)
    # A: the clade in tissue context, motile cells ringed, lines to the clade centre
    a = ax[r, 0]
    a.scatter(XY[:, 0], XY[:, 1], s=3, c="#ececea", edgecolors="none")
    for i in np.where(hi)[0]:
        a.plot([XY[i, 0], ctr[0]], [XY[i, 1], ctr[1]], color="#a83c10", lw=.35, alpha=.45,
               zorder=1)
    a.scatter(XY[m, 0], XY[m, 1], s=15, c="#2a78d6", edgecolors="none", zorder=2)
    a.scatter(XY[hi, 0], XY[hi, 1], s=34, facecolors="none", edgecolors="#a83c10", lw=1.0,
              zorder=3)
    a.scatter(*ctr, s=210, marker="X", c="#111111", edgecolors="white", lw=1.3, zorder=5)
    a.set_ylabel(f"clade {c}\n{int(m.sum())} cells", fontsize=11.5)
    if r == 0:
        a.set_title("Where the clade sits\nrings = motile cells, X = clade centre", fontsize=10.5)
    a.set_xlabel(f"{np.mean(hi[m] if hi[m].size else [0]) if False else np.mean(mot[m] >= thr):.0%}"
                 f" of its cells are in the motile quartile", fontsize=9.5, color="#a83c10")
    # B: motility
    a = ax[r, 1]
    a.scatter(XY[:, 0], XY[:, 1], s=3, c="#f2f1ef", edgecolors="none")
    s1 = a.scatter(XY[m, 0], XY[m, 1], s=17, c=np.clip(mot[m], 0, 6), cmap="magma_r",
                   vmin=0, vmax=6, edgecolors="none")
    if r == 0:
        a.set_title("Motility of its cells", fontsize=10.5)
    a.set_xlabel(f"median {np.median(mot[m]):.2f}   (all cells {np.median(mot):.2f})",
                 fontsize=9.5)
    fig.colorbar(s1, ax=a, fraction=.046, pad=.03).ax.tick_params(labelsize=7)
    # C: inferred sigma
    a = ax[r, 2]
    a.scatter(XY[:, 0], XY[:, 1], s=3, c="#f2f1ef", edgecolors="none")
    s2 = a.scatter(XY[m, 0], XY[m, 1], s=17, c=np.log10(sig[m]), cmap="viridis",
                   vmin=0, vmax=np.log10(70), edgecolors="none")
    if r == 0:
        a.set_title("Inferred bandwidth $\\hat\\sigma$", fontsize=10.5)
    a.set_xlabel(f"median {np.median(sig[m]):g}   (all cells {np.median(sig):g})", fontsize=9.5)
    cb = fig.colorbar(s2, ax=a, fraction=.046, pad=.03)
    cb.set_ticks(np.log10([1, 3, 10, 30, 70])); cb.set_ticklabels(["1", "3", "10", "30", "70"])
    cb.ax.tick_params(labelsize=7)
    for k in range(3):
        ax[r, k].set_aspect("equal"); ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
fig.suptitle("Three clades of PEtracer M3-4, spanning the motility range", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(HERE / "clade_examples.png", dpi=150, facecolor="white")
for c in CLADES:
    m = cc == c
    print(f"clade {c}: n={int(m.sum())}, motile {np.mean(mot[m] >= thr):.0%}, "
          f"median motility {np.median(mot[m]):.2f}, median sigma {np.median(sig[m]):g}")
