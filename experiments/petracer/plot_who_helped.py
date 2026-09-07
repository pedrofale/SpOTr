import numpy as np, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
d = np.load(HERE / "petracerFAIR_M3-4.npz", allow_pickle=True)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
T = pickle.loads((HERE / "where_cells_go.pkl").read_bytes())
e0 = np.linalg.norm(xy_st[np.asarray(T[0.0]).argmax(1)] - xy_sc, axis=1)
e6 = np.linalg.norm(xy_st[np.asarray(T[0.75]).argmax(1)] - xy_sc, axis=1)
delta = e0 - e6
cen = {c: np.median(xy_sc[cc == c], 0) for c in np.unique(cc)}
spr = {c: np.median(np.linalg.norm(xy_sc[cc == c] - cen[c], axis=1)) for c in cen}
rel = np.array([np.linalg.norm(xy_sc[i] - cen[cc[i]]) / max(spr[cc[i]], 1e-9)
                for i in range(len(cc))])
r = spearmanr(rel, delta)
q = np.quantile(rel, [0, .25, .5, .75, 1.0])
grp = [(rel >= q[k]) & (rel <= q[k+1]) for k in range(4)]
LAB = ["core\n(closest 25%)", "inner", "outer", "strays\n(furthest 25%)"]
COL = ["#1a7a3a", "#6aa84f", "#e08a3c", "#a83c10"]

fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.2))
a = ax[0]
core, stray = grp[0], grp[3]
a.scatter(xy_sc[:, 0], xy_sc[:, 1], s=3, c="#e8e8e6", edgecolors="none")
a.scatter(xy_sc[core, 0], xy_sc[core, 1], s=9, c=COL[0], edgecolors="none", label="core cells")
a.scatter(xy_sc[stray, 0], xy_sc[stray, 1], s=9, c=COL[3], edgecolors="none", label="strays")
a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
a.set_title("Cells sitting with their clone,\nand cells that have left it", fontsize=11)
a.legend(frameon=False, fontsize=9, loc="lower left")

a = ax[1]
v = [(delta[g] > 0).mean() for g in grp]
a.bar(range(4), v, .66, color=COL, edgecolor="white", lw=1.4)
a.axhline(.5, color="0.4", ls=":", lw=1.6)
for i, x in enumerate(v):
    a.text(i, x + .012, f"{x:.0%}", ha="center", fontsize=10, fontweight="bold")
a.set_xticks(range(4)); a.set_xticklabels(LAB, fontsize=8.5)
a.set_ylim(0, .72); a.set_ylabel("fraction of cells the tree improves")
a.set_title("Who does the tree help?", fontsize=11)
a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", alpha=.25); a.set_axisbelow(True)

a = ax[2]
v = [np.median(delta[g]) for g in grp]
a.bar(range(4), v, .66, color=COL, edgecolor="white", lw=1.4)
for i, x in enumerate(v):
    a.text(i, x + 5, f"{x:+.0f}", ha="center", fontsize=10, fontweight="bold")
a.set_xticks(range(4)); a.set_xticklabels(LAB, fontsize=8.5)
a.set_ylabel("median µm of placement error removed")
a.set_title("By how much?", fontsize=11)
a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", alpha=.25); a.set_axisbelow(True)

fig.suptitle("The tree term enforces clonal coherence — so it helps cells sitting inside their "
             f"clone and does nothing for cells that have strayed\n"
             f"(Spearman ρ = {r.statistic:+.3f}, p = {r.pvalue:.0e}, n = {len(cc)}).  "
             "Most of this variation is WITHIN clades, not between them.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.85])
fig.savefig(HERE / "who_the_tree_helps.png", dpi=150, facecolor="white")
print("wrote who_the_tree_helps.png")
