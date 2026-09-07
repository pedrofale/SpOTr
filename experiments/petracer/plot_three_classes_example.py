"""The three classes of tree partner: measured across every cell, plus a representative cell."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
d = np.load(HERE / "petracer_M3-4.npz", allow_pickle=True)
CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
n = len(cc); cap = CT.max()
D = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
same = cc[:, None] == cc[None, :]; res = CT < cap - 1e-9
np.fill_diagonal(same, False); np.fill_diagonal(res, False)
CLS = [same & res, ~same & res, ~res]
COLS = ["#1a6b3c", "#e0a53c", "#a83c10"]
NAMES = ["same clade,\nresolved", "different clade,\nstill resolved", "at the saturation\nceiling"]
R = np.array([[np.median(D[i][m[i]]) for m in CLS] for i in range(n)])
base = np.array([np.median(D[i][np.arange(n) != i]) for i in range(n)])
Rn = R / base[:, None]
# a cell that behaves like the median cell
tgt = np.median(Rn, 0)
CELL = int(np.argmin(np.abs(Rn - tgt).sum(1)))
rng = np.random.default_rng(0)

fig = plt.figure(figsize=(14.6, 4.6))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, .12, 1.5], wspace=.30,
                      left=.045, right=.985, top=.80, bottom=.20)
for k in range(3):
    a = fig.add_subplot(gs[0, k])
    m = CLS[k][CELL].copy()
    sel = np.where(m)[0]
    if len(sel) > 900: sel = rng.choice(sel, 900, replace=False)
    a.scatter(xy[:, 0], xy[:, 1], s=2.5, c="#ececea", edgecolors="none")
    a.scatter(xy[sel, 0], xy[sel, 1], s=7, c=COLS[k], alpha=.55, edgecolors="none")
    a.scatter(*xy[CELL], s=280, marker="*", c="#2a78d6", edgecolors="white", lw=1.4, zorder=6)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(NAMES[k], fontsize=11)
    a.set_xlabel(f"{m.mean():.0%} of cells\n{Rn[CELL, k]:.2f}× a random cell",
                 fontsize=10, color=COLS[k], fontweight="bold")

a = fig.add_subplot(gs[0, 4])
parts = a.violinplot([Rn[:, k] for k in range(3)], showmedians=True, widths=.8)
for b, c in zip(parts["bodies"], COLS):
    b.set_facecolor(c); b.set_alpha(.55); b.set_edgecolor(c)
for key in ("cbars", "cmins", "cmaxes", "cmedians"):
    parts[key].set_color("0.3"); parts[key].set_linewidth(1.2)
a.axhline(1.0, color="0.35", ls="--", lw=1.6)
a.text(3.45, 1.02, "no information", fontsize=9, color="0.35", ha="right")
for k in range(3):
    a.text(k + 1, np.median(Rn[:, k]) + .07, f"{np.median(Rn[:, k]):.2f}", ha="center",
           fontsize=10, fontweight="bold", color=COLS[k])
a.set_xticks([1, 2, 3]); a.set_xticklabels(["near", "mid", "far"], fontsize=10)
a.set_ylabel("distance to that class ÷ to a random cell")
a.set_title("Every cell in the section (n = 2281)\n"
            "91.6% put their saturated partners further\nthan their same-clade partners",
            fontsize=10.5)
a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", alpha=.25); a.set_axisbelow(True)

fig.suptitle("The tree tells each cell three different things — and the current model gives the "
             "last two one shared weight", fontsize=12.5, y=0.95)
fig.savefig(HERE / "three_classes_example.png", dpi=150, facecolor="white")
print(f"representative cell {CELL} (clade {cc[CELL]}), ratios {Rn[CELL].round(2)}")
