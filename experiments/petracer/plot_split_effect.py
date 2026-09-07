"""What splitting the tree's weight actually does to the mapping -- including the churn."""
import numpy as np, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
d = np.load(HERE / "petracerFAIR_M3-4.npz", allow_pickle=True)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
A = pickle.loads((HERE / "split_effect.pkl").read_bytes())
e = {k: np.linalg.norm(xy_st[v] - xy_sc, axis=1) for k, v in A.items()}
delta = e["uniform"] - e["split"]; n = len(delta)
same = A["uniform"] == A["split"]
imp = delta > 1; wor = delta < -1

fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.5))
a = ax[0]
for k, c, lab in [("uniform", "#7d7873", "one weight for the whole tree"),
                  ("split", "#1a6b3c", "near / mid / far weighted separately")]:
    a.plot(np.sort(e[k]), np.linspace(0, 1, n), lw=2.8, color=c,
           label=f"{lab}\nmedian {np.median(e[k]):.0f} µm")
a.set_xlim(0, 1600); a.set_xlabel("placement error (µm)"); a.set_ylabel("fraction of cells")
a.set_title("The whole section", fontsize=11.5)
a.legend(frameon=False, fontsize=9, loc="lower right")
a.grid(alpha=.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)

a = ax[1]
v = [same.mean(), imp.mean(), wor.mean()]
cols = ["#c9c5be", "#1a6b3c", "#a83c10"]
a.bar(range(3), v, .62, color=cols, edgecolor="white", lw=1.4)
for i, (x, t) in enumerate(zip(v, ["", f"median\n+{np.median(delta[imp]):.0f} µm",
                                   f"median\n−{np.median(-delta[wor]):.0f} µm"])):
    a.text(i, x + .012, f"{x:.0%}", ha="center", fontsize=11, fontweight="bold")
    if t: a.text(i, x / 2, t, ha="center", va="center", fontsize=9.5, color="white",
                 fontweight="bold")
a.set_xticks(range(3)); a.set_xticklabels(["assignment\nunchanged", "improved", "worsened"],
                                          fontsize=10)
a.set_ylim(0, .68); a.set_ylabel("fraction of cells")
a.set_title("Wins barely outnumber losses", fontsize=11.5)
a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", alpha=.25); a.set_axisbelow(True)

a = ax[2]
a.scatter(xy_sc[same, 0], xy_sc[same, 1], s=5, c="#dedbd5", edgecolors="none", label="unchanged")
a.scatter(xy_sc[wor, 0], xy_sc[wor, 1], s=9, c="#a83c10", alpha=.75, edgecolors="none",
          label="worsened")
a.scatter(xy_sc[imp, 0], xy_sc[imp, 1], s=9, c="#1a6b3c", alpha=.75, edgecolors="none",
          label="improved")
a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
a.set_title("and they are mixed together in space", fontsize=11.5)
a.legend(frameon=False, fontsize=9, loc="lower left", markerscale=2)

fig.suptitle("What the three-way split buys on M3-4: median error 398 → 381 µm.\n"
             "Nearly half the cells move, 23.6% improve and 20.9% get worse — a real net gain, "
             "but a small one bought with a lot of churn.", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.85])
fig.savefig(HERE / "split_effect.png", dpi=150, facecolor="white")
print("wrote split_effect.png")
