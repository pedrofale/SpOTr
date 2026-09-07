"""The mapping itself: where cells actually are, and where each cost puts them."""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = pickle.load(open(HERE / "final_maps.pkl", "rb"))
SEC = "M3-4"
d = D[SEC]
xy_sc, xy_st = d["xy_sc"], d["xy_st"]
ARMS = [("expression only", "expression only"), ("FGW", "FGW  (tree, square loss)"),
        ("learned", "learned loss + observed density\n(this paper)")]

fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.6))

# row 1: displacement arrows for a random sample of cells
rng = np.random.default_rng(1)
sub = rng.choice(len(xy_sc), 220, replace=False)
for k, (key, name) in enumerate(ARMS):
    a = ax[0, k]
    am = d["res"][key]
    err = np.linalg.norm(xy_st[am] - xy_sc, axis=1)
    a.scatter(xy_st[:, 0], xy_st[:, 1], s=9, c="0.86", edgecolors="none", zorder=1)
    for i in sub:
        a.plot([xy_sc[i, 0], xy_st[am[i], 0]], [xy_sc[i, 1], xy_st[am[i], 1]],
               "-", lw=.7, color="#b8320f" if err[i] > 400 else "#1b6ca8", alpha=.65, zorder=2)
    a.scatter(xy_sc[sub, 0], xy_sc[sub, 1], s=7, c="k", zorder=3)
    a.set_title(f"{name}\nmedian error {np.median(err):.0f} um   "
                f"{np.mean(err < 250):.0%} within one spot", fontsize=11)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)

# row 2: three clades, true positions vs where the method puts those same cells
clade = d["clade"]; names = d["clade_names"]
sizes = np.bincount(clade)
pick = np.argsort(-sizes)[:3]
for k, c in enumerate(pick):
    a = ax[1, k]
    idx = np.where(clade == c)[0]
    a.scatter(xy_st[:, 0], xy_st[:, 1], s=9, c="0.9", edgecolors="none")
    a.scatter(xy_sc[idx, 0], xy_sc[idx, 1], s=13, c="#1b6ca8", label="true positions",
              edgecolors="none")
    am = d["res"]["learned"][idx]
    a.scatter(xy_st[am, 0] + 18, xy_st[am, 1] + 18, s=13, c="#b8320f", alpha=.75,
              label="where the method places them", edgecolors="none")
    a.set_title(f"clade {names[c]}   ({len(idx)} cells)", fontsize=11)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)
    if k == 0:
        a.legend(frameon=False, fontsize=8.5, loc="upper left")

fig.suptitle(f"{SEC}: transforms fitted on the other half of the cells, spot marginal from\n"
             f"observed density — the protocol used throughout", fontsize=12.5)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(HERE / "maps.png", dpi=135, facecolor="white")
print("wrote maps.png")
for SEC2 in D:
    for key, _ in ARMS:
        am = D[SEC2]["res"][key]
        e = np.linalg.norm(D[SEC2]["xy_st"][am] - D[SEC2]["xy_sc"], axis=1)
        print(f"{SEC2} {key:<16} median {np.median(e):6.1f}  within one spot {np.mean(e<250):.1%}")
