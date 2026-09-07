"""Demonstration figure under the clone-spread cost."""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = pickle.load(open(HERE / "final_maps_cs.pkl", "rb"))
SEC = "M3-4"; d = D[SEC]
xy_sc, xy_st = d["xy_sc"], d["xy_st"]
ARMS = [("expression only", "expression only"), ("GW", "GW  (square loss)"),
        ("clone spread", "clone spread  (this model)")]
fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.4))
rng = np.random.default_rng(1)
sub = rng.choice(len(xy_sc), 220, replace=False)
for k, (key, name) in enumerate(ARMS):
    a = ax[0, k]; am = d["res"][key]
    err = np.linalg.norm(xy_st[am] - xy_sc, axis=1)
    a.scatter(xy_st[:, 0], xy_st[:, 1], s=9, c="0.86", edgecolors="none", zorder=1)
    for i in sub:
        a.plot([xy_sc[i, 0], xy_st[am[i], 0]], [xy_sc[i, 1], xy_st[am[i], 1]], "-", lw=.7,
               color="#b8320f" if err[i] > 400 else "#1b6ca8", alpha=.65, zorder=2)
    a.scatter(xy_sc[sub, 0], xy_sc[sub, 1], s=7, c="k", zorder=3)
    a.set_title(f"{name}\nmedian error {np.median(err):.0f} um   "
                f"{np.mean(err < 250):.0%} within one spot", fontsize=11.5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)
clade = d["clade"]; names = d["clade_names"]
pick = np.argsort(-np.bincount(clade))[:3]
for k, c in enumerate(pick):
    a = ax[1, k]; idx = np.where(clade == c)[0]
    a.scatter(xy_st[:, 0], xy_st[:, 1], s=9, c="0.9", edgecolors="none")
    a.scatter(xy_sc[idx, 0], xy_sc[idx, 1], s=13, c="#1b6ca8", label="true positions",
              edgecolors="none")
    am = d["res"]["clone spread"][idx]
    a.scatter(xy_st[am, 0] + 18, xy_st[am, 1] + 18, s=13, c="#b8320f", alpha=.75,
              label="where the model places them", edgecolors="none")
    a.set_title(f"clade {names[c]}   ({len(idx)} cells)", fontsize=11.5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)
    if k == 0:
        a.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle(f"{SEC}: r(a) fitted on the other half of the cells, spot marginal from observed "
             f"density", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(HERE / "maps_cs.png", dpi=135, facecolor="white", bbox_inches="tight")
print("wrote maps_cs.png")
