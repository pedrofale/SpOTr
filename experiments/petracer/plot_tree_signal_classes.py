import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
HERE = Path(__file__).resolve().parent
SECS = ["M3-4", "M3-1", "M3-2"]
CLS = [("same clade,\nresolved", "#1a6b3c"), ("different clade,\nstill resolved", "#e0a53c"),
       ("at the saturation\nmaximum", "#a83c10")]
res = {}
for SEC in SECS:
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    iu = np.triu_indices(len(xy), 1)
    t = CT[iu]; s = squareform(pdist(xy))[iu]; base = s.mean()
    same = cc[iu[0]] == cc[iu[1]]; resv = t < t.max() - 1e-9
    masks = [same & resv, ~same & resv, ~resv]
    res[SEC] = ([m.mean() for m in masks], [s[m].mean()/base for m in masks])

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
x = np.arange(3); w = 0.26
for k, SEC in enumerate(SECS):
    ax[0].bar(x + (k-1)*w, res[SEC][0], w, label=SEC,
              color=["#2a78d6", "#6aa8e0", "#a8c8ea"][k], edgecolor="white", lw=1.2)
    ax[1].bar(x + (k-1)*w, res[SEC][1], w, label=SEC,
              color=["#2a78d6", "#6aa8e0", "#a8c8ea"][k], edgecolor="white", lw=1.2)
ax[0].set_ylabel("fraction of all cell pairs")
ax[0].set_title("How the tree's pairs divide up", fontsize=11.5)
ax[1].axhline(1.0, color="#a83c10", lw=1.8, ls="--")
ax[1].annotate("no information", (2.42, 1.006), fontsize=9, color="#a83c10", ha="right")
ax[1].set_ylabel("spatial distance ÷ a random pair")
ax[1].set_title("What each kind of pair says about space", fontsize=11.5)
ax[1].set_ylim(0, 1.42)
for a in ax:
    a.set_xticks(x); a.set_xticklabels([c[0] for c in CLS], fontsize=9.5)
    a.legend(frameon=False, fontsize=9)
    a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", alpha=.25)
    a.set_axisbelow(True)
for k, SEC in enumerate(SECS):
    for i, v in enumerate(res[SEC][1]):
        ax[1].text(i + (k-1)*w, v + .02, f"{v:.2f}", ha="center", fontsize=7.6)
fig.suptitle("The tree carries three different messages, and the current model lumps two of them "
             "together.\nSame-clade pairs ATTRACT strongly, saturated pairs REPEL — and the "
             "large middle class attracts weakly.", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.85])
fig.savefig(HERE / "tree_signal_classes.png", dpi=150, facecolor="white")
print("wrote tree_signal_classes.png")
