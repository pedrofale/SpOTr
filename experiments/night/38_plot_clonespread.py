"""The clone-spread cost: what it is, how it does, and what it says about motility."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.clonespread import fit_spread
COL = {"M3-4": "#1b6ca8", "M3-1": "#d1852a", "M3-2": "#2f7d4f"}

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# A -- r(a), the one interpretable function
for SEC, c in COL.items():
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    sp = fit_spread(CT[np.triu_indices(len(xy), 1)], pdist(xy))
    ax[0].plot(sp["age"], sp["r"], "o-", color=c, lw=2.4, ms=5, label=SEC)
ax[0].set_xlabel("days since the last common ancestor")
ax[0].set_ylabel("r(a)   clone spread (um)")
ax[0].set_title("A.  The whole model: one function,\nin microns", fontsize=12)
ax[0].legend(frameon=False, fontsize=9.5)
ax[0].annotate("cost = $d^2/2r(a)^2$\nrecent relatives pulled hard,\nancient ones barely at all",
               xy=(0.04, 0.94), xycoords="axes fraction", va="top", fontsize=9.5,
               color="0.3", bbox=dict(boxstyle="round,pad=0.4", fc="#eef3f8", ec="none"))

# B -- head to head
H = pd.read_csv(HERE / "headtohead.csv"); G = H[H.w > 0]
arms = ["FGW", "clone spread", "learned rank 2"]
lab = ["GW\n(square loss)", "clone spread\n(this model)", "fitted transforms\n(SVD, 2 tables)"]
cols = ["#9aa7b1", "#1b6ca8", "#7fb0d8"]
x = np.arange(3); w = 0.26
for i, SEC in enumerate(COL):
    v = [G[(G.section == SEC) & (G.arm == a)].groupby(["seed", "half"]).err.min().mean()
         for a in arms]
    for j in range(3):
        ax[1].bar(i + (j - 1) * w, v[j], w, color=cols[j], label=lab[j] if i == 0 else None)
        ax[1].text(i + (j - 1) * w, v[j] + 5, f"{v[j]:.0f}", ha="center", fontsize=8.5)
    e0 = H[(H.section == SEC) & (H.w == 0)].err.mean()
    ax[1].plot([i - 1.6*w, i + 1.6*w], [e0, e0], color="#b8320f", lw=2, ls=":")
ax[1].text(2.05, 600, "expression only", fontsize=8.5, color="#b8320f", ha="right")
ax[1].set_xticks(x); ax[1].set_xticklabels(list(COL))
ax[1].set_ylabel("median placement error (um)"); ax[1].set_ylim(0, 660)
ax[1].set_title("B.  One interpretable function does what\ntwo fitted tables do", fontsize=12)
ax[1].legend(frameon=False, fontsize=8.5, loc="lower right")

# C -- motility index
T = pd.read_csv(HERE / "motility.csv", index_col=0)
T = T.loc[T.mean(1).sort_values().index]
xs = np.arange(len(T))
for SEC, c in COL.items():
    ax[2].plot(xs, T[SEC].values, "o", color=c, ms=7, label=SEC, alpha=.85)
ax[2].axhline(1.0, color="0.4", ls="--", lw=1.6)
ax[2].text(len(T) - .3, 1.03, "as dispersed as its\nlineage predicts", fontsize=8.5,
           color="0.4", ha="right")
ax[2].set_xticks(xs); ax[2].set_xticklabels(T.index, fontsize=8.5)
ax[2].set_xlabel("clade"); ax[2].set_ylabel("motility index   observed spread / r(a)")
ax[2].set_title("C.  Motility per clade, from a single slice\n(clade 8 settled, clade 14 motile)",
                fontsize=12)
ax[2].legend(frameon=False, fontsize=9)

for a_ in ax:
    a_.grid(alpha=.22); a_.set_axisbelow(True)
    for sp_ in ("top", "right"): a_.spines[sp_].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "clonespread.png", dpi=140, facecolor="white", bbox_inches="tight")
print("wrote clonespread.png")
