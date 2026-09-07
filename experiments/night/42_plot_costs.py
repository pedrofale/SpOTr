"""What the two costs actually ask for, side by side."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.clonespread import fit_spread
COL = {"M3-4": "#1b6ca8", "M3-1": "#d1852a", "M3-2": "#2f7d4f"}

fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))

# A -- r(a)
SP = {}
for SEC, c in COL.items():
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    SP[SEC] = fit_spread(CT[np.triu_indices(len(xy), 1)], pdist(xy))
    ax[0].plot(SP[SEC]["age"], SP[SEC]["r"], "o-", color=c, lw=2.4, ms=5, label=SEC)
ax[0].set_xlabel("days since the last common ancestor", fontsize=10.5)
ax[0].set_ylabel("r(a)   clone spread (um)", fontsize=10.5)
ax[0].set_title("A.  The entire model is this curve\n— one function, in microns", fontsize=12)
ax[0].legend(frameon=False, fontsize=9.5, loc="upper left")

sp = SP["M3-4"]
age = np.linspace(sp["age"].min(), 35, 120)
dist = np.linspace(20, 2200, 120)
Ag, Dg = np.meshgrid(age, dist, indexing="ij")
r = np.interp(Ag, sp["age"], sp["r"])

# B -- GW: matching numbers with different units
t_s = (Ag * 2 - (Ag * 2).mean()) / (Ag * 2).std()
d_s = (Dg - Dg.mean()) / Dg.std()
GW = (t_s - d_s) ** 2
im0 = ax[1].pcolormesh(dist, age, GW, cmap="magma_r", shading="auto")
ax[1].set_title("B.  GW asks the two distances to MATCH\n(days against microns)", fontsize=12)
plt.colorbar(im0, ax=ax[1], fraction=.046, label="cost")

# C -- clone spread
CS = Dg ** 2 / (2 * r ** 2)
im1 = ax[2].pcolormesh(dist, age, np.minimum(CS, np.quantile(CS, .97)), cmap="magma_r",
                       shading="auto")
ax[2].plot(np.interp(age, sp["age"], sp["r"]) * np.sqrt(2), age, "-", color="#39d0d8", lw=2.6,
           label="r(a): the spread the\nlineage predicts")
ax[2].legend(fontsize=9, loc="upper right", labelcolor="#0b6a70",
             facecolor="white", framealpha=.9, edgecolor="none")
ax[2].set_title("C.  Clone spread asks whether the separation\nis TYPICAL for that divergence age",
                fontsize=12)
plt.colorbar(im1, ax=ax[2], fraction=.046, label="cost")

for a_ in ax[1:]:
    a_.set_xlabel("spatial distance between the assigned spots (um)", fontsize=10.5)
    a_.set_ylabel("days since the last common ancestor", fontsize=10.5)
for a_ in ax:
    a_.grid(alpha=.18); a_.set_axisbelow(True)
    for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "costs.png", dpi=140, facecolor="white", bbox_inches="tight")
print("wrote costs.png")
