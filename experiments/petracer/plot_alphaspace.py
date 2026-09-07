import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95])
SEC = "M3-4"
H = np.load(HERE / f"exact_halves_{SEC}.npy")
EA, EB = H[0, 0], H[0, 1]
Z = (EA - EA.mean(1, keepdims=True)) / (EA.std(1, keepdims=True) + 1e-9)
lab = KMeans(2, n_init=10, random_state=0).fit_predict(Z)
COL = ["#1f77b4", "#d62728"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
for p, (E, ttl) in enumerate([(EA, "Half A — where the clusters were defined"),
                              (EB, "Half B — held out")]):
    for k in (0, 1):
        c = E[lab == k].mean(0)
        ax[p].plot(GRID, c / c.mean(), "o-", color=COL[k], lw=2.2, ms=5,
                   label=f"cluster {k+1}  (n={int((lab==k).sum())})")
        b = GRID[c.argmin()]
        ax[p].axvline(b, color=COL[k], ls=":", lw=1.6)
        ax[p].annotate(f"α*={b:g}", (b, 1.0 + .012*(1-2*k)), color=COL[k], fontsize=10,
                       ha="center", fontweight="bold")
    ax[p].set_title(ttl, fontsize=10.5)
    ax[p].set_xlabel("α"); ax[p].legend(frameon=False, fontsize=9)
ax[0].set_ylabel("mean placement error\n(scaled per cluster)")
fig.suptitle("Clustering cells in α-space, PEtracer M3-4:  the two groups want different α "
             "only on the data that defined them", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(HERE / "alphaspace_clusters.png", dpi=155)
print("wrote alphaspace_clusters.png")
