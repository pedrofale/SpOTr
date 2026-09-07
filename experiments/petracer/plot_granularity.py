import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d8d8d4"
C1, C2 = "#2a78d6", "#eb6834"
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90])
SEC = "M3-4"
r = pd.read_csv(f"granularity_controlled_{SEC}.csv")
E = np.load(f"percell_err_{SEC}.npy")
d = np.load(f"petracer_{SEC}.npz", allow_pickle=True)
cc = np.asarray(d["cell_clade"]).astype(str)
dd = np.load("petracer_xy.npz", allow_pickle=True)
Dcl = np.asarray(dd["D"], float); allcl = list(dd["clades"].astype(str))
Zc = linkage(squareform(Dcl, checks=False), "average")
lab6 = np.array([(fcluster(Zc, 6, "maxclust")-1)[allcl.index(c)] for c in cc])
best = GRID[E.argmin(1)]

fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4))

# A: the coarse version of the middle panel the user liked
ax = axes[0]
for k in sorted(set(lab6)):
    mu = E[lab6 == k].mean(0)
    ax.plot(GRID, mu/mu.mean(), lw=2, label=f"group {k} (n={int((lab6==k).sum())})")
mu = E.mean(0)
ax.plot(GRID, mu/mu.mean(), c=INK, lw=3, ls="--", label="all cells", zorder=5)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("mean placement error (scaled per group)")
ax.set_title("Coarsened to 6 groups\n(cut of the clade-level tree)", fontsize=10.5)
ax.legend(frameon=False, fontsize=7.5, ncol=2)
ax.grid(color=GRIDC, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# B: reproducibility vs granularity, against the size-matched random null
ax = axes[1]
x = np.arange(len(r))
ax.fill_between(x, r.exact_null - r.exact_null_sd, r.exact_null + r.exact_null_sd,
                color=GRIDC, alpha=0.75, label="size-matched random\ngrouping (±1 sd)")
ax.plot(x, r.exact_null, c=INK2, lw=2, ls="--")
ax.plot(x, r.exact, c=C2, lw=2.6, marker="o", ms=8, markeredgecolor="white",
        markeredgewidth=1.1, label="clade grouping", zorder=4)
ax.set_xticks(x); ax.set_xticklabels(r.groups)
ax.set_xlabel("number of groups"); ax.set_ylabel("exact agreement of selected $\\alpha$\nbetween spot halves")
ax.set_title("Clade sits ON or BELOW the random null\nat every granularity", fontsize=10.5)
ax.legend(frameon=False, fontsize=8)
ax.grid(color=GRIDC, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# C: z scores
ax = axes[2]
ax.bar(x, r.z, 0.6, color=[C2 if v > 0 else GRIDC for v in r.z],
       edgecolor="white", linewidth=1.2)
ax.axhline(0, c=INK, lw=1)
ax.axhline(1.64, c=INK2, ls=":", lw=1.4)
ax.text(len(x)-0.4, 1.72, "p = 0.05", ha="right", fontsize=8, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(r.groups)
ax.set_xlabel("number of groups"); ax.set_ylabel("z vs the random null")
ax.set_ylim(-1.6, 2.2)
ax.set_title("No granularity beats chance\n(coarsening does not rescue it)", fontsize=10.5)
ax.grid(axis="y", color=GRIDC, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Is 18 clades too fine a resolution for $\\alpha_k$?  Coarsening does not help.",
             fontsize=13, y=0.985, color=INK)
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig("petracer_granularity.png", dpi=150, facecolor="white")
print("wrote petracer_granularity.png")
