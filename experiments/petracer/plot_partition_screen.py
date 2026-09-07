import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95])
LAM = np.linspace(0, 1, 11)
rng = np.random.default_rng(0)
COL = {"M3-4": "#1f77b4", "M3-1": "#d62728"}
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

for SEC in ["M3-4", "M3-1"]:
    H = np.load(HERE / f"exact_halves_{SEC}.npy"); n = H.shape[2]
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    cc = np.asarray(d["cell_clade"]).astype(str); CT = np.asarray(d["C_tree"], float)
    clades = list(np.asarray(d["clades"]).astype(str))
    cidx = np.array([clades.index(c) for c in cc])
    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1

    cur, gains, nulls = [], {"2 branches": [], "18 clades": []}, {2: [], 18: []}
    for rep in range(H.shape[0]):
        for A, B in [(0, 1), (1, 0)]:
            EA, EB = H[rep, A], H[rep, B]
            gi = EA.mean(0).argmin(); bi = EA.argmin(1); g0 = EB[:, gi].mean()
            cur.append([np.mean([np.interp(GRID[gi] + l*(GRID[bi[i]]-GRID[gi]), GRID, EB[i])
                                 for i in range(n)]) for l in LAM])
            def gp(lab):
                o = np.empty(n)
                for k in np.unique(lab):
                    s = lab == k; o[s] = EB[s, EA[s].mean(0).argmin()]
                return g0 - o.mean()
            gains["2 branches"].append(gp(br)); gains["18 clades"].append(gp(cidx))
            for K in (2, 18):
                nulls[K] += [gp(rng.integers(0, K, n)) for _ in range(40)]
    C = np.array(cur).mean(0)

    axes[0].plot(LAM, C - C[0], "o-", color=COL[SEC], lw=2, ms=5, label=SEC)
    k = int(C.argmin())
    if k: axes[0].scatter([LAM[k]], [C[k]-C[0]], s=170, facecolors="none",
                          edgecolors=COL[SEC], lw=2.2, zorder=5)
    xs = [0, 1] if SEC == "M3-4" else [2, 3]
    for x, key, K in zip(xs, ["2 branches", "18 clades"], [2, 18]):
        nn = np.array(nulls[K])
        axes[1].bar(x, np.mean(gains[key]), color=COL[SEC], width=.62, zorder=3)
        axes[1].errorbar(x, nn.mean(), yerr=nn.std(), fmt="_", ms=26, color="0.25",
                         lw=2.2, capsize=6, zorder=4)

axes[0].axhline(0, color="0.4", lw=1)
axes[0].set_xlabel("shrinkage λ   (0 = one global α,  1 = each cell its own best α)")
axes[0].set_ylabel("held-out placement error,\nchange vs a single α (µm)")
axes[0].set_title("Per-cell α: signal in M3-4, none in M3-1", fontsize=10.5)
axes[0].legend(frameon=False)
axes[0].text(.03, .70, "below 0 = better\nthan one global α", transform=axes[0].transAxes,
             ha="left", va="top", fontsize=8.5, color="0.35")

axes[1].axhline(0, color="0.4", lw=1)
axes[1].set_xticks(range(4))
axes[1].set_xticklabels(["2 branches", "18 clades", "2 branches", "18 clades"], fontsize=9)
axes[1].set_ylabel("held-out µm saved vs a single α")
axes[1].set_title("Grouped α: no better than a random grouping of the same size\n"
                  "(grey = mean ± sd of 240 random groupings)", fontsize=10.5)
for x, s in [(0.5, "M3-4"), (2.5, "M3-1")]:
    axes[1].text(x, axes[1].get_ylim()[1]*.93, s, ha="center", fontsize=10, color=COL[s])
fig.tight_layout()
fig.savefig(HERE / "partition_screen.png", dpi=155)
print("wrote partition_screen.png")
