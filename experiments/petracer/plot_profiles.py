import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulation"))
from sim_varalpha_branches import simulate
from local_concordance import local_score

rng = np.random.default_rng(0)
pts = []
def add(name, CT, xy, grp, colour, marker):
    s = local_score(CT, xy, 5, restrict=grp)
    for g, tag in zip(np.unique(grp), ["A", "B"]):
        idx = np.where(grp == g)[0]
        sm = rng.choice(idx, min(1200, len(idx)), replace=False)
        iu = np.triu_indices(len(sm), 1)
        ct = CT[np.ix_(sm, sm)][iu]; cs = squareform(pdist(xy[sm]))[iu]
        pts.append((f"{name} {tag}", pearsonr(ct, cs)[0], 1 - s[idx].mean(), colour, marker))

for SEC in ["M3-4", "M3-1", "M3-2"]:
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float)
    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1
    add(SEC, CT, np.asarray(d["xy_sc"], float), br, "#eb6834", "o")
for tn, mk in [(0.10, "s"), (0.35, "^")]:
    s_ = simulate(0, tree_noise=tn)
    add(f"sim {tn}", s_["C_tree"], s_["xy_sc"], s_["branch"], "#2a78d6", mk)

fig, ax = plt.subplots(figsize=(8.6, 6.0))
for name, gx, ly, c, m in pts:
    ax.scatter(gx, ly, s=150, color=c, marker=m, edgecolors="white", lw=1.4, zorder=3)
    ax.annotate(name, (gx, ly), textcoords="offset points", xytext=(9, -3),
                fontsize=8.5, color="#333")
ax.axvspan(0.6, 1.05, color="#2a78d6", alpha=.07, zorder=0)
ax.text(0.82, 0.05, "square-loss GW\nneeds this region", ha="center", fontsize=9,
        color="#1a5490")
ax.annotate("PEtracer has MORE local structure than sim 0.35 A,\n"
            "where GW found nothing — but less global structure.\n"
            "That gap is what a Laplacian term could reach.",
            xy=(0.30, 0.30), xytext=(0.44, 0.60), fontsize=8.8, color="#a83c10",
            arrowprops=dict(arrowstyle="->", color="#a83c10", lw=1.2))
ax.axhspan(0.4, 0.75, xmin=0, xmax=0.42, color="#eb6834", alpha=.07, zorder=0)
ax.set_xlabel("GLOBAL: corr(tree distance, spatial distance)   ← what square-loss GW reads")
ax.set_ylabel("LOCAL: 1 − (spatial distance to 5 nearest tree relatives / random)\n"
              "← what a Laplacian term would read")
ax.set_title("PEtracer carries NEIGHBOURHOOD structure without METRIC structure.\n"
             "The simulation only ever has both together — which is why it is not a fair test "
             "of the cost.", fontsize=11)
ax.grid(alpha=.25); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-0.02, 1.05)
fig.tight_layout()
fig.savefig(HERE / "concordance_profiles.png", dpi=155)
print("wrote concordance_profiles.png")
for name, gx, ly, *_ in pts:
    print(f"  {name:<10} global {gx:+.3f}   local {ly:.3f}")
