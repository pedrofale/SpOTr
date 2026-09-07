import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

BASE = "/home/pedro/projects/spotr/phylospatial_mapping/experiments"
OUT  = BASE + "/night/slide_clade.png"

d4 = np.load(BASE + "/petracer/petracerFAIR_M3-4.npz", allow_pickle=True)
d1 = np.load(BASE + "/petracer/petracerFAIR_M3-1.npz", allow_pickle=True)

xy4, cc4 = np.asarray(d4["xy_sc"], float), np.asarray(d4["cell_clade"]).astype(str)
xy1, cc1 = np.asarray(d1["xy_sc"], float), np.asarray(d1["cell_clade"]).astype(str)

labels = sorted(set(cc4.tolist()) | set(cc1.tolist()), key=lambda s: int(s))
n4 = {l: int((cc4 == l).sum()) for l in labels}
n1 = {l: int((cc1 == l).sum()) for l in labels}
tot = {l: n4[l] + n1[l] for l in labels}

chosen = sorted(labels, key=lambda l: (-tot[l], int(l)))[:3]

print("combined ranking (clade, M3-4, M3-1, total):")
for l in sorted(labels, key=lambda l: (-tot[l], int(l))):
    print("   clade %-3s %5d %5d %5d" % (l, n4[l], n1[l], tot[l]))
print("chosen:", chosen)

colors = ["#1b9e77", "#d95f02", "#7570b3"]   # Dark2, colourblind-safe

# common limits so the two sections are directly comparable
allxy = np.vstack([xy4, xy1])
lo, hi = allxy.min(0), allxy.max(0)
pad = 0.04 * np.ptp(allxy, 0).max()
xlim = (lo[0] - pad, hi[0] + pad + 0.33 * np.ptp(allxy, 0)[1])   # room for the legend
ylim = (lo[1] - pad, hi[1] + pad)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

for ax, xy, cc, counts, title in [
    (axes[0], xy4, cc4, n4, "M3-4"),
    (axes[1], xy1, cc1, n1, "M3-1 - the same three clades"),
]:
    ax.scatter(xy[:, 0], xy[:, 1], s=7, c="#d9d9d9", linewidths=0, zorder=1)
    for l, col in zip(chosen, colors):
        m = cc == l
        ax.scatter(xy[m, 0], xy[m, 1], s=16, c=col, linewidths=0, zorder=2,
                   label="clade %s  (n=%d)" % (l, counts[l]))
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("x (µm)", fontsize=11)
    ax.set_ylabel("y (µm)", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.9,
              markerscale=1.8, borderpad=0.5, handletextpad=0.5)

fig.text(0.5, 0.015,
         "the same clade is a large territory in one section and a handful of cells in the next "
         "- so a per-clade parameter is a property of the slice, not of the clade",
         ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
print("WROTE", OUT)
