"""slide_negative.png -- where the premise holds and where it fails.

Three spatial maps, identical treatment.  Each point at its real coordinate, coloured by its
group label.  Title line 2 carries the territoriality ratio:

    mean spatial distance over SAME-group pairs / mean spatial distance over DIFFERENT-group pairs

Well below 1  -> the groups own territory.
Near 1.00     -> they do not, and the whole approach has nothing to lock onto.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist

HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/night")
PET = HERE.parent / "petracer"
BCS = HERE.parent / "bcsa3"
OUT = HERE / "slide_negative.png"

MAXN = 3000
UM_PER_PX = 100.0 / 174.0   # Visium: 100 um centre-to-centre, median NN = 174 full-res px


def territoriality(xy, lab, maxn=MAXN, seed=0):
    """mean(d | same group) / mean(d | different group), over all pairs."""
    xy = np.asarray(xy, float)
    lab = np.asarray(lab)
    if len(xy) > maxn:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(xy), maxn, replace=False)
        xy, lab = xy[keep], lab[keep]
    _, code = np.unique(lab, return_inverse=True)
    d = pdist(xy)
    same = pdist(code[:, None], metric="chebyshev") == 0
    return float(d[same].mean()), float(d[~same].mean()), float(d[same].mean() / d[~same].mean()), \
           int(len(xy)), int(same.sum()), int((~same).sum())


def colours(labels, order):
    """Stable tab20 colour per group, by position in the canonical group list."""
    cmap = plt.get_cmap("tab20")
    lut = {g: cmap(i % 20) for i, g in enumerate(order)}
    return np.array([lut[v] for v in labels])


# ---------------------------------------------------------------- data
p = np.load(PET / "petracerFAIR_M3-4.npz", allow_pickle=True)
xy1 = np.asarray(p["xy_sc"], float)                      # um
lab1 = np.asarray(p["cell_clade"]).astype(str)
ord1 = list(np.asarray(p["clades"]).astype(str))

e = np.load(HERE / "etracer_spots_LE.npz", allow_pickle=True)
xy2 = np.asarray(e["xy_st"], float)                      # um
clones2 = np.asarray(e["clones"]).astype(str)
lab2 = clones2[np.asarray(e["spot_comp"], float).argmax(1)]   # dominant clone per spot
ord2 = list(clones2)

b = np.load(BCS / "bcsa3_D1_inputs.npz", allow_pickle=True)
xy3 = np.asarray(b["xy"], float) * UM_PER_PX             # px -> um
lab3 = np.asarray(b["spot_clone_numbat"]).astype(int)    # 1-based clone ids, used as labels only
ord3 = [int(c) for c in np.asarray(b["clones"])]

panels = [
    ("PEtracer (lineage recorder)",      xy1, lab1, ord1, 7.0),
    ("eTRACER (lineage recorder)",       xy2, lab2, ord2, 30.0),
    ("BCSA3 D1 (CNV clones, Numbat)",    xy3, lab3, ord3, 7.0),
]

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

report = []
for ax, (name, xy, lab, order, s) in zip(axes, panels):
    ms, md, ratio, n, npair_s, npair_d = territoriality(xy, lab)
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c=colours(lab, order), linewidths=0, alpha=0.95)
    ax.set_title(name, fontsize=13.5, pad=10)
    good = ratio < 0.85
    ax.text(0.02, 0.985, f"{ratio:.2f}x", transform=ax.transAxes, ha="left", va="top",
            fontsize=22, fontweight="bold",
            color=("#1b6ca8" if good else "#b8320f"), bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.82))
    ax.text(0.02, 0.895, "same-group vs different-group\nmean pair distance",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="0.35",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.82))
    ax.text(0.02, 0.03, ("territories" if good else "no territories"),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
            fontweight="bold", color=("#1b6ca8" if good else "#b8320f"), bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.82))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    report.append((name, len(np.unique(lab)), n, ms, md, ratio, npair_s, npair_d))

fig.suptitle("The method needs lineage groups that occupy space  —  and CNV clones on Visium do not", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")

for name, ng, n, ms, md, ratio, ps, pd_ in report:
    print(f"{name} | groups={ng} | points={n} | same_pairs={ps} | diff_pairs={pd_} | "
          f"mean_same_um={ms:.4f} | mean_diff_um={md:.4f} | ratio={ratio:.6f} | onfig={ratio:.2f}")
print("UM_PER_PX", UM_PER_PX)
print("OUT", OUT)
