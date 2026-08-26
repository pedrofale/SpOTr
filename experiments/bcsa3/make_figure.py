"""Diagnostic figure: why SpOTr does not yet clear the baseline bar on BCSA3."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

d = np.load(HERE / "bcsa3_B1_inputs.npz", allow_pickle=True)
xy, clones = d["xy"], d["clones"]
truth, A = d["spot_clone_numbat"], d["clone_assignments"]
prop = np.load(HERE / "prop_fixed.npy")
pred = clones[(prop / prop.sum(1, keepdims=True)).argmax(1)]

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.32, wspace=0.25)
cmap = plt.get_cmap("tab20")
norm = matplotlib.colors.BoundaryNorm(np.arange(0.5, len(clones) + 1.5), cmap.N)

for ax, lab, title in [(fig.add_subplot(gs[0, 0]), truth, "Numbat spot clones (target)"),
                       (fig.add_subplot(gs[0, 1]), pred, "SpOTr predicted dominant clone")]:
    ax.scatter(xy[:, 0], -xy[:, 1], c=lab, cmap=cmap, norm=norm, s=5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

# composition mismatch — the binding constraint
ax = fig.add_subplot(gs[0, 2])
ss3 = A.sum(0) / A.sum()
vis = np.array([(truth == c).sum() for c in clones], dtype=float); vis /= vis.sum()
x = np.arange(len(clones)); w = 0.4
ax.bar(x - w/2, ss3, w, label="SS3 reference cells", color="steelblue")
ax.bar(x + w/2, vis, w, label="Visium B1 spots (Numbat)", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels([str(c) for c in clones], fontsize=8)
ax.set_xlabel("clone"); ax.set_ylabel("fraction")
ax.set_title("Composition mismatch between modalities", fontsize=11)
ax.legend(fontsize=8)

# what balanced OT actually returns
ax = fig.add_subplot(gs[1, 0])
mp = (prop / prop.sum(1, keepdims=True)).mean(0)
ax.scatter(ss3, mp, s=45, color="crimson")
for i, c in enumerate(clones):
    ax.annotate(str(c), (ss3[i], mp[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
lim = [0, max(ss3.max(), mp.max()) * 1.1]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("SS3 reference clone frequency"); ax.set_ylabel("SpOTr mean spot proportion")
ax.set_title(f"SpOTr output = the reference prior (r={np.corrcoef(ss3, mp)[0,1]:.3f})", fontsize=11)

# alpha sweep: no weighting recovers the clones
ax = fig.add_subplot(gs[1, 1])
import json
sw = json.load(open(HERE / "sweep_alpha.json"))
al = [float(r["method"].split("=")[1]) for r in sw]
ax.plot(al, [r["argmax_acc"] for r in sw], "o-", label="argmax accuracy", color="crimson")
ax.plot(al, [r["ari"] for r in sw], "s-", label="adjusted Rand", color="slategray")
ax.axhline(sw[0]["majority"], ls="--", c="k", lw=0.9, label="majority baseline")
ax.set_xlabel(r"structure weight $\alpha$"); ax.set_ylabel("score")
ax.set_title(r"No $\alpha$ recovers the clones", fontsize=11)
ax.legend(fontsize=8)

# head-to-head
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
tbl = [["method", "acc", "ARI", "clone r"],
       ["majority", "0.280", "0.000", "--"],
       ["DestVI", "0.242", "0.032", "0.012"],
       ["scDOT", "0.262", "0.008", "0.017"],
       ["NNLS on CNV", "0.254", "0.016", "0.086"],
       ["SpOTr", "0.046", "0.036", "0.018"]]
t = ax.table(cellText=tbl[1:], colLabels=tbl[0], loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.55)
for j in range(4):
    t[0, j].set_facecolor("#dddddd")
ax.set_title("BCSA3 / B1 head-to-head", fontsize=11)

fig.suptitle("SpOTr on real BCSA3 data: the mapping is pinned to the reference composition",
             fontsize=13, y=0.975)
out = HERE / "bcsa3_B1_diagnosis.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
