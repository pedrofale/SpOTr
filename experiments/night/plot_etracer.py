"""eTRACER: a weaker reference still delivers the tree, but not a better way to read it."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
D = pd.read_csv(HERE / "etracer_map.csv")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

# A: clone-territory recovery
for i, tag in enumerate(["LE", "EE"]):
    g = D[D.tag == tag]
    e = g[g.w == 0].r.mean(); nul = g.null.mean()
    vals, errs = [nul, e], [0, 0]
    for name in ["FGW", "learned"]:
        b = g[(g.arm == name) & (g.w > 0)].groupby("seed").r.max()
        vals.append(b.mean()); errs.append(b.std())
    x = np.arange(4) + i * 5
    ax[0].bar(x, vals, yerr=errs, capsize=3,
              color=["#c9ced3", "#9aa7b1", "#7a99b5", "#1b6ca8"])
    for xx, v in zip(x, vals):
        ax[0].text(xx, v + .012, f"{v:.2f}", ha="center", fontsize=8.5)
    ax[0].text(x.mean(), 0.645, tag, ha="center", fontsize=11, fontweight="bold")
ax[0].set_xticks(list(np.arange(4)) + list(np.arange(4) + 5))
ax[0].set_xticklabels(["permuted", "expression", "FGW", "learned"] * 2, rotation=35,
                      ha="right", fontsize=8.5)
ax[0].set_ylabel("clone-territory agreement (r)")
ax[0].set_title("A.  eTRACER: the tree helps a lot,\nthe learned loss does not", fontsize=10.5)
ax[0].axhline(0, color="0.7", lw=1); ax[0].set_ylim(-0.05, 0.70)

# B: why -- the canonical spectrum of each reference
REF = {"PEtracer M3-4": [0.530, 0.230, 0.047], "PEtracer M3-2": [0.379, 0.171, 0.042],
       "PEtracer M3-1": [0.306, 0.146, 0.032], "eTRACER LE": [0.269, 0.007, 0.003],
       "eTRACER EE": [0.184, 0.024, 0.005]}
CL = ["#1b6ca8", "#2f7d4f", "#d1852a", "#b8320f", "#8a4fa8"]
for (k, v), c in zip(REF.items(), CL):
    ax[1].plot([1, 2, 3], v, "o-", color=c, lw=2.2, ms=6, label=k)
ax[1].set_xticks([1, 2, 3]); ax[1].set_xlabel("canonical component")
ax[1].set_ylabel("canonical correlation  rho_r")
ax[1].set_title("B.  A clone-level reference has NO second\ncomponent — nothing to learn",
                fontsize=10.5)
ax[1].legend(frameon=False, fontsize=8.5)

# C: rho_2 against the gain the learned loss delivers
pts = [("PEtracer M3-4", 0.230, 4.5), ("PEtracer M3-2", 0.171, 3.5), ("PEtracer M3-1", 0.146, 3.1),
       ("simulation", 0.056, 3.2), ("eTRACER EE", 0.024, 0.4), ("eTRACER LE", 0.007, 0.2)]
for (nm, r2, gain), c in zip(pts, CL + ["#555"]):
    ax[2].scatter(r2, gain, s=70, color=c, zorder=3, edgecolors="none")
    ax[2].annotate(nm, (r2, gain), textcoords="offset points", xytext=(7, -3), fontsize=8.5)
ax[2].axvspan(-0.005, 0.04, color="#f0e4dc", zorder=0)
ax[2].text(0.005, 4.7, "no second component:\nuse plain FGW", fontsize=8, color="#8a5a3a")
ax[2].set_xlim(-0.01, 0.30); ax[2].set_ylim(-0.4, 5.4)
ax[2].set_xlabel("rho_2 of the reference (computable before any mapping)")
ax[2].set_ylabel("improvement of the learned loss over FGW (%)")
ax[2].set_title("C.  rho_2 is a pre-flight check on whether\nthe learned loss can help at all",
                fontsize=10.5)

for a in ax:
    a.grid(alpha=.22); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "etracer.png", dpi=140, facecolor="white")
print("wrote etracer.png")
