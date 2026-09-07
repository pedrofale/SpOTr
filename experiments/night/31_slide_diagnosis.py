"""Slide figure: what a square loss can and cannot read from a lineage tree."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import fit_kernel

SECS = ["M3-4", "M3-1", "M3-2"]
COL = {"M3-4": "#1b6ca8", "M3-1": "#d1852a", "M3-2": "#2f7d4f"}
K, RAW = {}, {}
for S in SECS:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{S}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    t = CT[np.triu_indices(len(xy), 1)]; s = pdist(xy)
    K[S] = fit_kernel(t, s, ncomp=3); RAW[S] = np.corrcoef(t, s)[0, 1]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

# A -- the one number, measured against its ceiling
x = np.arange(3); w = 0.36
raw = [RAW[S] for S in SECS]; mx = [K[S]["rho"][0] for S in SECS]
ax[0].bar(x - w/2, raw, w, color="#9aa7b1", label="what a square loss reads\n(plain correlation)")
ax[0].bar(x + w/2, mx, w, color="#1b6ca8", label="the most any transform\ncould read")
for i in range(3):
    ax[0].text(x[i]-w/2, raw[i]+.012, f"{raw[i]:.2f}", ha="center", fontsize=10.5, color="#5a646d")
    ax[0].text(x[i]+w/2, mx[i]+.012, f"{mx[i]:.2f}", ha="center", fontsize=10.5,
               color="#1b6ca8", fontweight="bold")
    ax[0].annotate("", xy=(x[i]+w/2, mx[i]-.015), xytext=(x[i]-w/2, raw[i]-.015),
                   arrowprops=dict(arrowstyle="->", color="#b8320f", lw=1.6,
                                   connectionstyle="arc3,rad=-0.35"))
ax[0].set_xticks(x); ax[0].set_xticklabels(SECS, fontsize=11); ax[0].set_ylim(0, 0.63)
ax[0].set_ylabel("correlation between tree distance\nand spatial distance", fontsize=10.5)
ax[0].set_title("A.  The tree enters through one number —\nand it is not maximised", fontsize=12)
ax[0].legend(frameon=False, fontsize=9, loc="upper right")

# B -- the transform on the tree side
for S in SECS:
    ax[1].plot(K[S]["tc"], K[S]["phi"][0], "o-", color=COL[S], lw=2.3, ms=5, label=S)
g = np.linspace(0, 71, 100)
ax[1].plot(g, (g - 55) / 14, "k--", lw=2, label="what FGW assumes\n(distance itself)")
ax[1].axvline(70, color="0.75", lw=1, ls=":")
ax[1].text(69, -4.4, "ceiling\n28-49% of pairs", fontsize=8.5, color="0.45", ha="right")
ax[1].set_xlabel("patristic distance", fontsize=10.5)
ax[1].set_ylabel("$\\varphi$(tree distance)", fontsize=11)
ax[1].set_title("B.  How the tree should be read:\nnearly a step at the ceiling", fontsize=12)
ax[1].legend(frameon=False, fontsize=9, loc="upper left")

# C -- the transform on the spatial side
for S in SECS:
    ax[2].plot(K[S]["sc"], K[S]["psi"][0], "o-", color=COL[S], lw=2.1, ms=4, label=S)
g = np.linspace(0, 2250, 100)
ax[2].plot(g, (g - 900) / 480, "k--", lw=2, label="what FGW assumes")
ax[2].set_xlabel("spatial distance (um)", fontsize=10.5)
ax[2].set_ylabel("$\\psi$(spatial distance)", fontsize=11)
ax[2].set_title("C.  …and how space should be read:\ncompressive past ~1.5 mm", fontsize=12)
ax[2].legend(frameon=False, fontsize=9, loc="upper left")

for a in ax:
    a.grid(alpha=.22); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "slide_diagnosis.png", dpi=150, facecolor="white", bbox_inches="tight")
print("wrote slide_diagnosis.png")
for S in SECS:
    print(f"  {S}: plain {RAW[S]:+.3f}  maximal {K[S]['rho'][0]:.3f}  "
          f"gain {(K[S]['rho'][0]/RAW[S]-1)*100:+.0f}%   rho2 {K[S]['rho'][1]:.3f}")
