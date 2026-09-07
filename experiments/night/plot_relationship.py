"""THE relationship figure: what a lineage tree tells you about space, measured."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import fit_kernel

COL = {"M3-4": "#1b6ca8", "M3-1": "#d1852a", "M3-2": "#2f7d4f"}
K, D = {}, {}
for SEC in COL:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    iu = np.triu_indices(len(xy), 1)
    t = CT[iu]; s = pdist(xy)
    K[SEC] = fit_kernel(t, s, ncomp=3)
    D[SEC] = (t, s)

fig = plt.figure(figsize=(15.5, 8.6))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

# ---- A: the raw joint distribution, M3-4 -----------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
t, s = D["M3-4"]
h = ax.hist2d(t, s, bins=[np.linspace(0, 71, 90), np.linspace(0, 2250, 90)],
              cmap="Blues", norm=matplotlib.colors.LogNorm())
kk = K["M3-4"]
Es = np.array([s[(t >= kk["te"][i]) & (t < kk["te"][i+1])].mean() if
               ((t >= kk["te"][i]) & (t < kk["te"][i+1])).sum() > 30 else np.nan
               for i in range(len(kk["tc"]))])
ax.plot(kk["tc"], Es, "o-", color="#b8320f", lw=2.6, ms=6, label="mean spatial distance")
ax.axhline(s.mean(), color="0.35", ls=":", lw=1.6, label="two random cells")
ax.set_xlabel("patristic distance  (= 2 x days since common ancestor)")
ax.set_ylabel("spatial distance (um)")
ax.set_title("A.  M3-4: every cell pair\n2.6 M pairs; 49% sit at the 70-day ceiling", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")

# ---- B: the conditional mean, all sections, as a fraction of random --------------------------
ax = fig.add_subplot(gs[0, 1])
for SEC in COL:
    t, s = D[SEC]; kk = K[SEC]
    E = []
    for i in range(len(kk["tc"])):
        m = (t >= kk["te"][i]) & (t < kk["te"][i+1])
        E.append(s[m].mean() / s.mean() if m.sum() > 30 else np.nan)
    ax.plot(kk["tc"], E, "o-", color=COL[SEC], lw=2.2, ms=5, label=SEC)
ax.axhline(1.0, color="0.35", ls=":", lw=1.6)
ax.text(4, 1.012, "no information", fontsize=8, color="0.35")
ax.set_xlabel("patristic distance")
ax.set_ylabel("mean spatial distance / random pair")
ax.set_title("B.  Cells that diverged recently are close;\ncells that diverged at the founding "
             "are FURTHER than random", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5)

# ---- C: how much of the tree's information a square loss can see -----------------------------
ax = fig.add_subplot(gs[0, 2])
lab = list(COL)
raw = [np.corrcoef(*D[S])[0, 1] for S in lab]
mx = [K[S]["rho"][0] for S in lab]
x = np.arange(3); wd = 0.36
ax.bar(x - wd/2, raw, wd, color="#9aa7b1", label="square loss reads this\n(plain correlation)")
ax.bar(x + wd/2, mx, wd, color="#1b6ca8", label="best any transform can read\n(maximal correlation)")
for i in range(3):
    ax.text(x[i]-wd/2, raw[i]+.01, f"{raw[i]:.2f}", ha="center", fontsize=9)
    ax.text(x[i]+wd/2, mx[i]+.01, f"{mx[i]:.2f}", ha="center", fontsize=9, color="#1b6ca8")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, .68)
ax.set_ylabel("correlation between tree distance\nand spatial distance")
ax.set_title("C.  A square loss on raw distances leaves\n15-29% of the concordance unread",
             fontsize=10.5)
ax.legend(frameon=False, fontsize=8, loc="upper right")

# ---- D: the learned tree transform -----------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
for SEC in COL:
    kk = K[SEC]
    ax.plot(kk["tc"], kk["phi"][0], "o-", color=COL[SEC], lw=2.2, ms=5, label=SEC)
g = np.linspace(0, 71, 100)
ax.plot(g, (g - 55) / 14, "k--", lw=1.7, label="what FGW assumes\n(distance itself)")
ax.set_xlabel("patristic distance"); ax.set_ylabel("phi(tree distance)")
ax.set_title("D.  What the tree should actually be read as:\nnearly a step at the saturation "
             "ceiling", fontsize=10.5)
ax.legend(frameon=False, fontsize=8, loc="upper left")

# ---- E: the learned spatial transform --------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
for SEC in COL:
    kk = K[SEC]
    ax.plot(kk["sc"], kk["psi"][0], "o-", color=COL[SEC], lw=2.0, ms=4, label=SEC)
g = np.linspace(0, 2250, 100)
ax.plot(g, (g - 900) / 480, "k--", lw=1.7, label="what FGW assumes")
ax.set_xlabel("spatial distance (um)"); ax.set_ylabel("psi(spatial distance)")
ax.set_title("E.  …and how space should be read:\ncompressive past ~1.5 mm", fontsize=10.5)
ax.legend(frameon=False, fontsize=8, loc="upper left")

# ---- F: stability across sections -------------------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
g = np.linspace(2, 70, 150)
P = {S: np.interp(g, K[S]["tc"], K[S]["phi"][0]) for S in COL}
pairs = [("M3-4", "M3-1"), ("M3-4", "M3-2"), ("M3-1", "M3-2")]
for i, (a, b) in enumerate(pairs):
    ax.scatter(P[a], P[b], s=16, color=["#1b6ca8", "#d1852a", "#2f7d4f"][i],
               alpha=.75, edgecolors="none",
               label=f"{a} vs {b}   r = {np.corrcoef(P[a], P[b])[0,1]:.3f}")
lim = [-5, 2]
ax.plot(lim, lim, "k--", lw=1.4); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("phi fitted on one section"); ax.set_ylabel("phi fitted on another")
ax.set_title("F.  The transform is a property of the TUMOUR,\nnot of the section — so it "
             "transfers", fontsize=10.5)
ax.legend(frameon=False, fontsize=8, loc="upper left")

for a in fig.axes:
    a.grid(alpha=.22); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("The lineage tree and the tissue agree — but not in the way a square loss assumes",
             fontsize=13.5, y=0.985)
fig.savefig(HERE / "relationship.png", dpi=140, facecolor="white", bbox_inches="tight")
print("wrote relationship.png")
for S in COL:
    t, s = D[S]
    print(f"{S}: raw corr {np.corrcoef(t,s)[0,1]:+.3f}  maximal {K[S]['rho'][0]:.3f}  "
          f"rho2 {K[S]['rho'][1]:.3f}  gain {(K[S]['rho'][0]/np.corrcoef(t,s)[0,1]-1)*100:+.0f}%")
