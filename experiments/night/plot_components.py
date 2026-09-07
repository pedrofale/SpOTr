"""What do the extra components add?  Show the transforms and the loss surfaces they build."""
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

SEC = "M3-4"
d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
iu = np.triu_indices(len(xy), 1)
t = CT[iu]; s = pdist(xy)
K = fit_kernel(t, s, ncomp=3)
tc, sc = K["tc"], K["sc"]

fig, ax = plt.subplots(2, 4, figsize=(18, 8))
CC = ["#1b6ca8", "#b8320f", "#2f7d4f"]
for r in range(3):
    ax[0, 0].plot(tc, K["phi"][r], "o-", color=CC[r], lw=2.2, ms=5,
                  label=f"component {r+1}   rho = {K['rho'][r]:.2f}")
    ax[1, 0].plot(sc, K["psi"][r], "o-", color=CC[r], lw=2.2, ms=4)
ax[0, 0].set_xlabel("patristic distance"); ax[0, 0].set_ylabel("phi(t)")
ax[0, 0].set_title("Tree transforms", fontsize=11); ax[0, 0].legend(frameon=False, fontsize=8.5)
ax[1, 0].set_xlabel("spatial distance (um)"); ax[1, 0].set_ylabel("psi(s)")
ax[1, 0].set_title("Spatial transforms", fontsize=11)

pa = K["a"]; pb = K["b"]
ts = (tc - (tc*pa).sum()) / np.sqrt(((tc**2)*pa).sum() - ((tc*pa).sum())**2)
ss = (sc - (sc*pb).sum()) / np.sqrt(((sc**2)*pb).sum() - ((sc*pb).sum())**2)
SQ = (ts[:, None] - ss[None, :]) ** 2
mats = [("square loss (FGW)", SQ)]
run = np.zeros_like(SQ)
for r in range(3):
    run = run - K["rho"][r] * np.outer(K["phi"][r], K["psi"][r])
    mats.append((f"learned, rank {r+1}", run.copy()))
w = np.outer(pa, pb)
for i, (nm, L) in enumerate(mats):
    mu = float((L*w).sum()); sd = float(np.sqrt((L**2*w).sum() - mu**2))
    Z = (L - mu) / sd
    a_ = ax[0, i+1] if i < 3 else ax[1, 3]
    if i == 3: a_ = ax[1, 3]
    a_ = [ax[0,1], ax[0,2], ax[0,3], ax[1,1]][i]
    v = np.abs(Z).max() * 0.8
    im = a_.pcolormesh(sc, tc, Z, cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
    a_.set_title(nm, fontsize=11)
    a_.set_xlabel("spatial distance (um)"); a_.set_ylabel("patristic distance")
    plt.colorbar(im, ax=a_, fraction=.045)

# where the two losses disagree, and the marginal weight of each cell
diff = None
mu = float((SQ*w).sum()); SQz = (SQ - mu)/np.sqrt((SQ**2*w).sum()-mu**2)
mu = float((mats[-1][1]*w).sum()); LRz = (mats[-1][1]-mu)/np.sqrt((mats[-1][1]**2*w).sum()-mu**2)
v = np.abs(LRz - SQz).max()*0.8
im = ax[1, 2].pcolormesh(sc, tc, LRz - SQz, cmap="PuOr_r", vmin=-v, vmax=v, shading="nearest")
ax[1, 2].set_title("learned rank 3  minus  square loss", fontsize=11)
ax[1, 2].set_xlabel("spatial distance (um)"); ax[1, 2].set_ylabel("patristic distance")
plt.colorbar(im, ax=ax[1, 2], fraction=.045)

ax[1, 3].barh(tc, pa, height=np.diff(K["te"])[:len(tc)]*0.8, color="#1b6ca8")
ax[1, 3].set_ylabel("patristic distance"); ax[1, 3].set_xlabel("fraction of all cell pairs")
ax[1, 3].set_title("where the pairs are", fontsize=11)

for a_ in ax.ravel():
    a_.grid(alpha=.2); a_.set_axisbelow(True)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
fig.suptitle(f"{SEC}: the structural loss, learned.  Blue = this (tree distance, spatial "
             f"distance) combination is FAVOURED;  red = penalised.", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(HERE / "components.png", dpi=135, facecolor="white")
print("wrote components.png"); print("rho:", K["rho"].round(3))
