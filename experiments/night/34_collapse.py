"""THE TEST THAT DECIDES WHETHER AN INTERPRETABLE LAW CAN TRANSFER ACROSS TUMOURS.

Proposal: the tree's spatial meaning is carried by ONE monotone function with units of microns --
r(a), the spread of a clone whose founding cell lived a days ago.  If the rescaled separation
s / r(a) has the SAME distribution in every tumour, then r(a) is the only thing that has to be
measured per tumour, and it is measurable from a single slice.  That is a dimensionless collapse,
and it is what "works on any tumour" would have to mean.

Tested on the three PEtracer sections (same tumour, so a weak test) AND on three independently
grown Eden tumours from the calibrated simulator (different tumours, the real test).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulation"))
import sim_eden


def curve(t, s, nb=22):
    """r(a) = sqrt(E[s^2 | a] / 2), the clone-spread function, on age bins"""
    a = t / 2.0
    cap = a.max()
    lo = a[a < cap - 1e-9]
    e = np.r_[np.linspace(lo.min(), lo.max(), nb), cap - 1e-9, cap + 1e-9]
    ac, r = [], []
    for i in range(len(e) - 1):
        m = (a >= e[i]) & (a < e[i + 1])
        if m.sum() < 200:
            continue
        ac.append(a[m].mean()); r.append(np.sqrt((s[m] ** 2).mean() / 2))
    return np.array(ac), np.array(r), a


def rescaled(t, s):
    ac, r, a = curve(t, s)
    return s / np.interp(a, ac, r), ac, r


DATA = {}
for SEC in ["M3-4", "M3-1", "M3-2"]:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    DATA[f"PEtracer {SEC}"] = (CT[np.triu_indices(len(xy), 1)], pdist(xy))
for sd in (0, 1, 2):
    S = sim_eden.simulate(seed=sd, n=1200, m_spots=128, dim=3, sigma_expr=6.0)
    CT = S["C_tree"]; xy = S["pos"]
    DATA[f"simulated tumour {sd}"] = (CT[np.triu_indices(len(xy), 1)], pdist(xy))

rng = np.random.default_rng(0)
Z, R = {}, {}
for k, (t, s) in DATA.items():
    z, ac, r = rescaled(t, s)
    Z[k] = rng.choice(z, min(200_000, len(z)), replace=False)
    R[k] = (ac / ac.max(), r / r.max())          # dimensionless: fractional age, fractional spread

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))
CL = ["#1b6ca8", "#d1852a", "#2f7d4f", "#b8320f", "#8a4fa8", "#6b7280"]
for (k, (x, y)), c in zip(R.items(), CL):
    ls = "-" if k.startswith("PEtracer") else "--"
    ax[0].plot(x, y, ls, color=c, lw=2.1, marker="o", ms=3.5, label=k)
ax[0].set_xlabel("age of the common ancestor / oldest")
ax[0].set_ylabel("clone spread r(a) / largest")
ax[0].set_title("A.  The clone-spread function,\nmade dimensionless", fontsize=11.5)
ax[0].legend(frameon=False, fontsize=8)

g = np.linspace(0, 4, 160)
for (k, z), c in zip(Z.items(), CL):
    ls = "-" if k.startswith("PEtracer") else "--"
    ax[1].plot(g[:-1], np.histogram(z, bins=g, density=True)[0], ls, color=c, lw=2.1, label=k)
ax[1].set_xlabel("rescaled separation   s / r(a)")
ax[1].set_ylabel("density")
ax[1].set_title("B.  Does the rescaled separation\ncollapse onto one law?", fontsize=11.5)
ax[1].legend(frameon=False, fontsize=8)

keys = list(Z)
M = np.zeros((len(keys), len(keys)))
for i in range(len(keys)):
    for j in range(len(keys)):
        M[i, j] = ks_2samp(Z[keys[i]][:40000], Z[keys[j]][:40000]).statistic
im = ax[2].imshow(M, cmap="RdYlBu_r", vmin=0, vmax=0.35)
ax[2].set_xticks(range(len(keys))); ax[2].set_yticks(range(len(keys)))
lab = [k.replace("PEtracer ", "PE ").replace("simulated tumour ", "sim ") for k in keys]
ax[2].set_xticklabels(lab, rotation=45, ha="right", fontsize=9)
ax[2].set_yticklabels(lab, fontsize=9)
for i in range(len(keys)):
    for j in range(len(keys)):
        ax[2].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8.5,
                   color="white" if M[i, j] > 0.2 else "black")
ax[2].set_title("C.  KS distance between the rescaled\nlaws  (0 = identical)", fontsize=11.5)
plt.colorbar(im, ax=ax[2], fraction=.046)

for a_ in ax[:2]:
    a_.grid(alpha=.22); a_.set_axisbelow(True)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "collapse.png", dpi=140, facecolor="white", bbox_inches="tight")
print("wrote collapse.png\n")
pe = [i for i, k in enumerate(keys) if k.startswith("PEtracer")]
si = [i for i, k in enumerate(keys) if k.startswith("sim")]
print(f"KS within PEtracer sections      {np.mean([M[i,j] for i in pe for j in pe if i<j]):.3f}")
print(f"KS within simulated tumours      {np.mean([M[i,j] for i in si for j in si if i<j]):.3f}")
print(f"KS PEtracer vs simulated         {np.mean([M[i,j] for i in pe for j in si]):.3f}")
