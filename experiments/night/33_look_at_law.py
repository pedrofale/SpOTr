"""Look at the actual conditional law before assuming a functional form."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
HERE = Path(__file__).resolve().parent
COL = {"M3-4": "#1b6ca8", "M3-1": "#d1852a", "M3-2": "#2f7d4f"}

fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
store = {}
for SEC, c in COL.items():
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    t = CT[np.triu_indices(len(xy), 1)]; s = pdist(xy)
    a = t / 2.0                                   # days since the last common ancestor
    cap = a.max()
    e = np.r_[np.linspace(a[a < cap - 1e-9].min(), a[a < cap - 1e-9].max(), 22),
              cap - 1e-9, cap + 1e-9]
    ac, sig, sd, n = [], [], [], []
    for i in range(len(e) - 1):
        m = (a >= e[i]) & (a < e[i + 1])
        if m.sum() < 200:
            continue
        ac.append(a[m].mean()); sig.append(np.sqrt((s[m] ** 2).mean() / 2))
        sd.append(s[m].std()); n.append(m.sum())
    ac, sig, sd, n = map(np.array, (ac, sig, sd, n))
    store[SEC] = (a, s, ac, sig, n, np.ptp(xy, 0).max())
    ax[0].plot(ac, sig, "o-", color=c, lw=2.2, ms=5, label=SEC)
    R = np.ptp(xy, 0).max() / 2
    ax[0].axhline(R / np.sqrt(2) if SEC == "M3-4" else np.nan, color=c, ls=":", lw=1)

ax[0].set_xlabel("days since the last common ancestor")
ax[0].set_ylabel("sigma = sqrt( E[ |xi - xk|^2 ] / 2 )   (um)")
ax[0].set_title("A.  The spread the tree predicts,\nin microns", fontsize=11.5)
ax[0].legend(frameon=False, fontsize=9)

# B: is it linear in sqrt(age)?  Brownian => sigma proportional to sqrt(age)
for SEC, c in COL.items():
    a, s, ac, sig, n, ext = store[SEC]
    ax[1].plot(np.sqrt(ac), sig, "o-", color=c, lw=2.2, ms=5, label=SEC)
ax[1].set_xlabel("sqrt(days since common ancestor)")
ax[1].set_ylabel("sigma (um)")
ax[1].set_title("B.  Brownian motion predicts\na straight line here", fontsize=11.5)

# C: the conditional distribution at three ages, against a Rayleigh with the same mean
SEC = "M3-4"; a, s, ac, sig, n, ext = store[SEC]
cap = a.max()
for lab, m, c in [("recent (< 10 d)", a < 10, "#1b6ca8"),
                  ("mid (20-30 d)", (a >= 20) & (a < 30), "#d1852a"),
                  ("at the ceiling (35 d)", a >= cap - 1e-9, "#b8320f")]:
    v = s[m]
    ax[2].hist(v, bins=np.linspace(0, 2300, 60), density=True, histtype="step",
               color=c, lw=2.2, label=f"{lab}  n={m.sum():,}")
    sg = np.sqrt((v ** 2).mean() / 2); x = np.linspace(1, 2300, 300)
    ax[2].plot(x, x / sg ** 2 * np.exp(-x ** 2 / (2 * sg ** 2)), color=c, ls=":", lw=1.6)
ax[2].set_xlabel("spatial distance (um)"); ax[2].set_ylabel("density")
ax[2].set_title("C.  M3-4: the conditional law, against\na Rayleigh of the same spread (dotted)",
                fontsize=11.5)
ax[2].legend(frameon=False, fontsize=8.5)

# D: what fraction of the variance does the tree explain at all?
for SEC, c in COL.items():
    a, s, ac, sig, n, ext = store[SEC]
    tot = s.var()
    within = np.average([s[(a >= lo) & (a < hi)].var()
                         for lo, hi in zip(np.r_[0, ac[:-1]], np.r_[ac[:-1], a.max() + 1])
                         if ((a >= lo) & (a < hi)).sum() > 200],
                        weights=[((a >= lo) & (a < hi)).sum()
                                 for lo, hi in zip(np.r_[0, ac[:-1]], np.r_[ac[:-1], a.max() + 1])
                                 if ((a >= lo) & (a < hi)).sum() > 200])
    ax[3].bar(list(COL).index(SEC), 1 - within / tot, color=c, width=.6)
    ax[3].text(list(COL).index(SEC), 1 - within / tot + .004, f"{1-within/tot:.3f}",
               ha="center", fontsize=10)
ax[3].set_xticks(range(3)); ax[3].set_xticklabels(list(COL))
ax[3].set_ylabel("fraction of the variance in spatial\ndistance explained by the tree")
ax[3].set_title("D.  How much there is to work with\nat all", fontsize=11.5)

for x in ax:
    x.grid(alpha=.22); x.set_axisbelow(True)
    for sp in ("top", "right"): x.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "law.png", dpi=140, facecolor="white", bbox_inches="tight")
print("wrote law.png")
for SEC in COL:
    a, s, ac, sig, n, ext = store[SEC]
    print(f"{SEC}: sigma spans {sig.min():.0f}-{sig.max():.0f} um over ages "
          f"{ac.min():.1f}-{ac.max():.1f} d;  section extent {ext:.0f} um;  "
          f"sigma at the ceiling / sigma at the youngest = {sig[-1]/sig[0]:.2f}x")
