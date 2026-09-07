"""Are the two geometries comparable? Measure E[spatial | tree] over the FULL tree range."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
HERE = Path(__file__).resolve().parent
COL = {"M3-4": "#2a78d6", "M3-1": "#e0a53c", "M3-2": "#1a6b3c"}
fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.5))
rng = np.random.default_rng(0)
stats = {}
for SEC in COL:
    d = np.load(HERE / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    iu = np.triu_indices(len(xy), 1)
    t = CT[iu]; t = t / t.mean(); s = pdist(xy); s = s / s.mean()
    E = np.linspace(t.min(), t.max(), 30)
    mid, ms, frac = [], [], []
    for k in range(len(E)-1):
        m = (t >= E[k]) & (t < E[k+1] if k < len(E)-2 else t <= E[k+1])
        if m.sum() < 100: continue
        mid.append(0.5*(E[k]+E[k+1])); ms.append(s[m].mean()); frac.append(m.mean())
    mid, ms, frac = map(np.array, (mid, ms, frac))
    stats[SEC] = (t, s, mid, ms, frac)
    ax[0].plot(mid, ms, "o-", color=COL[SEC], lw=2.2, ms=5, label=SEC)
    ax[1].bar(mid, frac, width=(mid[1]-mid[0])*0.9, color=COL[SEC], alpha=.45,
              label=SEC, edgecolor="none")
ax[0].plot([0, 1.15], [0, 1.15], "k--", lw=1.7, label="what GW assumes (y = x)")
ax[0].set_xlabel("tree distance, normalised  (= divergence time)")
ax[0].set_ylabel("mean spatial distance, normalised")
ax[0].set_title("Shape is linear, but the SLOPE is 2.4\nGW assumes 1", fontsize=11)
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
ax[1].set_xlabel("tree distance, normalised"); ax[1].set_ylabel("fraction of all pairs")
ax[1].set_title("…and almost all the mass sits\nat the far end", fontsize=11)
ax[1].legend(frameon=False, fontsize=8.5)
t, s, mid, ms, frac = stats["M3-4"]
sub = rng.choice(len(t), 80000, replace=False)
ax[2].scatter(t[sub], s[sub], s=1.5, alpha=.04, c="#2a78d6", edgecolors="none")
ax[2].plot(mid, ms, "-", color="#a83c10", lw=3, label="E[space | tree]")
ax[2].plot([0, 1.15], [0, 1.15], "k--", lw=1.7, label="y = x")
ax[2].set_xlabel("tree distance, normalised"); ax[2].set_ylabel("spatial distance, normalised")
ax[2].set_title("M3-4, every pair", fontsize=11)
ax[2].legend(frameon=False, fontsize=8.5, loc="upper left")
for a in ax:
    a.grid(alpha=.25); a.set_axisbelow(True); a.spines[["top","right"]].set_visible(False)
fig.suptitle("Mean-normalisation does NOT make the geometries comparable: the fitted relation is "
             "space = 2.36 x tree - 1.36, where GW assumes space = tree.", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(HERE / "link_function.png", dpi=150, facecolor="white")
print(f"{'section':<8}{'tree spans':>16}{'E[space] spans':>18}{'ratio of ranges':>17}{'corr':>8}")
print("-" * 68)
for SEC in COL:
    t, s, mid, ms, frac = stats[SEC]
    tr = t.max()/max(t.min(), 1e-9); sr = ms.max()/ms.min()
    print(f"{SEC:<8}{f'{t.min():.3f}-{t.max():.2f}':>16}{f'{ms.min():.2f}-{ms.max():.2f}':>18}"
          f"{f'{tr:.0f}x vs {sr:.1f}x':>17}{np.corrcoef(t, s)[0,1]:>8.3f}")
print("\nwrote link_function.png")
