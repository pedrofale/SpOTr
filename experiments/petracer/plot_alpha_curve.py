import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
P = pd.read_csv(HERE / "fair_sweep.csv")
P = P[(P.section == "M3-4") & (P.fair)].sort_values("alpha").reset_index(drop=True)
FLOOR = 0.0977
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))

a = ax[0]
a.plot(P.alpha, P.err, "o-", color="#2a78d6", lw=2.6, ms=7)
b = int(P.err.idxmin())
a.scatter([P.alpha[b]], [P.err[b]], s=230, facecolors="none", edgecolors="#1a7a3a", lw=2.6, zorder=5)
a.annotate(f"best: {P.err[b]:.0f} µm at α={P.alpha[b]:g}", (P.alpha[b], P.err[b]),
           textcoords="offset points", xytext=(10, 22), fontsize=10, color="#1a7a3a",
           fontweight="bold")
a.annotate(f"no tree: {P.err[0]:.0f} µm", (0, P.err[0]), textcoords="offset points",
           xytext=(14, -4), fontsize=10, color="#a83c10")
a.set_xlabel("α   (0 = expression only, 1 = tree only)")
a.set_ylabel("median placement error (µm)")
a.set_title("How far off is each cell?", fontsize=11.5)
a.grid(alpha=.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)

a = ax[1]
a.plot(P.alpha, P.soft, "o-", color="#eb6834", lw=2.6, ms=7)
a.axhline(FLOOR, color="0.5", ls=":", lw=1.8)
a.annotate("random-spot floor", (0.62, FLOOR), textcoords="offset points", xytext=(0, 6),
           fontsize=9, color="0.4")
bs = int(P.soft.idxmax())
a.scatter([P.alpha[bs]], [P.soft[bs]], s=230, facecolors="none", edgecolors="#a83c10",
          lw=2.6, zorder=5)
a.annotate(f"α={P.alpha[bs]:g}", (P.alpha[bs], P.soft[bs]), textcoords="offset points",
           xytext=(2, -22), fontsize=10, color="#a83c10", fontweight="bold")
a.scatter([0.75], [float(P[P.alpha == 0.75].soft.iloc[0])], s=230, facecolors="none",
          edgecolors="#1a7a3a", lw=2.6, zorder=5)
a.set_xlabel("α"); a.set_ylabel("soft accuracy (clade composition)")
a.set_title("Does it land among the right relatives?", fontsize=11.5)
a.grid(alpha=.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)

fig.suptitle("PEtracer M3-4: adding the tree cuts placement error by a third.  "
             "The two metrics pick different α —\ngreen rings mark α=0.60, where both agree; "
             "both metrics now agree.", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(HERE / "alpha_curve.png", dpi=150, facecolor="white")
print("wrote alpha_curve.png")
