import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_csv(HERE / "branching_sweep.csv")
R["sat"] = R.sat.fillna("None").astype(str)
SE = sorted(R.sigma_expr.unique())
fig, ax = plt.subplots(1, len(SE), figsize=(14.4, 3.9), sharex=True)
for k, se in enumerate(SE):
    a = ax[k]
    for sat, c, ls, lab in [("0.5", "#eb6834", "-", "saturated (PEtracer-like)"),
                            ("None", "#2a78d6", "--", "unsaturated")]:
        g = R[(R.sigma_expr == se) & (R.sat == sat)].groupby("alpha").acc.agg(["mean", "std"])
        a.plot(g.index, g["mean"] / g["mean"].iloc[0], ls, color=c, lw=2.2, ms=5,
               marker="o", label=lab)
    a.axhline(1.0, color="0.5", lw=1)
    a0 = R[(R.sigma_expr == se) & (R.sat == "0.5") & (R.alpha == 0)].acc.mean()
    a.set_title(f"expression noise {se:g}\nα=0 at {a0*128:.0f}× chance", fontsize=10)
    a.set_xlabel("α")
    a.grid(alpha=.25); a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
    if k == 0:
        a.set_ylabel("accuracy relative to α=0")
        a.legend(frameon=False, fontsize=8)
ax[-1].annotate("PEtracer reaches 1.82×\non its own metric",
                xy=(0.5, 1.35), fontsize=8.5, color="#a83c10", ha="center")
fig.suptitle("Square-loss FGW on a REALISTIC lineage tree (spatial branching process): the tree "
             "adds little,\nand only where expression is weak.  Saturation is not the cause.",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(HERE / "branching_sweep.png", dpi=155)
print("wrote branching_sweep.png")
