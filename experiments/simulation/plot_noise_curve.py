import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_csv(HERE / "sim_noise_curve.csv")
PET = {"M3-4": (0.216, 0.200), "M3-1": (0.240, 0.155), "M3-2": (0.218, 0.161)}
lo = min(min(v) for v in PET.values()); hi = max(max(v) for v in PET.values())

print(f"{'tree_noise':>11}{'corr':>8}{'real gain':>12}{'random gain':>22}"
      f"{'draws >= real':>15}")
print("-" * 68)
rows = []
for tn, g in R.groupby("tree_noise"):
    o = g[g.real].gain.mean(); v = g[~g.real].gain.values; c = g["corr"].mean()
    ge = int((v >= o - 1e-12).sum())
    rows.append((tn, c, o, v.mean(), v.std(), ge, len(v)))
    print(f"{tn:>11.2f}{c:>8.3f}{o:>+12.4f}{v.mean():>+13.4f} ± {v.std():.4f}"
          f"{ge:>10}/{len(v)}")
D = pd.DataFrame(rows, columns=["tn", "corr", "real", "nullm", "nulls", "ge", "nd"])

fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.axvspan(lo, hi, color="#eb6834", alpha=.16, zorder=0)
ax.fill_between(D["corr"], D.nullm - D.nulls, D.nullm + D.nulls, color="0.75",
                alpha=.7, zorder=1, label="random splits, same sizes (±1 sd)")
ax.plot(D["corr"], D.nullm, color="0.45", ls="--", lw=1.4, zorder=2)
ax.plot(D["corr"], D.real, "o-", color="#2a78d6", lw=2.6, ms=8, zorder=3,
        label="true branch split")
for _, r in D.iterrows():
    if r["real"] > 1e-4:
        ax.annotate(f"{int(r['ge'])}/{int(r['nd'])} random ≥ real", (r["corr"], r["real"]),
                    textcoords="offset points", xytext=(6, 9), fontsize=8, color="#1a5490")
ax.axhline(0, color="0.5", lw=1)
ax.set_xlabel("within-branch corr(tree distance, spatial distance)")
ax.set_ylabel("two-branch gain  (best − best uniform)")
ax.annotate("PEtracer\nlives here\n0.16 – 0.24", xy=((lo+hi)/2, 0.085), ha="center",
            fontsize=10, color="#a83c10", fontweight="bold")
ax.set_title("How much tree–space concordance does per-clade α need?\n"
             "3 seeds per level, 5 random splits each; the effect is gone by 0.38",
             fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=.25); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "noise_curve.png", dpi=155)
print("\nwrote noise_curve.png")
