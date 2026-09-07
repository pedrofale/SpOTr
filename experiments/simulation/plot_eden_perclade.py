import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_csv(HERE / "eden_perclade.csv")
CON = {"(0.45, 0.45)": -0.026, "(0.35, 0.55)": 0.102,
       "(0.25, 0.65)": 0.225, "(0.15, 0.85)": 0.380}
rows = []
for mix, g in R.groupby("mix"):
    o = g[g.real].gain.mean(); v = g[~g.real].gain.values
    rows.append((CON[mix], o, v.mean(), v.std(), int((v >= o - 1e-12).sum()), len(v)))
D = pd.DataFrame(rows, columns=["con", "real", "nm", "ns", "ge", "nd"]).sort_values("con")

fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.axvspan(0.011, 0.100, color="#eb6834", alpha=.16, zorder=0)
ax.fill_between(D.con, D.nm - D.ns, D.nm + D.ns, color="0.75", alpha=.7, zorder=1,
                label="random partitions, same sizes (±1 sd)")
ax.plot(D.con, D.nm, color="0.45", ls="--", lw=1.4, zorder=2)
ax.plot(D.con, D.real, "o-", color="#2a78d6", lw=2.6, ms=9, zorder=3,
        label="planted motility groups")
for _, r in D.iterrows():
    ax.annotate(f"{int(r['ge'])}/{int(r['nd'])}", (r["con"], r["real"]), textcoords="offset points",
                xytext=(4, 11), fontsize=9,
                color="#1a5490" if r["ge"] <= 1 else "#999")
ax.axhline(0, color="0.5", lw=1)
ax.annotate("PEtracer\n0.011 – 0.100", xy=(0.055, 0.0165), ha="center", fontsize=10,
            color="#a83c10", fontweight="bold")
ax.set_xlabel("planted contrast in local tree–space concordance between the two groups")
ax.set_ylabel("two-group gain  (best − best uniform)")
ax.set_title("Per-clade α with REALISTIC heterogeneity (clades differing in motility)\n"
             "3 seeds per level, 15 random partitions each; labels give random ≥ real",
             fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=.25); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "eden_perclade.png", dpi=155)
print(D.to_string(index=False))
print("\nwrote eden_perclade.png")
