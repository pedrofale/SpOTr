import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
HERE = Path(__file__).resolve().parent
RA = pd.read_csv(HERE / "petracer_region_alpha.csv")
SECS = ["M3-1", "M3-2", "M3-4"]
CMAP = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.7))
for ax, s in zip(axes, SECS):
    d = np.load(HERE / f"petracer_{s}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    lab = KMeans(4, n_init=10, random_state=0).fit_predict(xy_st)
    sub = RA[RA.section == s].set_index("region")
    for r in range(4):
        m = lab == r
        ax.scatter(xy_st[m, 0], xy_st[m, 1], s=42, color=CMAP(r), alpha=.9,
                   edgecolors="none")
        cx, cy = xy_st[m, 0].mean(), xy_st[m, 1].mean()
        if r in sub.index:
            ax.text(cx, cy, f"α*={sub.loc[r,'best_alpha']:g}\n{int(sub.loc[r,'n_cells'])} cells",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.32", fc="0.15", ec="none", alpha=.82))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    gl = sub.acc_best.notna()
    ax.set_title(f"{s}   section-wide best α = "
                 f"{RA[RA.section==s].attrs.get('g','')}", fontsize=10)
for ax, s, g in zip(axes, SECS, ["0.2", "0.2", "0.6"]):
    ax.set_title(f"{s}    one global α would use {g}", fontsize=10.5)
fig.suptitle("What a “region” is: k-means (k=4) on the spot coordinates.  "
             "α* = the α that scores best for the cells whose TRUE position falls in that blob.",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(HERE / "regions.png", dpi=155)
print("wrote regions.png")
