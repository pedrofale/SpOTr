"""Do the preconditions improve if the tree is cut coarser?"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "granularity.csv")
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8))
colors = dict(zip(df["dataset"].unique(), ["#444444", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]))

for ds, sub in df.groupby("dataset", sort=False):
    axs[0].plot(sub["n_groups"], sub["coherence"], "o-", label=ds, color=colors[ds])
    m = sub.dropna(subset=["mantel_r"])
    axs[1].plot(m["n_groups"], m["mantel_r"], "o-", label=ds, color=colors[ds])
    for _, r in m.iterrows():
        if r["mantel_p"] < 0.05:
            axs[1].plot(r["n_groups"], r["mantel_r"], "*", ms=16, color="gold",
                        markeredgecolor="k", zorder=5)

axs[0].axhline(0.30, ls="--", c="k", lw=1.2)
axs[0].text(0.98, 0.31, "need > 0.3", transform=axs[0].get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8)
axs[0].set_xlabel("number of clades the tree is cut into")
axs[0].set_ylabel("spatial coherence of clade labels")
axs[0].set_title("Coarsening the tree does not create territories", fontsize=11)
axs[0].legend(fontsize=8)

axs[1].axhline(0, c="k", lw=0.8)
axs[1].set_xlabel("number of clades the tree is cut into")
axs[1].set_ylabel("Mantel r (tree distance vs territory distance)")
axs[1].set_title("Tree-space agreement: never significant at any granularity\n"
                 "(B1 positive, D1 negative -- same tumour)", fontsize=11)
axs[1].legend(fontsize=8)

fig.suptitle("Does a coarser tree rescue the premise? No.", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.9])
out = HERE / "bcsa_granularity.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
