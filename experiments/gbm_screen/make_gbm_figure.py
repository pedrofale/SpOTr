"""Spatial clone maps for the GBM sections that pass the coherence criterion."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screen_gbm import load_sample, cnv_profile, N_CLONES, CNV_SCORE_MIN  # noqa: E402

gpos = pd.read_csv(HERE / "gene_positions.csv")
res = pd.read_csv(HERE / "gbm_screen.csv")
show = ["GBM_ZH916bulk", "GBM_ZH916T1", "GBM_MGH258", "GBM_ZH8812bulk", "GBM_ZH8811Abulk", "GBM_ZH881inf"]

fig, axs = plt.subplots(2, 3, figsize=(15.5, 9.4))
for ax, name in zip(axs.ravel(), show):
    X, genes, xy = load_sample(HERE / "ext" / name)
    S, chrom, keep = cnv_profile(X, genes, gpos)
    xy = xy[keep]
    clone = fcluster(linkage(S, method="ward"), N_CLONES, criterion="maxclust")
    cl = np.unique(clone)
    means = np.stack([S[clone == c].mean(0) for c in cl])
    score = np.array([means[i][chrom == 7].mean() - means[i][chrom == 10].mean()
                      for i in range(len(cl))])
    mal_cl = cl[score > CNV_SCORE_MIN]
    mal = np.isin(clone, mal_cl)

    ax.scatter(xy[~mal, 1], -xy[~mal, 0], c="0.85", s=5, label="non-malignant")
    for i, c in enumerate(mal_cl):
        m = clone == c
        ax.scatter(xy[m, 1], -xy[m, 0], s=5, color=plt.get_cmap("tab10")(i % 10))
    sub = res[res["sample"] == name]
    coh = sub["coherence_malignant"].max()
    mm = sub.dropna(subset=["mantel_r"])
    lab = ""
    if len(mm):
        b = mm.loc[mm["mantel_p"].idxmin()]
        lab = f"\nMantel r={b['mantel_r']:+.2f} (p={b['mantel_p']:.3f}, {int(b['n_groups_tested'])} clades)"
    ax.set_title(f"{name.replace('GBM_','')}  —  coherence {coh:.2f}{lab}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

fig.suptitle("Glioblastoma: malignant CNV subclones form spatial territories\n"
             "(grey = non-malignant spots, colours = CNV clones; compare BCSA coherence 0.03–0.17)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.91])
out = HERE / "gbm_clone_maps.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
