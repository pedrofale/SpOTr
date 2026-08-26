"""Are the 'clones' distinct subclones, or just a tumour-content gradient?

A purity gradient makes every chromosome move together, so clone profiles are near-perfect
scalar multiples of each other (r ~ 1). Genuine subclones differ on SPECIFIC chromosomes,
which lowers the correlation. This is the same test that disqualified the B178 scRNA-seq
split, applied to the Visium clones the GBM screen found.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from itertools import combinations

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screen_gbm import load_sample, cnv_profile, N_CLONES, CNV_SCORE_MIN  # noqa: E402

gpos = pd.read_csv(HERE / "gene_positions.csv")
rows = []
for name in ["GBM_ZH916bulk", "GBM_ZH916T1", "GBM_MGH258", "GBM_ZH8812bulk",
             "GBM_ZH8811Abulk", "GBM_ZH881T1"]:
    got = load_sample(HERE / "ext" / name)
    if got is None:
        continue
    X, genes, xy = got
    S, chrom, keep = cnv_profile(X, genes, gpos)
    clone = fcluster(linkage(S, method="ward"), N_CLONES, criterion="maxclust")
    cl = np.unique(clone)
    means = np.stack([S[clone == c].mean(0) for c in cl])
    score = np.array([means[i][chrom == 7].mean() - means[i][chrom == 10].mean()
                      for i in range(len(cl))])
    mal = cl[score > CNV_SCORE_MIN]
    M = means[np.isin(cl, mal)]
    if len(M) < 2:
        continue
    rs = [np.corrcoef(M[i], M[j])[0, 1] for i, j in combinations(range(len(M)), 2)]
    rows.append(dict(sample=name.replace("GBM_", ""), n_clones=len(M),
                     mean_pairwise_r=float(np.mean(rs)), max_r=float(np.max(rs)),
                     min_r=float(np.min(rs))))

df = pd.DataFrame(rows)
df["verdict"] = np.where(df["mean_pairwise_r"] > 0.8, "purity gradient",
                np.where(df["mean_pairwise_r"] > 0.5, "mixed", "distinct events"))
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nB178 scRNA-seq for reference: mean pairwise r = +0.979  -> purity gradient")
df.to_csv(HERE / "purity_test.csv", index=False)
