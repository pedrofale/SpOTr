"""Are the clusters real CNV clones, or expression programs smoothed along the genome?

GBM has canonical whole-chromosome events: chr7 gain and chr10 loss in essentially every
IDH-wt tumour. If the malignant clusters are CNV-driven they must show them; if they are
transcriptional states they will not.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screen_gbm import load_sample, cnv_profile, N_CLONES  # noqa: E402

gpos = pd.read_csv(HERE / "gene_positions.csv")
for name in ["GBM_ZH8811Abulk", "GBM_ZH916bulk", "GBM_ZH916T1", "GBM_ZH881inf"]:
    got = load_sample(HERE / "ext" / name)
    if got is None:
        continue
    X, genes, xy = got
    S, chrom, keep = cnv_profile(X, genes, gpos)
    clone = fcluster(linkage(S, method="ward"), N_CLONES, criterion="maxclust")
    cl = np.unique(clone)
    means = np.stack([S[clone == c].mean(0) for c in cl])
    mag = np.abs(means).mean(1)
    normal = cl[mag.argmin()]

    print(f"\n=== {name}   (normal-like cluster = {normal})")
    hdr = "clust    n  " + "".join(f"{c:>6}" for c in [1, 7, 9, 10, 19, 20])
    print(hdr)
    for i, c in enumerate(cl):
        vals = [means[i][chrom == ch].mean() for ch in [1, 7, 9, 10, 19, 20]]
        tag = "  <- normal-like" if c == normal else ""
        print(f"{c:>5} {int((clone==c).sum()):>5}  " +
              "".join(f"{v:>+6.2f}" for v in vals) + tag)
    mal = means[cl != normal]
    print(f"  malignant mean chr7 {mal[:, chrom==7].mean():+.3f}   "
          f"chr10 {mal[:, chrom==10].mean():+.3f}   "
          f"(expect chr7 > 0, chr10 < 0 for IDH-wt GBM)")
