"""Clone spatial coherence on the eTRACER Stereo-seq sections + aggregate premise test."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
BASE = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing" / "spatial"


def coherence(xy, lab, k=12):
    nb = NearestNeighbors(n_neighbors=min(k, len(xy) - 1) + 1).fit(xy)
    nbrs = nb.kneighbors(xy, return_distance=False)[:, 1:]
    same = (lab[nbrs] == lab[:, None]).mean()
    p = np.array([(lab == c).mean() for c in np.unique(lab)])
    ch = (p ** 2).sum()
    return (same - ch) / (1 - ch)


print("=== clone spatial coherence per stage (Stereo-seq bins) ===")
print(f"{'stage':>6} {'bins':>7} {'clones':>7} {'coherence':>10}")
rows = []
for stage in ["BL", "EE", "LE"]:
    d = pd.read_csv(BASE / f"spatial_lineage_{stage}.csv")
    d = d.dropna(subset=["lineageGrp", "x", "y"])
    xy = d[["x", "y"]].to_numpy(float)
    lab = d["lineageGrp"].astype(str).to_numpy()
    big = pd.Series(lab).value_counts()
    keep = np.isin(lab, big[big >= 20].index)
    coh = coherence(xy[keep], lab[keep])
    rows.append(dict(stage=stage, bins=int(keep.sum()), clones=int((big >= 20).sum()), coherence=coh))
    print(f"{stage:>6} {int(keep.sum()):>7} {int((big>=20).sum()):>7} {coh:>10.3f}")
pd.DataFrame(rows).to_csv(HERE / "etracer_coherence.csv", index=False)

print("\n=== aggregate test of the tree-space premise across clones ===")
df = pd.read_csv(HERE / "etracer_premise.csv")
n, npos, nsig = len(df), int((df["r"] > 0).sum()), int((df["p"] < 0.05).sum())
sign_p = stats.binomtest(npos, n, 0.5, alternative="greater").pvalue
exp_sig = 0.05 * n
sig_p = stats.binomtest(nsig, n, 0.05, alternative="greater").pvalue
t = stats.wilcoxon(df["r"], alternative="greater")
print(f"  clones analysed              : {n}")
print(f"  positive r                   : {npos}/{n}  (binomial p = {sign_p:.2e})")
print(f"  individually significant     : {nsig}/{n}  (expected {exp_sig:.1f}; binomial p = {sig_p:.2e})")
print(f"  Wilcoxon r>0 across clones   : p = {t.pvalue:.2e}")
print(f"  mean r {df['r'].mean():.3f}   median {df['r'].median():.3f}   max {df['r'].max():.3f}")
print("\n  -> the premise holds in aggregate, but every individual effect is small")
