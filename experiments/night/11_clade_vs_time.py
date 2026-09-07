"""Is a CLADE-level decomposition anything more than a DIVERGENCE-TIME one?

The clades are subtrees of an ultrametric tree, so "same clade" is close to "patristic distance
below a threshold".  If it is exactly that, then a free transform of divergence time already
contains every within/between-clade weighting, and there is nothing left for a per-clade
parameter to add.  Three measurements:

  1. how well tree distance alone predicts same-clade membership (AUC)
  2. whether the within-clade tree->space relationship is a stable property of the CLADE:
     does it replicate between halves of one section, and between sections?
  3. how much of the between-clade variation in that relationship is explained by clade SIZE
     and by how much of the clade the section happens to contain
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
HERE = Path(__file__).resolve().parent
SECS = ["M3-4", "M3-1", "M3-2"]

print("1.  Does patristic distance already encode clade membership?")
for SEC in SECS:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    _, cl = np.unique(cc, return_inverse=True)
    iu = np.triu_indices(len(cl), 1)
    t = CT[iu]; same = (cl[iu[0]] == cl[iu[1]])
    o = np.argsort(t)
    r = np.empty(len(t)); r[o] = np.arange(len(t))
    n1 = same.sum(); n0 = len(same) - n1
    auc = 1 - (r[same].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)   # small t -> same clade
    thr = np.quantile(t, same.mean())
    print(f"   {SEC}: AUC {auc:.3f}   |  a pure threshold on tree distance at the same "
          f"prevalence recovers {np.mean(same[t <= thr]) * 100:.0f}% of same-clade pairs "
          f"(chance {same.mean()*100:.1f}%)")

print("\n2.  Is the WITHIN-CLADE tree-space correlation a stable property of the clade?")
rng = np.random.default_rng(0)
per = {}
for SEC in SECS:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    S = squareform(pdist(xy))
    n = len(cc)
    for rep in range(2):                      # two halves of this section
        h = rng.permutation(n)
        idx = h[:n // 2] if rep == 0 else h[n // 2:]
        for c in np.unique(cc):
            j = idx[cc[idx] == c]
            if len(j) < 30:
                continue
            i2 = np.triu_indices(len(j), 1)
            t = CT[np.ix_(j, j)][i2]; s = S[np.ix_(j, j)][i2]
            if t.std() < 1e-9:
                continue
            per[(SEC, rep, c)] = pearsonr(t, s).statistic

W = pd.Series(per).rename("corr").reset_index()
W.columns = ["section", "half", "clade", "corr"]
P = W.pivot_table(index=["section", "clade"], columns="half", values="corr").dropna()
print(f"   between HALVES of the same section (n={len(P)} clade-halves): "
      f"pearson {pearsonr(P[0], P[1]).statistic:+.3f}   spearman "
      f"{spearmanr(P[0], P[1]).statistic:+.3f}")
Q = W.groupby(["section", "clade"])["corr"].mean().unstack(0).dropna()
for i in range(3):
    for j in range(i + 1, 3):
        a, b = SECS[i], SECS[j]
        print(f"   between SECTIONS {a} vs {b} (n={len(Q)} clades): "
              f"pearson {pearsonr(Q[a], Q[b]).statistic:+.3f}   spearman "
              f"{spearmanr(Q[a], Q[b]).statistic:+.3f}")

print("\n3.  What predicts a clade's within-clade concordance?")
R = pd.read_csv(HERE / "perclade_relationship.csv")
for SEC in SECS:
    g = R[R.section == SEC]
    print(f"   {SEC}: vs clade size in this section  rho {spearmanr(g.n, g['corr']).statistic:+.3f}"
          f"   |  vs fraction of its pairs at the ceiling  rho "
          f"{spearmanr(g.frac_cap, g['corr']).statistic:+.3f}"
          f"   |  vs spatial cohesion  rho {spearmanr(g.cohesion, g['corr']).statistic:+.3f}")
