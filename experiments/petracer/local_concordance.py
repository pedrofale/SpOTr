"""Does LOCAL tree-space concordance have level and spread across clades?

The global statistic (Pearson corr of tree vs spatial distance over all pairs) is what
square-loss GW consumes, and it is 0.16-0.24 on PEtracer -- inside the dead zone. But the
LOCAL statistic is completely different: a cell's nearest relative in the tree sits at 0.23x
the random spatial distance. The tree carries neighbourhood information and no metric
information.

A Laplacian-style term uses only the local part. So the question that decides whether changing
the cost could rescue per-clade alpha is: does the LOCAL concordance vary ACROSS CLADES?
Per-clade alpha needs SPREAD, not just level.

Statistic per cell: mean spatial distance to its k nearest tree-relatives, divided by the mean
spatial distance to k random cells. 0 = perfect local concordance, 1 = none.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster
HERE = Path(__file__).resolve().parent
K = 5

def local_score(CT, xy, K=5, restrict=None):
    """Mean spatial distance to the K nearest TREE relatives / mean to a random cell.

    restrict: if given (a group label per cell), neighbours and the random baseline are both
    taken WITHIN the cell's group. Without this the simulation's two branches overlap in tree
    space and a branch-A cell's 'tree neighbours' include spatially-random branch-B cells,
    which inflates the score and makes the two datasets incomparable.
    """
    CT = CT.copy(); np.fill_diagonal(CT, np.inf)
    CS = squareform(pdist(xy))
    out = np.empty(len(CT))
    groups = [np.arange(len(CT))] if restrict is None else \
             [np.where(restrict == g)[0] for g in np.unique(restrict)]
    for idx in groups:
        sub_t = CT[np.ix_(idx, idx)]; sub_s = CS[np.ix_(idx, idx)]
        k = min(K, len(idx) - 1)
        nb = np.argpartition(sub_t, k, axis=1)[:, :k]
        out[idx] = np.take_along_axis(sub_s, nb, 1).mean(1) / (sub_s.sum(1) / (len(idx) - 1))
    return out

rows = []
for SEC in ["M3-4", "M3-1", "M3-2"]:
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1
    s = local_score(CT, xy, K, restrict=br)
    big = [c for c in np.unique(cc) if (cc == c).sum() >= 30]
    per = np.array([s[cc == c].mean() for c in big])
    print(f"\n=== {SEC} ===  local score (0 = tree neighbours are spatial neighbours, 1 = none)")
    print(f"  all cells {s.mean():.3f}   |   branch A {s[br==0].mean():.3f}   "
          f"branch B {s[br==1].mean():.3f}   |  branch difference {abs(s[br==0].mean()-s[br==1].mean()):.3f}")
    print(f"  across {len(big)} clades: mean {per.mean():.3f}  sd {per.std():.3f}  "
          f"range {per.min():.3f}-{per.max():.3f}   CV {per.std()/per.mean():.3f}")
    rows.append(dict(data=SEC, mean=s.mean(), brdiff=abs(s[br==0].mean()-s[br==1].mean()),
                     clade_sd=per.std(), clade_cv=per.std()/per.mean()))

sys.path.insert(0, str(HERE.parent / "simulation"))
from sim_varalpha_branches import simulate
for tn in [0.10, 0.35]:
    sim = simulate(0, tree_noise=tn)
    b = sim["branch"]
    s = local_score(sim["C_tree"], sim["xy_sc"], K, restrict=b)
    print(f"\n=== SIMULATION tree_noise={tn} ===")
    print(f"  all cells {s.mean():.3f}   |   branch A {s[b==0].mean():.3f}   "
          f"branch B {s[b==1].mean():.3f}   |  branch difference {abs(s[b==0].mean()-s[b==1].mean()):.3f}")
    rows.append(dict(data=f"sim tn={tn}", mean=s.mean(),
                     brdiff=abs(s[b==0].mean()-s[b==1].mean()), clade_sd=np.nan, clade_cv=np.nan))
pd.DataFrame(rows).to_csv(HERE / "local_concordance.csv", index=False)
print("\n" + "="*70)
print("The branch DIFFERENCE is what per-clade alpha needs. Compare PEtracer against the")
print("simulation at tree_noise=0.10 (where the method recovered the effect) and 0.35 (where")
print("it found nothing).")
