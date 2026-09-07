"""CLAIM (b), asked as a measurement rather than as a fitted parameter.

Does the tree->space relationship DIFFER BETWEEN CLADES, and does the difference REPLICATE across
sections?  Only if it does can a per-clade weight ever transfer.

For each clade tau, using only the pairs of cells that are both in tau:
    corr_tau  = corr(patristic distance, spatial distance)
    coh_tau   = mean within-clade spatial distance / mean over all pairs   (spatial cohesion)
Nulls: random groupings of cells matched in size, so a clade cannot look special merely for
being small.  Then correlate each statistic across the three sections -- the clades are the same
18 objects in all three, since all three sections come from one tumour and one phylogeny.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

SECS = ["M3-4", "M3-1", "M3-2"]
rng = np.random.default_rng(0)
recs = []
D = {}
for SEC in SECS:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    names, clade = np.unique(cc, return_inverse=True)
    S = squareform(pdist(xy))
    allmean = pdist(xy).mean()
    D[SEC] = dict(CT=CT, S=S, clade=clade, names=names, allmean=allmean)
    for k, nm in enumerate(names):
        idx = np.where(clade == k)[0]
        if len(idx) < 25:
            continue
        i2 = np.triu_indices(len(idx), 1)
        t = CT[np.ix_(idx, idx)][i2]; s = S[np.ix_(idx, idx)][i2]
        if t.std() < 1e-9:
            continue
        # size-matched random grouping null
        nulls = []
        for _ in range(60):
            j = rng.choice(len(clade), len(idx), replace=False)
            j2 = np.triu_indices(len(j), 1)
            tt = CT[np.ix_(j, j)][j2]; ss = S[np.ix_(j, j)][j2]
            if tt.std() > 1e-9:
                nulls.append(pearsonr(tt, ss).statistic)
        r = pearsonr(t, s).statistic
        recs.append(dict(section=SEC, clade=nm, n=len(idx), corr=r,
                         null_mean=np.mean(nulls), null_sd=np.std(nulls),
                         z=(r - np.mean(nulls)) / max(np.std(nulls), 1e-9),
                         cohesion=s.mean() / allmean,
                         frac_cap=float((t >= CT.max() - 1e-9).mean())))

R = pd.DataFrame(recs); R.to_csv(HERE / "perclade_relationship.csv", index=False)
print("WITHIN-CLADE tree-space correlation, per section")
print(R.pivot_table(index="clade", columns="section", values="corr").round(3).to_string())
print("\nclade sizes")
print(R.pivot_table(index="clade", columns="section", values="n").to_string())

print("\nDoes the per-clade value REPLICATE across sections?")
for stat in ["corr", "cohesion"]:
    P = R.pivot_table(index="clade", columns="section", values=stat).dropna()
    print(f"  {stat}: n={len(P)} clades present in all three")
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = SECS[i], SECS[j]
            print(f"     {a} vs {b}:  pearson {pearsonr(P[a], P[b]).statistic:+.3f}"
                  f"   spearman {spearmanr(P[a], P[b]).statistic:+.3f}")

print("\nIs any clade's correlation different from a size-matched random group of cells?")
print(R.pivot_table(index="clade", columns="section", values="z").round(2).to_string())
print(f"\n|z| > 2 in: {(R.z.abs() > 2).sum()} of {len(R)} clade-section combinations")
print(f"spread of within-clade correlations per section:")
for SEC in SECS:
    g = R[R.section == SEC]
    print(f"   {SEC}: {g["corr"].min():+.3f} to {g["corr"].max():+.3f}   sd {g["corr"].std():.3f}   "
          f"null sd {g.null_sd.mean():.3f}")
