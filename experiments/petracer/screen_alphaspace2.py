"""Split two things the first screen conflated:
   (a) does the PARTITION transfer?   -> cluster on A, read each cluster's optimum on B
   (b) do the alpha VALUES transfer?  -> cluster on A, fit alpha on A, score on B
If (a) holds but (b) fails, the grouping is real and the estimator is noisy -- a fixable problem.
Clusters are ordered by their half-A optimum so labels are comparable across splits (kmeans
labels are arbitrary; averaging them raw mixes the groups up).
"""
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95])
rng = np.random.default_rng(0)

for SEC in ["M3-4", "M3-1"]:
    H = np.load(HERE / f"exact_halves_{SEC}.npy"); n = H.shape[2]
    aA, aB, gh, go, nl = [], [], [], [], []
    for rep in range(H.shape[0]):
        for A, B in [(0, 1), (1, 0)]:
            EA, EB = H[rep, A], H[rep, B]
            Z = (EA - EA.mean(1, keepdims=True)) / (EA.std(1, keepdims=True) + 1e-9)
            lab = KMeans(2, n_init=10, random_state=0).fit_predict(Z)
            oA = np.array([GRID[EA[lab == k].mean(0).argmin()] for k in (0, 1)])
            order = np.argsort(oA); lab = np.argsort(order)[lab]      # low-alpha group = 0
            oA = oA[order]
            oB = np.array([GRID[EB[lab == k].mean(0).argmin()] for k in (0, 1)])
            aA.append(oA); aB.append(oB)
            g0 = EB[:, EA.mean(0).argmin()].mean()
            hon = np.mean([EB[lab == k, EA[lab == k].mean(0).argmin()].mean()
                           * (lab == k).sum() for k in (0, 1)]) * 2 / n
            orc = np.mean([EB[lab == k, EB[lab == k].mean(0).argmin()].mean()
                           * (lab == k).sum() for k in (0, 1)]) * 2 / n
            gh.append(g0 - hon); go.append(g0 - orc)
            for _ in range(40):
                r = rng.integers(0, 2, n)
                orr = np.mean([EB[r == k, EB[r == k].mean(0).argmin()].mean()
                               * (r == k).sum() for k in (0, 1)]) * 2 / n
                nl.append(g0 - orr)
    aA, aB, nl = np.array(aA), np.array(aB), np.array(nl)
    print(f"\n=== {SEC} ===")
    print(f"  alpha chosen on A  (low group, high group), per split:")
    for i in range(len(aA)):
        print(f"     split {i}:  A {aA[i][0]:.2f} / {aA[i][1]:.2f}      "
              f"held-out optimum on B {aB[i][0]:.2f} / {aB[i][1]:.2f}")
    sep = (aB[:, 1] - aB[:, 0])
    print(f"  (a) PARTITION: held-out separation of the two optima "
          f"{sep.mean():+.3f} +- {sep.std():.3f}, positive in {(sep>0).sum()}/{len(sep)} splits")
    print(f"  (b) VALUES:    held-out gain using alpha fitted on A   {np.mean(gh):+7.1f} um")
    print(f"      ORACLE alpha on B, honest partition                {np.mean(go):+7.1f} um"
          f"   random 2-way, oracle alpha {nl.mean():+.1f} +- {nl.std():.1f}"
          f"   z={(np.mean(go)-nl.mean())/nl.std():+.2f}")
