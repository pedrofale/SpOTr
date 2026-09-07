"""Cluster cells in ALPHA SPACE, using ground truth. Does a 2-group alpha beat one global alpha?

Each cell has a curve E_i(alpha) -- its placement error across the 9 alphas -- from uniform-alpha
exact solves. z-score each curve across alpha so we cluster on SHAPE (which alpha the cell wants),
not on how well-placed the cell is overall. Then kmeans, fit alpha per cluster, evaluate.

Two versions, and the difference between them is the whole point:
  CIRCULAR : cluster on half B, fit on half B, score on half B     <- absolute ceiling
  HONEST   : cluster on half A, fit on half A, score on half B     <- does it transfer?
"""
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95])
rng = np.random.default_rng(0)

def shape(E):
    Z = (E - E.mean(1, keepdims=True)) / (E.std(1, keepdims=True) + 1e-9)
    return Z

def apply(Efit, Escore, lab):
    o = np.empty(len(lab))
    for k in np.unique(lab):
        s = lab == k
        o[s] = Escore[s, Efit[s].mean(0).argmin()]
    return o.mean()

for SEC in ["M3-4", "M3-1"]:
    H = np.load(HERE / f"exact_halves_{SEC}.npy"); n = H.shape[2]
    print(f"\n=== {SEC} ===  {n} cells")
    for K in (2, 3, 5):
        circ, hon, glob, nul, sizes, alphas = [], [], [], [], [], []
        for rep in range(H.shape[0]):
            for A, B in [(0, 1), (1, 0)]:
                EA, EB = H[rep, A], H[rep, B]
                g = EB[:, EA.mean(0).argmin()].mean(); glob.append(g)
                lc = KMeans(K, n_init=10, random_state=0).fit_predict(shape(EB))
                circ.append(g - apply(EB, EB, lc))
                lh = KMeans(K, n_init=10, random_state=0).fit_predict(shape(EA))
                hon.append(g - apply(EA, EB, lh))
                sizes.append(np.bincount(lh, minlength=K))
                alphas.append([GRID[EA[lh == k].mean(0).argmin()] for k in range(K)])
                for _ in range(40):
                    nul.append(g - apply(EA, EB, rng.integers(0, K, n)))
        nl = np.array(nul)
        print(f"  k={K}   global {np.mean(glob):6.1f} um")
        print(f"        circular ceiling (cluster+fit+score on the same half) {np.mean(circ):+7.1f} um")
        print(f"        HONEST  (cluster+fit on A, score on B)                {np.mean(hon):+7.1f} um"
              f"   random {nl.mean():+.1f} +- {nl.std():.1f}   z={(np.mean(hon)-nl.mean())/nl.std():+.2f}")
        print(f"        group sizes {np.mean(sizes,0).round(0).astype(int).tolist()}"
              f"   chosen alphas {np.mean(alphas,0).round(2).tolist()}")
