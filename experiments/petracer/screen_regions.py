"""Does a spatial region partition beat a random grouping of the same size?

Same free screen as the clades: choose alpha per group on spot-half A, apply on half B.
Groups = kmeans(k) on the cell's TRUE xy. Null = random k-way groupings, and also SPATIALLY
COHERENT random groupings (kmeans on a random rotation/jitter is not enough, so: kmeans on
random 2-D Gaussian coordinates of the same n, which gives blobs of the same size profile).
"""
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95])
rng = np.random.default_rng(0)

def gain(EA, EB, lab):
    g0 = EB[:, EA.mean(0).argmin()].mean()
    o = np.empty(len(lab))
    for k in np.unique(lab):
        s = lab == k; o[s] = EB[s, EA[s].mean(0).argmin()]
    return g0 - o.mean()

for SEC in ["M3-4", "M3-1"]:
    H = np.load(HERE / f"exact_halves_{SEC}.npy"); n = H.shape[2]
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    xy_sc = np.asarray(d["xy_sc"], float)
    print(f"\n=== {SEC} ===  {n} cells")
    for K in (2, 4, 8):
        reg = KMeans(K, n_init=10, random_state=0).fit_predict(xy_sc)
        obs, rnd, spa = [], [], []
        for rep in range(H.shape[0]):
            for A, B in [(0, 1), (1, 0)]:
                EA, EB = H[rep, A], H[rep, B]
                obs.append(gain(EA, EB, reg))
                for _ in range(40):
                    rnd.append(gain(EA, EB, rng.integers(0, K, n)))
                for _ in range(20):                      # spatially-blobby null
                    Z = rng.normal(size=(n, 2))
                    spa.append(gain(EA, EB, KMeans(K, n_init=3,
                                random_state=int(rng.integers(1e6))).fit_predict(Z)))
        o = np.mean(obs); r = np.array(rnd); s = np.array(spa)
        print(f"  k={K:>2}  spatial regions {o:>+7.1f} um   |  random {r.mean():>+6.1f} +- {r.std():.1f}"
              f"  z={(o-r.mean())/max(r.std(),1e-9):+.2f}   |  blobby-random {s.mean():>+6.1f} +- {s.std():.1f}"
              f"  z={(o-s.mean())/max(s.std(),1e-9):+.2f}")
