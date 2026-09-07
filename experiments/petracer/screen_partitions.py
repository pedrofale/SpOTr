"""Does the per-cell-error screen predict the gain a REAL mixed-alpha solve delivers?

exact_halves_*.npy holds E[split, half, cell, alpha] = placement error (um) from UNIFORM-alpha
solves. Reading 'this cell prefers alpha=x' off those solves is how we concluded that different
cells want different alphas. But the coupling is global: if cell i had actually been given a
different alpha, every other cell's assignment would move too. So the screen is an ORACLE UPPER
BOUND, not a prediction. The two-branch experiment gives us the ground truth to calibrate it
against: +0.0000 (M3-4) and +0.0302 (M3-1) on the full grid.
"""
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95])
rng = np.random.default_rng(0)
NDRAW = 200

def gain(EA, EB, lab):
    """Choose alpha per group on half A, apply on half B. Return um saved vs one global alpha."""
    g_glob = EB[:, EA.mean(0).argmin()].mean()
    out = np.empty(len(lab))
    for k in np.unique(lab):
        s = lab == k
        out[s] = EB[s, EA[s].mean(0).argmin()]
    return g_glob - out.mean(), g_glob

for SEC in ["M3-4", "M3-1"]:
    H = np.load(HERE / f"exact_halves_{SEC}.npy")
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    cc = np.asarray(d["cell_clade"]).astype(str)
    CT = np.asarray(d["C_tree"], float)
    clades = list(np.asarray(d["clades"]).astype(str))
    cidx = np.array([clades.index(c) for c in cc])
    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1
    n = len(cc)
    print(f"\n=== {SEC} ===  {n} cells")

    parts = {"2 branches": br, "18 clades": cidx}
    res = {k: [] for k in parts}
    res["per-cell oracle"] = []
    rnd = {2: [], 18: []}
    glob = []
    for rep in range(H.shape[0]):
        for A, B in [(0, 1), (1, 0)]:
            EA, EB = H[rep, A], H[rep, B]
            for k, lab in parts.items():
                g, gg = gain(EA, EB, lab); res[k].append(g); glob.append(gg)
            res["per-cell oracle"].append(gain(EA, EB, np.arange(n))[0])
            for K in (2, 18):
                for _ in range(NDRAW // (H.shape[0] * 2)):
                    rnd[K].append(gain(EA, EB, rng.integers(0, K, n))[0])

    g0 = np.mean(glob)
    print(f"  one global alpha, chosen on A, scored on B: {g0:.1f} um")
    print(f"  {'partition':<20}{'um saved':>10}{'   random same #groups':>24}")
    print("  " + "-"*56)
    for k in ["2 branches", "18 clades", "per-cell oracle"]:
        v = np.mean(res[k])
        if k == "per-cell oracle":
            extra = ""
        else:
            K = 2 if k == "2 branches" else 18
            r = np.array(rnd[K])
            extra = f"{r.mean():>+9.1f} +- {r.std():.1f}   z={(v-r.mean())/max(r.std(),1e-9):+.2f}"
        print(f"  {k:<20}{v:>+10.1f}   {extra}")
