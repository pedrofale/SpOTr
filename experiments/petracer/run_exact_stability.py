"""Is the preferred alpha stable, now that the solver is exact?

Exact replication of yesterday's stability battery, with ot.gromov.fused_gromov_wasserstein in
place of the broken entropic solver. Yesterday (broken kernel):
    per-CELL  best-alpha test-retest r:  M3-4 +0.142,  M3-1 -0.039
    per-CLADE optimal-alpha reliability: r = +0.426 vs size-matched random null +0.143 +- 0.277,
                                          p = 0.150 (NOT significant)
Same protocol: split the SPOTS into disjoint halves, solve each independently, derive each
cell's and each clade's preferred alpha from placement error, correlate across halves, and
control against >=100 size-matched RANDOM groupings.
"""
import numpy as np, pandas as pd, ot, time
from pathlib import Path
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95])
NSPLIT, NDRAW = 3, 300
rng = np.random.default_rng(0)

def solve(M, C1, C2, p, q, a):
    return ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun="square_loss",
                                              alpha=a, max_iter=200, tol_rel=1e-9, tol_abs=1e-9)

for SEC in ["M3-4", "M3-1"]:
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    X_sc, X_st = np.asarray(d["X_sc"], np.float64), np.asarray(d["X_st"], np.float64)
    CT = np.asarray(d["C_tree"], np.float64); CS = np.asarray(d["C_space"], np.float64)
    xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    cidx = np.array([clades.index(c) for c in cc])
    n, m = len(cc), X_st.shape[0]
    Mfull = ot.dist(X_sc, X_st); Mfull = Mfull / Mfull.max()
    C1 = CT / CT.mean()
    keep = [k for k in range(len(clades)) if (cidx == k).sum() >= 30]
    sizes = [int((cidx == k).sum()) for k in keep]
    print(f"\n=== {SEC} ===  {n} cells, {m} spots, {len(keep)} clades with >=30 cells", flush=True)

    def errors(cols):
        Ms = Mfull[:, cols]; C2 = CS[np.ix_(cols, cols)]; C2 = C2 / C2.mean()
        q = np.ones(len(cols)) / len(cols); p = np.ones(n) / n
        E = np.zeros((n, len(GRID)))
        for i, a in enumerate(GRID):
            T = solve(Ms, C1, C2, p, q, a)
            E[:, i] = np.linalg.norm(xy_st[cols][T.argmax(1)] - xy_sc, axis=1)
        return E

    t0 = time.time(); HALVES = []
    for rep in range(NSPLIT):
        h = rng.random(m) < 0.5
        HALVES.append((errors(np.where(h)[0]), errors(np.where(~h)[0])))
        print(f"  split {rep} solved ({time.time()-t0:.0f}s)", flush=True)

    # per-CELL reliability
    rc = []
    for EA, EB in HALVES:
        bA, bB = GRID[EA.argmin(1)], GRID[EB.argmin(1)]
        rc.append(pearsonr(bA, bB)[0])
    print(f"  per-CELL best-alpha test-retest r = {np.mean(rc):+.3f} "
          f"({', '.join(f'{v:+.3f}' for v in rc)})   [yesterday: "
          f"{'+0.142' if SEC=='M3-4' else '-0.039'}]")

    # per-CLADE reliability with a size-matched random control
    def gopt(E, lab, groups): return np.array([GRID[E[lab==k].mean(0).argmin()] for k in groups])
    obs, null = [], []
    for EA, EB in HALVES:
        oA, oB = gopt(EA, cidx, keep), gopt(EB, cidx, keep)
        obs.append(pearsonr(oA, oB)[0])
        for _ in range(NDRAW):
            perm = rng.permutation(n); rl = np.full(n, -1); p0 = 0
            for gi, sz in enumerate(sizes): rl[perm[p0:p0+sz]] = gi; p0 += sz
            a_, b_ = gopt(EA, rl, range(len(sizes))), gopt(EB, rl, range(len(sizes)))
            if a_.std() > 0 and b_.std() > 0:
                v = pearsonr(a_, b_)[0]
                if np.isfinite(v): null.append(v)
    null = np.array(null); o = float(np.mean(obs))
    pv = (np.sum(null >= o) + 1) / (len(null) + 1)
    print(f"  per-CLADE optimal-alpha reliability r = {o:+.3f} "
          f"({', '.join(f'{v:+.3f}' for v in obs)})")
    print(f"    size-matched random null ({len(null)} draws): {null.mean():+.3f} ± {null.std():.3f}"
          f"   z = {(o-null.mean())/null.std():+.2f}   p = {pv:.3f}   [yesterday: p = 0.150]")
    np.save(HERE / f"exact_halves_{SEC}.npy", np.array(HALVES))
