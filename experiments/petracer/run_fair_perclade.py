"""Re-run the per-clade analysis on the FAIR expression cost.

The result most at risk: per-clade tree benefit correlated rho = +0.53 with how badly expression
placed that clade. The Jensen bias in the old spot build was worst exactly where a spot's member
cells were most heterogeneous, so that correlation may have been measuring the bug.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np, pandas as pd, ot
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
BEST = {"M3-4": 0.75, "M3-1": 0.45, "M3-2": 0.45}     # fair-cost optima

for SEC in ["M3-4"]:
    d = np.load(HERE / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    X_sc = np.asarray(d["X_sc"], float); X_st = np.asarray(d["X_st"], float)
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    M = ot.dist(X_sc, X_st); M = M / M.max()
    n, m = M.shape; p, q = np.ones(n)/n, np.ones(m)/m
    C1, C2 = CT/CT.mean(), CS/CS.mean()
    T = {}
    for a in (0.0, BEST[SEC]):
        T[a] = np.asarray(ot.gromov.fused_gromov_wasserstein(
            M, C1, C2, p, q, loss_fun="square_loss", alpha=a,
            max_iter=200, tol_rel=1e-9, tol_abs=1e-9))
    e0 = np.linalg.norm(xy_st[T[0.0].argmax(1)] - xy_sc, axis=1)
    eb = np.linalg.norm(xy_st[T[BEST[SEC]].argmax(1)] - xy_sc, axis=1)
    delta = e0 - eb
    print(f"=== {SEC}, FAIR cost, alpha = {BEST[SEC]} ===")
    print(f"  no tree {np.median(e0):.1f} um -> {np.median(eb):.1f} um\n")

    # 1. who does the tree help?
    cen = {c: np.median(xy_sc[cc == c], 0) for c in np.unique(cc)}
    spr = {c: np.median(np.linalg.norm(xy_sc[cc == c] - cen[c], axis=1)) for c in cen}
    rel = np.array([np.linalg.norm(xy_sc[i] - cen[cc[i]]) / max(spr[cc[i]], 1e-9)
                    for i in range(n)])
    qs = np.quantile(rel, [0, .25, .5, .75, 1.0])
    print(f"  {'cells by how far they stray':<30}{'tree helps':>12}{'median gain':>14}")
    for k, lab in enumerate(["core (closest 25%)", "inner", "outer", "strays (furthest 25%)"]):
        s = (rel >= qs[k]) & (rel <= qs[k+1])
        print(f"  {lab:<30}{(delta[s] > 0).mean():>11.0%}{np.median(delta[s]):>+13.0f} um")
    print(f"  stray vs benefit: rho = {spearmanr(rel, delta).statistic:+.3f}\n")

    # 2. do clades differ, and what predicts it?
    big = [c for c in np.unique(cc) if (cc == c).sum() >= 30]
    R = pd.DataFrame([dict(clade=c, n=int((cc == c).sum()),
                           cohesion=spr[c] / np.median(np.linalg.norm(
                               xy_sc - np.median(xy_sc, 0), axis=1)),
                           err_a0=float(np.median(e0[cc == c])),
                           mean_delta=float(delta[cc == c].mean())) for c in big])
    lab = np.array([big.index(c) if c in big else -1 for c in cc]); mk = lab >= 0
    def eta2(v, l):
        gm = v.mean()
        return sum((v[l == k].mean()-gm)**2*(l == k).sum() for k in np.unique(l))/((v-gm)**2).sum()
    obs = eta2(delta[mk], lab[mk])
    null = [eta2(delta[mk], rng.permutation(lab[mk])) for _ in range(2000)]
    print(f"  clade explains eta^2 = {obs:.4f}  (null {np.mean(null):.4f} +- {np.std(null):.4f},"
          f" p = {(1+sum(x>=obs for x in null))/2001:.4f})")
    print(f"  between-clade SD {R.mean_delta.std():.0f} um, "
          f"within-clade SD {np.mean([delta[cc==c].std() for c in big]):.0f} um\n")
    print(f"  {'predictor of per-clade benefit':<38}{'rho':>8}{'p':>9}   [was]")
    was = {"cohesion": "+0.168 / 0.550", "err_a0": "+0.529 / 0.043", "n": "+0.435 / 0.105"}
    for nm, col in [("clade spatial cohesion", "cohesion"),
                    ("how badly expression alone placed it", "err_a0"),
                    ("clade size", "n")]:
        r = spearmanr(R[col], R.mean_delta)
        print(f"  {nm:<38}{r.statistic:>+8.3f}{r.pvalue:>9.3f}   [{was[col]}]")
