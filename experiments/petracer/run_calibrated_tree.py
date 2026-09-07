"""Does recalibrating the tree into spatial units improve the mapping?

Both matrices are mean-normalised before GW, so the square loss assumes space = tree. The fitted
relation is space = 2.36 x tree - 1.36 on M3-4: the tree's variation is compressed ~2.4x relative
to the spatial variation it predicts. Three versions of C_tree, everything else identical:

  raw        mean-normalised (what the pipeline does now)
  affine     a*tree + b, with (a, b) the least-squares fit of spatial on tree distance
  monotone   E[spatial | tree], the full empirical link, interpolated over 40 bins

affine and monotone use the true spatial distances, so they bound what calibration can buy.
In application the link would have to be learned -- which is the point of the iterative scheme.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
GRID = [0.0, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
_S = {}


def prep(sec, mode):
    d = np.load(HERE / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    xy = np.asarray(d["xy_sc"], float)
    n, m = M.shape
    iu = np.triu_indices(n, 1)
    t = CT[iu] / CT[iu].mean(); s = pdist(xy); s = s / s.mean()
    C1 = CT / CT.mean()
    if mode == "affine":
        a, b = np.polyfit(t, s, 1)
        C1 = np.clip(a * C1 + b, 0.0, None); np.fill_diagonal(C1, 0.0)
    elif mode == "monotone":
        E = np.linspace(t.min(), t.max(), 41); mid, ms = [], []
        for k in range(40):
            mk = (t >= E[k]) & (t <= E[k+1])
            if mk.sum() >= 50: mid.append(0.5*(E[k]+E[k+1])); ms.append(s[mk].mean())
        C1 = np.interp(C1, mid, ms); np.fill_diagonal(C1, 0.0)
    return dict(M=M, C1=C1/C1.mean(), C2=CS/CS.mean(), p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=xy, xy_st=np.asarray(d["xy"], float))


def _init(sec, mode): _S["d"] = prep(sec, mode)


def _solve(a):
    S = _S["d"]
    T = np.asarray(ot.gromov.fused_gromov_wasserstein(
        S["M"], S["C1"], S["C2"], S["p"], S["q"], loss_fun="square_loss", alpha=a,
        max_iter=200, tol_rel=1e-9, tol_abs=1e-9))
    am = T.argmax(1)
    return a, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))


if __name__ == "__main__":
    rows = []
    for sec in ["M3-4", "M3-1", "M3-2"]:
        for mode in ("raw", "affine", "monotone"):
            with Pool(7, initializer=_init, initargs=(sec, mode)) as pool:
                for a, e in pool.imap_unordered(_solve, GRID):
                    rows.append(dict(section=sec, mode=mode, alpha=a, err=e))
    R = pd.DataFrame(rows); R.to_csv(HERE / "calibrated_tree.csv", index=False)
    print(f"{'section':<8}{'tree':<11}{'best err':>10}{'at alpha':>10}{'vs raw':>9}")
    print("-" * 48)
    for sec, g in R.groupby("section"):
        base = g[g["mode"] == "raw"].err.min()
        for mode in ("raw", "affine", "monotone"):
            h = g[g["mode"] == mode]
            i = h.err.idxmin()
            print(f"{sec:<8}{mode:<11}{h.err[i]:>10.1f}{h.alpha[i]:>10.2f}"
                  f"{base - h.err[i]:>+9.1f}")
        print()
