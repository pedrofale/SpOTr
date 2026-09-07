"""Run the neighbourhood model on the actual mapping task and compare with FGW.

Inputs are exactly what an scRNA-seq + Visium pair supplies:
  from the dissociated side : expression X_sc and the tree D^T   (no positions)
  from the spatial side     : expression X_st and spot xy        (no tree)
No single-cell positions are used to fit anything; they are only used to score.
"""
import os, sys, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.neighbourhood as nb

SIGMA = [2.0, 5.0, 12.0, 30.0]
LAM = [0.0, 0.05, 0.15, 0.4, 1.0, 2.5, 6.0]
_S = {}


def prep(sec):
    d = np.load(HERE / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    DT = np.asarray(d["C_tree"], float)
    Y = np.asarray(d["xy"], float)
    xy_sc = np.asarray(d["xy_sc"], float)
    R = np.median(np.linalg.norm(Y - np.median(Y, 0), axis=1))
    n, m = M.shape
    return dict(M=M, DT=DT, Y=(Y - Y.mean(0)) / R, Yraw=Y, xy_sc=xy_sc,
                p=np.ones(n)/n, q=np.ones(m)/m, n=n, m=m)


def _init(sec): _S["d"] = prep(sec)


def _solve(job):
    sig, lam = job
    S = _S["d"]
    W = nb.tree_kernel(S["DT"], sig)
    W = W / W.sum() * S["n"]                     # scale-free: mean degree 1
    T, log = nb.solve2(S["M"], S["p"], S["q"], S["Y"], W, lam, max_iter=300, tol=1e-10)
    am = np.asarray(T).argmax(1)
    return sig, lam, float(np.median(np.linalg.norm(S["Yraw"][am] - S["xy_sc"], axis=1))), \
        log["n_iter"]


if __name__ == "__main__":
    FGW = {"M3-4": 356.8, "M3-1": 410.7, "M3-2": 419.7}
    rows = []
    for sec in ["M3-4", "M3-1", "M3-2"]:
        with Pool(7, initializer=_init, initargs=(sec,)) as pool:
            for sig, lam, err, it in pool.imap_unordered(
                    _solve, list(itertools.product(SIGMA, LAM))):
                rows.append(dict(section=sec, sigma=sig, lam=lam, err=err, iters=it))
        g = pd.DataFrame([r for r in rows if r["section"] == sec])
        base = g[g.lam == 0].err.iloc[0]
        b = g.loc[g.err.idxmin()]
        print(f"\n=== {sec} ===")
        print(f"  expression only (lambda=0)   {base:7.1f} um")
        print(f"  neighbourhood model          {b.err:7.1f} um   at sigma={b.sigma:g}, "
              f"lambda={b.lam:g}")
        print(f"  best FGW on the same data    {FGW[sec]:7.1f} um")
        print(f"  {'':<29}{'lambda ->':>9}" + "".join(f"{l:>8g}" for l in LAM))
        for s in SIGMA:
            h = g[g.sigma == s].set_index("lam").err
            print(f"  {'sigma = ' + format(s, 'g'):<38}" + "".join(f"{h[l]:>8.0f}" for l in LAM))
    pd.DataFrame(rows).to_csv(HERE / "neighbourhood_map2.csv", index=False)
