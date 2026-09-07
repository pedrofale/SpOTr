"""Ceiling test for selective reweighting.

Set alpha_i per cell from a trust statistic and solve with the pair-mean cost
(w_ik = (a_i+a_k)/2), which is verified to be a genuine gradient for ANY per-cell alpha and to
reproduce POT exactly at uniform alpha. Nothing is estimated per cell -- alpha_i is COMPUTED
from the statistic, and only two global numbers (base, scale) are fitted.

The ORACLE statistic uses ground truth, so it bounds what any selective scheme could achieve.
If the ceiling is low, an observable predictor cannot help and the idea is dead.
"""
import os, sys, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot, pickle
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.perclade_exact as pe

BASE = [0.2, 0.4, 0.6, 0.8]
SCALE = [0.0, 0.10, 0.20, 0.35]
_S = {}


def prep():
    d = np.load(HERE / "petracer_M3-4.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    n, m = M.shape
    K = 25; CTd = CT.copy(); np.fill_diagonal(CTd, np.inf)
    nb = np.argpartition(CTd, K, axis=1)[:, :K]
    stats = {
        "oracle": -np.array([np.linalg.norm(xy_sc[i] - np.median(xy_sc[cc == cc[i]], 0))
                             for i in range(n)]),
        "observable": -np.take_along_axis(CT, nb, 1).mean(1),
        "shuffled": None,
    }
    rng = np.random.default_rng(0)
    stats["shuffled"] = rng.permutation(stats["oracle"])
    z = {k: (v - v.mean()) / (v.std() + 1e-9) for k, v in stats.items()}
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=xy_sc, xy_st=xy_st, z=z, n=n, m=m)


def _init():
    _S["d"] = prep()


def _solve(job):
    which, base, scale = job
    S = _S["d"]
    a = np.clip(base + scale * S["z"][which], 0.01, 0.98) if scale > 0 else np.full(S["n"], base)
    T, _ = pe.perclade_fgw(S["M"], S["C1"], S["C2"], S["p"], S["q"], a, max_iter=200, tol=1e-9)
    am = np.asarray(T).argmax(1)
    return which, base, scale, float(np.median(
        np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))


if __name__ == "__main__":
    jobs = [("oracle", b, s) for b in BASE for s in SCALE]
    jobs += [(w, b, s) for w in ("observable", "shuffled") for b in BASE for s in SCALE if s > 0]
    rows = []
    with Pool(8, initializer=_init) as pool:
        for which, b, s, err in pool.imap_unordered(_solve, jobs, 1):
            rows.append(dict(stat=which, base=b, scale=s, err=err))
    R = pd.DataFrame(rows); R.to_csv(HERE / "percell_trust.csv", index=False)
    uni = R[(R.scale == 0)].groupby("base").err.mean()
    print(f"uniform alpha (scale = 0):")
    for b, v in uni.items():
        print(f"   base {b}: {v:.1f} um")
    best_uni = uni.min()
    print(f"\n  best uniform: {best_uni:.1f} um\n")
    print(f"{'statistic':<14}{'best err':>10}{'at (base,scale)':>18}{'gain':>9}")
    print("-" * 52)
    for w in ("oracle", "observable", "shuffled"):
        g = R[(R.stat == w) & (R.scale > 0)]
        i = g.err.idxmin()
        print(f"{w:<14}{g.err[i]:>10.1f}{f'({g.base[i]:g}, {g.scale[i]:g})':>18}"
              f"{best_uni - g.err[i]:>+9.1f}")
