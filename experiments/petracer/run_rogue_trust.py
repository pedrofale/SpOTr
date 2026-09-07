"""Down-weight likely-misassigned ("rogue") cells so they stop dragging the tree term.

alpha_i in [0,1] is COMPUTED from a rogueness statistic; only two global numbers are fitted
(lam, the tree/expression ratio, and scale, how hard rogues are discounted). scale = 0 is
ordinary FGW. The PRODUCT weight means a rogue cell is removed from the tree term on both
sides -- it neither obeys nor imposes tree constraints.

  margin    OBSERVABLE: how confidently the tree assigns the cell to its clade
            (mean patristic distance to the nearest other clade, minus to its own).
            Low margin = the clade call is weak = likely misassigned.
  oracle    ground truth: how far the cell actually sits from its clade. Upper bound.
  shuffled  same values, no information.
"""
import os, sys, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.trustfgw as tf

LAM = [0.3, 0.6, 1.0, 2.0]
SCALE = [0.0, 0.2, 0.4, 0.7]
_S = {}


def prep():
    d = np.load(HERE / "petracer_M3-4.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
    cc = np.asarray(d["cell_clade"]).astype(str); clades = np.unique(cc)
    n, m = M.shape
    own = np.zeros(n); oth = np.zeros(n)
    for c in clades:
        idx = np.where(cc == c)[0]
        for i in idx:
            own[i] = CT[i][idx[idx != i]].mean()
            oth[i] = min(CT[i][np.where(cc == c2)[0]].mean() for c2 in clades if c2 != c)
    cen = {c: np.median(xy_sc[cc == c], 0) for c in clades}
    spr = {c: np.median(np.linalg.norm(xy_sc[cc == c] - cen[c], axis=1)) for c in clades}
    rog = {"margin": -(oth - own),                                   # observable
           "oracle": np.array([np.linalg.norm(xy_sc[i] - cen[cc[i]]) / spr[cc[i]]
                               for i in range(n)])}
    rog["shuffled"] = np.random.default_rng(0).permutation(rog["oracle"])
    z = {k: (v - v.mean()) / (v.std() + 1e-9) for k, v in rog.items()}
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=xy_sc, xy_st=xy_st, z=z, n=n, cc=cc)


def _init(): _S["d"] = prep()


def _solve(job):
    which, lam, scale = job
    S = _S["d"]
    a = np.ones(S["n"]) if scale == 0 else np.clip(1.0 - scale * S["z"][which], 0.0, 1.0)
    T, _ = tf.solve(S["M"], S["C1"], S["C2"], S["p"], S["q"], a, lam, max_iter=200, tol=1e-9)
    am = np.asarray(T).argmax(1)
    return which, lam, scale, float(np.median(
        np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1))), float(a.mean())


if __name__ == "__main__":
    jobs = [("margin", l, 0.0) for l in LAM]
    jobs += [(w, l, s) for w in ("margin", "oracle", "shuffled")
             for l in LAM for s in SCALE if s > 0]
    rows = []
    with Pool(8, initializer=_init) as pool:
        for w, l, s, err, am in pool.imap_unordered(_solve, jobs, 1):
            rows.append(dict(stat=w, lam=l, scale=s, err=err, mean_alpha=am))
    R = pd.DataFrame(rows); R.to_csv(HERE / "rogue_trust.csv", index=False)
    uni = R[R.scale == 0]
    bu = uni.err.min()
    print("uniform (scale = 0, ordinary FGW):")
    for _, r in uni.sort_values("lam").iterrows():
        print(f"   lam {r.lam}: {r.err:.1f} um")
    print(f"\n  best uniform {bu:.1f} um\n")
    print(f"{'rogueness statistic':<16}{'best err':>10}{'at (lam,scale)':>17}{'gain':>9}")
    print("-" * 54)
    for w in ("margin", "oracle", "shuffled"):
        g = R[(R.stat == w) & (R.scale > 0)]
        i = g.err.idxmin()
        print(f"{w:<16}{g.err[i]:>10.1f}{f'({g.lam[i]:g}, {g.scale[i]:g})':>17}"
              f"{bu - g.err[i]:>+9.1f}")
