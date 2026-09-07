"""Does splitting the tree by RESOLUTION beat one weight for the whole tree?

Grid over (w_near, w_mid, w_far). Uniform weights reproduce ordinary FGW, so the current model
is a point inside the grid and the comparison is exact rather than approximate.

Prediction to falsify: w_far should come out near zero on M3-1 and M3-2, the sections where a
FLATTENED tree beat the real one.
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.multimask as mm

W = [0.0, 0.25, 0.5, 1.0, 2.0]
NSPLIT = 3
_S = {}


def prep(sec):
    d = np.load(HERE / f"petracer_{sec}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    cidx = np.array([clades.index(c) for c in cc])
    _, clade = np.unique(cc, return_inverse=True)
    n, m = M.shape
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), clade=clade, CT=CT,
                p=np.ones(n)/n, q=np.ones(m)/m, n=n, m=m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float),
                cidx=cidx, COMP=np.load(HERE / f"comp_{sec}.npy").astype(float))


def _init(sec):
    S = prep(sec)
    S["bf"] = mm.BlockFGW(S["M"], S["C1"], S["C2"], S["clade"],
                          mm.clade_resolved_matrix(S["CT"], S["clade"]))
    _S["d"] = S


def _solve(w):
    S = _S["d"]
    T, _ = S["bf"].set_weights(*w).solve(S["p"], S["q"], max_iter=200, tol=1e-9)
    return w, np.asarray(T).argmax(1).astype(np.int32)


if __name__ == "__main__":
    rows = []
    for SEC in ["M3-4", "M3-1", "M3-2"]:
        S = prep(SEC)
        grid = list(itertools.product(W, W, W))
        t0 = time.time(); A = {}
        with Pool(8, initializer=_init, initargs=(SEC,)) as pool:
            for w, am in pool.imap_unordered(_solve, grid, 1):
                A[w] = am
        print(f"\n=== {SEC} ===  {len(grid)} solves in {time.time()-t0:.0f}s", flush=True)

        def err(am, cols=None):
            if cols is None:
                return float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))
            k = np.isin(am, cols)
            return float(np.median(np.linalg.norm(
                S["xy_st"][am[k]] - S["xy_sc"][k], axis=1))) if k.sum() else np.inf

        E = {w: err(am) for w, am in A.items()}
        uni = {w: v for w, v in E.items() if w[0] == w[1] == w[2] and w[0] > 0}
        bu = min(uni, key=lambda k: uni[k]); bb = min(E, key=lambda k: E[k])
        print(f"  best UNIFORM (= ordinary FGW)  w={bu[0]:g}      {uni[bu]:.1f} um")
        print(f"  best SPLIT   near/mid/far {bb}   {E[bb]:.1f} um   "
              f"gain {uni[bu]-E[bb]:+.1f} um")
        top = sorted(E, key=lambda k: E[k])[:5]
        print("  top 5:  " + "   ".join(f"({a:g},{b:g},{c:g}) {E[(a,b,c)]:.0f}" for a, b, c in top))

        rng = np.random.default_rng(0)
        for rep in range(NSPLIT):
            h = rng.random(S["m"]) < 0.5
            fit, hel = np.where(h)[0], np.where(~h)[0]
            ef = {w: err(am, fit) for w, am in A.items()}
            uf = {w: v for w, v in ef.items() if w[0] == w[1] == w[2] and w[0] > 0}
            bs, bun = min(ef, key=lambda k: ef[k]), min(uf, key=lambda k: uf[k])
            rows.append(dict(section=SEC, rep=rep, w=str(bs), w_uni=bun[0],
                             held_split=err(A[bs], hel), held_uni=err(A[bun], hel)))
            r = rows[-1]
            print(f"  split {rep}: chose {bs} vs uniform {bun[0]:g} | held-out "
                  f"{r['held_split']:.1f} vs {r['held_uni']:.1f} um || "
                  f"{r['held_uni']-r['held_split']:+.1f}", flush=True)
        for w, v in E.items():
            rows.append(dict(section=SEC, rep=-1, w=str(w), full_err=v))
    pd.DataFrame(rows).to_csv(HERE / "multimask_grid.csv", index=False)
    R = pd.DataFrame([r for r in rows if r["rep"] >= 0])
    print("\n" + "=" * 60)
    for s_, g in R.groupby("section"):
        print(f"{s_:<8} held-out split {g.held_split.mean():.1f} um   "
              f"uniform {g.held_uni.mean():.1f} um   "
              f"gain {(g.held_uni-g.held_split).mean():+.1f} um")
