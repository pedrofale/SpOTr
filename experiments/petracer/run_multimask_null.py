"""Is the resolution split doing real work, or would ANY 3-way split of the pairs do as well?

Null: replace the true clade-pair resolution matrix R with a random symmetric 0/1 matrix of the
same off-diagonal density (diagonal forced to 1, since same-clade pairs are always resolved).
That preserves the three-class sizes and the extra parameters, and changes only WHICH pairs are
called resolved. Real and null run the identical coarse grid.
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.multimask as mm

W = [0.0, 0.5, 1.0, 2.0]
NDRAW = 8
_S = {}


def prep(sec):
    d = np.load(HERE / f"petracer_{sec}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    cc = np.asarray(d["cell_clade"]).astype(str); _, clade = np.unique(cc, return_inverse=True)
    n, m = M.shape
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), clade=clade, CT=CT,
                p=np.ones(n)/n, q=np.ones(m)/m, n=n, m=m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float))


def _init(sec, Rs):
    S = prep(sec)
    S["bfs"] = {k: mm.BlockFGW(S["M"], S["C1"], S["C2"], S["clade"], R) for k, R in Rs.items()}
    _S["d"] = S


def _solve(job):
    draw, w = job
    S = _S["d"]
    T, _ = S["bfs"][draw].set_weights(*w).solve(S["p"], S["q"], max_iter=200, tol=1e-9)
    return draw, w, np.asarray(T).argmax(1).astype(np.int32)


if __name__ == "__main__":
    out = []
    for SEC in ["M3-4", "M3-1", "M3-2"]:
        S = prep(SEC)
        R0 = mm.clade_resolved_matrix(S["CT"], S["clade"])
        K = R0.shape[0]
        iu = np.triu_indices(K, 1)
        dens = float(R0[iu].mean())
        rng = np.random.default_rng(5)
        Rs = {-1: R0}
        for t in range(NDRAW):
            v = (rng.random(len(iu[0])) < dens).astype(float)
            Rr = np.zeros((K, K)); Rr[iu] = v; Rr = Rr + Rr.T; np.fill_diagonal(Rr, 1.0)
            Rs[t] = Rr
        grid = [(k, w) for k in Rs for w in itertools.product(W, W, W)]
        t0 = time.time(); A = {}
        with Pool(8, initializer=_init, initargs=(SEC, Rs)) as pool:
            for k, w, am in pool.imap_unordered(_solve, grid, 1):
                A[(k, w)] = am
        def err(am):
            return float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))
        best = {}
        for k in Rs:
            E = {w: err(A[(k, w)]) for w in itertools.product(W, W, W)}
            uni = {w: v for w, v in E.items() if w[0] == w[1] == w[2] and w[0] > 0}
            best[k] = (min(uni.values()) - min(E.values()), min(E, key=lambda x: E[x]))
        # a run whose optimum has w_mid == w_far is not testing the resolution split at all --
        # the two classes merge and R drops out. Report how many draws were degenerate.
        degen = sum(1 for k in Rs if best[k][1][1] == best[k][1][2])
        o = best[-1][0]; v = np.array([best[k][0] for k in Rs if k != -1])
        ge = int((v >= o - 1e-9).sum())
        print(f"\n=== {SEC} ===  ({time.time()-t0:.0f}s, off-diagonal resolved density {dens:.2f})")
        print(f"  real resolution split: gain over uniform {o:+.1f} um   at w={best[-1][1]}")
        print(f"  random 3-way splits:   {v.mean():+.1f} +- {v.std():.1f} um   "
              f"{ge}/{len(v)} reach the real one")
        print(f"  attributable to the true structure: {o - v.mean():+.1f} um "
              f"(the rest is extra capacity);  degenerate optima (w_mid==w_far): {degen}/{len(Rs)}")
        out.append(dict(section=SEC, real=o, null_mean=v.mean(), null_sd=v.std(), ge=ge,
                        nd=len(v), w=str(best[-1][1])))
    pd.DataFrame(out).to_csv(HERE / "multimask_null.csv", index=False)
