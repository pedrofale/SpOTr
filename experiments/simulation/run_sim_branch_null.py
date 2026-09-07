"""The same null the real data faced, applied to the simulation where the answer is known.

If the true branch split beats random splits HERE but not on PEtracer, the difference is the
data, not the method or the test.
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src")); sys.path.insert(0, str(HERE))
import spotr.perclade_exact as pe
from sim_varalpha_branches import simulate

U = [0.0, 0.15, 0.30, 0.45, 0.60]
CC, NDRAW, SEEDS = 0.0, 20, [0, 1, 2]
_S = {}


def prep(seed):
    d = simulate(seed)
    M = ot.dist(d["X_sc"], d["X_st"]); M = M / M.max()
    n, m = d["n"], d["m"]
    return dict(M=M, C1=d["C_tree"]/d["C_tree"].mean(), C2=d["C_space"]/d["C_space"].mean(),
                br=d["branch"], p=np.ones(n)/n, q=np.ones(m)/m, true=d["true_spot"], n=n, m=m)


def _init(seed):
    _S["d"] = prep(seed)


def _solve(job):
    draw, u1, u2, lab = job
    S = _S["d"]
    u = np.where(lab == 0, u1, u2); al = u + CC
    th = np.where(al > 0, u / np.maximum(al, 1e-12), 1.0)
    T, _ = pe.perclade_fgw(S["M"], S["C1"], S["C2"], S["p"], S["q"], al,
                           theta=th, labels=lab, max_iter=200, tol=1e-9)
    return draw, u1, u2, T.argmax(1).astype(np.int32)


if __name__ == "__main__":
    out = []
    for seed in SEEDS:
        S = prep(seed); n = S["n"]
        n0 = int((S["br"] == 0).sum())
        rng = np.random.default_rng(100 + seed)
        labs = {-1: S["br"]}
        for d_ in range(NDRAW):
            perm = rng.permutation(n); lb = np.ones(n, int); lb[perm[:n0]] = 0
            labs[d_] = lb
        jobs = [(d_, u1, u2, lb) for d_, lb in labs.items()
                for u1, u2 in itertools.product(U, U)]
        t0 = time.time(); A = {}
        with Pool(8, initializer=_init, initargs=(seed,)) as pool:
            for d_, u1, u2, am in pool.imap_unordered(_solve, jobs, 1):
                A[(d_, u1, u2)] = am
        print(f"\n=== seed {seed} === {len(jobs)} solves in {time.time()-t0:.0f}s", flush=True)
        rows = []
        for d_ in labs:
            f = {(u1, u2): float((A[(d_, u1, u2)] == S["true"]).mean())
                 for u1, u2 in itertools.product(U, U)}
            dg = max(f[(u, u)] for u in U)
            rows.append(dict(seed=seed, draw=d_, real=(d_ == -1),
                             gain=max(f.values()) - dg, best=max(f.values()), diag=dg))
        R = pd.DataFrame(rows); out.append(R)
        o, nul = R[R.real].iloc[0], R[~R.real]
        for tag, col in [("gain", "gain"), ("best acc", "best")]:
            v = nul[col].values; ov = float(o[col])
            p = (1 + int((v >= ov).sum())) / (1 + len(v))
            print(f"  {tag:<9} real {ov:+.4f}   random {v.mean():+.4f} +- {v.std():.4f}   "
                  f"z = {(ov-v.mean())/max(v.std(),1e-9):+.2f}   p = {p:.3f}")
    R = pd.concat(out); R.to_csv(HERE / "sim_branch_null.csv", index=False)
    print("\n" + "=" * 60)
    for col in ["gain", "best"]:
        zs = []
        for s_, g in R.groupby("seed"):
            o = float(g[g.real].iloc[0][col]); v = g[~g.real][col].values
            zs.append((o - v.mean()) / max(v.std(), 1e-9))
        print(f"{col:<6} z per seed: {', '.join(f'{z:+.2f}' for z in zs)}   mean {np.mean(zs):+.2f}")
