"""Can per-group alpha be recovered when the heterogeneity is planted REALISTICALLY?

The earlier positive control planted the effect by making one branch's tree a noisy copy of
position -- a regime no real dataset occupies. Here the mechanism is the paper's own: two
balanced, tree-coherent groups of clades that differ in MOTILITY. The low-motility group stays
where the lineage put it (its tree predicts position, it should want high alpha); the
high-motility group is scattered (it should want low alpha).

Everything else matches PEtracer: Eden growth with volume exclusion, local mixing, tree
reconstruction error, recorder saturation.

Each motility contrast is scored against 15 random partitions of the same sizes, using the same
within-partition contrast (best over the grid - best on the u1=u2 diagonal).
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src")); sys.path.insert(0, str(HERE))
import spotr.perclade_exact as pe
import sim_eden

U = [0.0, 0.15, 0.30, 0.45, 0.60]
CC, NDRAW, SE = 0.20, 5, 10.0
SEEDS = [0, 1, 2]
MIX = [(0.45, 0.45), (0.35, 0.55), (0.25, 0.65), (0.15, 0.85)]
_S = {}


def prep(mix, seed=0):
    d = sim_eden.simulate(seed, sigma_expr=SE, group_mix=mix)
    M = ot.dist(d["X_sc"], d["X_st"]); M = M / M.max()
    n, m = d["n"], d["m"]
    return dict(M=M, C1=d["C_tree"]/d["C_tree"].mean(), C2=d["C_space"]/d["C_space"].mean(),
                p=np.ones(n)/n, q=np.ones(m)/m, true=d["true_spot"], n=n, m=m,
                grp=d["mixgrp"])


def _init(mix, seed): _S["d"] = prep(mix, seed)


def _solve(job):
    draw, u1, u2, lab = job
    S = _S["d"]
    u = np.where(lab == 0, u1, u2); al = u + CC
    th = np.where(al > 0, u / np.maximum(al, 1e-12), 1.0)
    T, _ = pe.perclade_fgw(S["M"], S["C1"], S["C2"], S["p"], S["q"], al,
                           theta=th, labels=lab, max_iter=200, tol=1e-9)
    return draw, u1, u2, float((T.argmax(1) == S["true"]).mean())


if __name__ == "__main__":
    rows = []
    print(f"{'group_mix':>14}{'real gain':>11}{'random gain':>22}{'draws>=real':>13}{'best u':>12}")
    print("-" * 74)
    for mix in MIX:
        t0 = time.time(); obs, nul, picks = [], [], []
        for seed in SEEDS:
            S = prep(mix, seed); n = S["n"]; g0 = S["grp"]
            n0 = int((g0 == 0).sum())
            rng = np.random.default_rng(7 + seed)
            labs = {-1: g0}
            for d_ in range(NDRAW):
                perm = rng.permutation(n); lb = np.ones(n, int); lb[perm[:n0]] = 0
                labs[d_] = lb
            jobs = [(d_, u1, u2, lb) for d_, lb in labs.items()
                    for u1, u2 in itertools.product(U, U)]
            A = {}
            with Pool(8, initializer=_init, initargs=(mix, seed)) as pool:
                for d_, u1, u2, ac in pool.imap_unordered(_solve, jobs, 1):
                    A[(d_, u1, u2)] = ac
            for d_ in labs:
                f = {(u1, u2): A[(d_, u1, u2)] for u1, u2 in itertools.product(U, U)}
                dg = max(f[(u, u)] for u in U)
                bk = max(f, key=lambda k: f[k])
                g_ = f[bk] - dg
                rows.append(dict(mix=str(mix), seed=seed, draw=d_, real=(d_ == -1),
                                 gain=g_, best=f[bk], u1=bk[0], u2=bk[1]))
                (obs if d_ == -1 else nul).append(g_)
                if d_ == -1: picks.append(bk)
        o = float(np.mean(obs)); v = np.array(nul)
        ge = int((v >= o - 1e-12).sum())
        print(f"{str(mix):>14}{o:>+11.4f}{v.mean():>+14.4f} ± {v.std():.4f}"
              f"{ge:>8}/{len(v)}{str(picks):>34}   ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "eden_perclade.csv", index=False)
