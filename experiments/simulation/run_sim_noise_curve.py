"""How much tree-space concordance is needed before per-branch alpha becomes detectable?

Sweep tree_noise (branch A's tree = true position + N(0, tree_noise)). At each level, run the
(u1,u2) grid at c=0 for the true branch split and for 5 random splits of the same sizes, so the
gain is always read against its own null.
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src")); sys.path.insert(0, str(HERE))
import spotr.perclade_exact as pe
from sim_varalpha_branches import simulate

U = [0.0, 0.15, 0.30, 0.45, 0.60]
NOISE = [0.05, 0.10, 0.20, 0.35, 0.50, 0.80]
SEEDS, NRAND = [0, 1, 2], 5
_S = {}


def prep(seed, tn):
    d = simulate(seed, tree_noise=tn)
    M = ot.dist(d["X_sc"], d["X_st"]); M = M / M.max()
    n, m = d["n"], d["m"]
    s = np.where(d["branch"] == 0)[0]
    iu = np.triu_indices(len(s), 1)
    ct = d["C_tree"][np.ix_(s, s)][iu]
    cs = np.linalg.norm(d["xy_sc"][s][:, None] - d["xy_sc"][s][None, :], axis=-1)[iu]
    return dict(M=M, C1=d["C_tree"]/d["C_tree"].mean(), C2=d["C_space"]/d["C_space"].mean(),
                br=d["branch"], p=np.ones(n)/n, q=np.ones(m)/m, true=d["true_spot"],
                n=n, m=m, corr=float(pearsonr(ct, cs)[0]))


def _init(seed, tn):
    _S["d"] = prep(seed, tn)


def _solve(job):
    draw, u1, u2, lab = job
    S = _S["d"]
    u = np.where(lab == 0, u1, u2)
    th = np.where(u > 0, 1.0, 1.0)
    T, _ = pe.perclade_fgw(S["M"], S["C1"], S["C2"], S["p"], S["q"], u,
                           theta=th, labels=lab, max_iter=200, tol=1e-9)
    return draw, u1, u2, T.argmax(1).astype(np.int32)


if __name__ == "__main__":
    rows = []
    for tn in NOISE:
        for seed in SEEDS:
            S = prep(seed, tn); n = S["n"]; n0 = int((S["br"] == 0).sum())
            rng = np.random.default_rng(500 + seed)
            labs = {-1: S["br"]}
            for d_ in range(NRAND):
                perm = rng.permutation(n); lb = np.ones(n, int); lb[perm[:n0]] = 0
                labs[d_] = lb
            jobs = [(d_, u1, u2, lb) for d_, lb in labs.items()
                    for u1, u2 in itertools.product(U, U)]
            A = {}
            with Pool(8, initializer=_init, initargs=(seed, tn)) as pool:
                for d_, u1, u2, am in pool.imap_unordered(_solve, jobs, 1):
                    A[(d_, u1, u2)] = am
            for d_ in labs:
                f = {(u1, u2): float((A[(d_, u1, u2)] == S["true"]).mean())
                     for u1, u2 in itertools.product(U, U)}
                dg = max(f[(u, u)] for u in U)
                bk = max(f, key=lambda k: f[k])
                rows.append(dict(tree_noise=tn, seed=seed, draw=d_, real=(d_ == -1),
                                 corr=S["corr"], gain=max(f.values()) - dg,
                                 best=max(f.values()), diag=dg, u1=bk[0], u2=bk[1]))
            r = rows[-1 - NRAND]
            print(f"  tn={tn:.2f} seed {seed}: corr(tree,space)={S['corr']:+.3f}  "
                  f"gain {r['gain']:+.4f}  best {r['best']:.4f} at u={r['u1']:g},{r['u2']:g}",
                  flush=True)
    R = pd.DataFrame(rows); R.to_csv(HERE / "sim_noise_curve.csv", index=False)
    print("\n" + "=" * 74)
    print(f"{'tree_noise':>11}{'corr':>8}{'real gain':>12}{'random gain':>22}{'z':>8}")
    print("-" * 61)
    for tn, g in R.groupby("tree_noise"):
        o = g[g.real].gain.mean(); v = g[~g.real].gain.values
        print(f"{tn:>11.2f}{g.corr.mean():>8.3f}{o:>+12.4f}"
              f"{v.mean():>+14.4f} ± {v.std():.4f}{(o-v.mean())/max(v.std(),1e-9):>8.2f}")
