"""Does the (u1,u2,c) grid recover a planted per-branch alpha difference?"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
sys.path.insert(0, str(HERE))
import spotr.perclade_exact as pe
from sim_varalpha_branches import simulate

U = [0.0, 0.15, 0.30, 0.45, 0.60]
CG = [0.0, 0.10, 0.20, 0.30]
SEEDS = [0, 1, 2]
_S = {}


def prep(seed):
    d = simulate(seed)
    M = ot.dist(d["X_sc"], d["X_st"]); M = M / M.max()
    CT, CS = d["C_tree"], d["C_space"]
    n, m = d["n"], d["m"]
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), br=d["branch"],
                p=np.ones(n)/n, q=np.ones(m)/m, true=d["true_spot"],
                xy_st=d["xy_st"], xy_sc=d["xy_sc"], n=n, m=m)


def _init(seed):
    _S["d"] = prep(seed)


def _solve(job):
    u1, u2, c, lab = job
    S = _S["d"]
    u = np.where(lab == 0, u1, u2)
    al = u + c
    th = np.where(al > 0, u / np.maximum(al, 1e-12), 1.0)
    T, _ = pe.perclade_fgw(S["M"], S["C1"], S["C2"], S["p"], S["q"], al,
                           theta=th, labels=lab, max_iter=200, tol=1e-9)
    return (u1, u2, c), T.argmax(1).astype(np.int32)


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        S = prep(seed)
        jobs = [(u1, u2, c, S["br"]) for u1, u2 in itertools.product(U, U) for c in CG]
        t0 = time.time()
        A = {}
        with Pool(8, initializer=_init, initargs=(seed,)) as pool:
            for k, am in pool.imap_unordered(_solve, jobs, 1):
                A[k] = am
        print(f"\n=== seed {seed} ===  {len(jobs)} solves in {time.time()-t0:.0f}s", flush=True)

        def acc(am, sel=None):
            s = slice(None) if sel is None else sel
            return float((am[s] == S["true"][s]).mean())

        full = {k: acc(v) for k, v in A.items()}
        best = max(full, key=lambda k: full[k])
        diag = {k: v for k, v in full.items() if k[0] == k[1]}
        bd = max(diag, key=lambda k: diag[k])
        print(f"  best (u1,u2,c) = {best} -> {full[best]:.4f}"
              f"   [alpha=({best[0]+best[2]:.2f},{best[1]+best[2]:.2f})]")
        print(f"  best u1=u2     = {bd} -> {diag[bd]:.4f}"
              f"   two-branch gain {full[best]-diag[bd]:+.4f}")
        print(f"  alpha=0 (u=0,c=0): {full[(0.0,0.0,0.0)]:.4f}   chance {1/S['m']:.4f}")
        for c in CG:
            f_ = {k: v for k, v in full.items() if k[2] == c}
            d_ = {k: v for k, v in f_.items() if k[0] == k[1]}
            kb, kd = max(f_, key=lambda k: f_[k]), max(d_, key=lambda k: d_[k])
            print(f"     c={c:.2f}   two-branch {f_[kb]:.4f} at u=({kb[0]:g},{kb[1]:g})"
                  f"   |   uniform {d_[kd]:.4f} at u={kd[0]:g}   -> {f_[kb]-d_[kd]:+.4f}")
        # per-branch accuracy at the chosen point, to check WHICH branch wanted what
        amb = A[best]
        for b in (0, 1):
            s = S["br"] == b
            print(f"     branch {b} (planted {'tree-informative' if b==0 else 'tree-blind'}): "
                  f"u={best[b]:g}  accuracy {acc(amb, s):.4f}")
        for (u1, u2, c), v in full.items():
            rows.append(dict(seed=seed, u1=u1, u2=u2, c=c, acc=v))
    pd.DataFrame(rows).to_csv(HERE / "sim_branch_alpha.csv", index=False)
    R = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    g = []
    for s_, d_ in R.groupby("seed"):
        b = d_.acc.max(); u = d_[d_.u1 == d_.u2].acc.max()
        g.append(b - u)
        print(f"seed {s_}: best {b:.4f}   uniform {u:.4f}   gain {b-u:+.4f}")
    print(f"mean two-branch gain over {len(SEEDS)} seeds: {np.mean(g):+.4f}")
