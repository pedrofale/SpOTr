"""Two-branch alpha across the WELL-POSED (alpha, theta) family.

The paper writes 2K parameters (alpha_t, theta_t) but the cost is only a gradient when
v_t = alpha_t (1 - theta_t) is the same for every clade. So the honest parameterisation is

    u_t = alpha_t * theta_t   (within-clade weight, FREE per clade)
    c   = alpha_t (1-theta_t) (cross-clade weight, SHARED)
    ->  alpha_t = u_t + c,   theta_t = u_t / (u_t + c)

c = 0 is theta = 1 (branches fully decoupled; already run, kept here as a consistency check).
Larger c restores the between-clade placement that theta=1 throws away. The question is whether
some intermediate c lets the two branches differ in u AND still place clades relative to
each other.

Grid over (u1, u2, c). Both the two-branch best and the u1=u2 best are optimised over c, so
the comparison is fair.
"""
import os, sys, time, itertools, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, "/home/pedro/projects/spotr/phylospatial_mapping/src")
import spotr.perclade_exact as pe

HERE = Path(__file__).resolve().parent
U = [0.0, 0.15, 0.30, 0.45, 0.60]
C = [0.0, 0.10, 0.20, 0.30]
NSPLIT, NWORK = 3, 8
_S = {}


def load(sec):
    d = np.load(HERE / f"petracer_{sec}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1
    n, m = M.shape
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), br=br, n=n, m=m,
                p=np.ones(n)/n, q=np.ones(m)/m,
                cidx=np.array([clades.index(c) for c in cc]),
                COMP=np.load(HERE / f"comp_{sec}.npy").astype(float))


def _init(sec):
    _S["d"] = load(sec)


def _solve(args):
    u1, u2, c = args
    S = _S["d"]
    u = np.where(S["br"] == 0, u1, u2)
    al = u + c
    th = np.where(al > 0, u / np.maximum(al, 1e-12), 1.0)
    t0 = time.time()
    T, log = pe.perclade_fgw(S["M"], S["C1"], S["C2"], S["p"], S["q"], al,
                             theta=th, labels=S["br"], max_iter=200, tol=1e-9)
    return (u1, u2, c), T.argmax(1).astype(np.int32), T, time.time() - t0, log["n_iter"]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, land = [], []
    for SEC in ["M3-4", "M3-1"]:
        S = load(SEC)
        COMP, cidx, m = S["COMP"], S["cidx"], S["m"]
        grid = [g for g in itertools.product(U, U, C) if g[0] + g[2] <= 1 and g[1] + g[2] <= 1]
        print(f"\n=== {SEC} === branch sizes {np.bincount(S['br'])}, "
              f"{len(grid)} solves on {NWORK} workers", flush=True)
        t0 = time.time()
        Ts = {}
        with Pool(NWORK, initializer=_init, initargs=(SEC,)) as pool:
            for i, (key, _am, T, dt, ni) in enumerate(pool.imap_unordered(_solve, grid, 1)):
                Ts[key] = T
                if (i + 1) % 20 == 0:
                    print(f"    {i+1}/{len(grid)}  ({time.time()-t0:.0f}s)", flush=True)
        print(f"  {len(grid)} solves in {time.time()-t0:.0f}s", flush=True)

        def sc(T, cols):
            return float(COMP[cols[T[:, cols].argmax(1)], cidx].mean())

        allc = np.arange(m)
        full = {k: sc(v, allc) for k, v in Ts.items()}
        for (u1, u2, c), v in full.items():
            land.append(dict(section=SEC, u1=u1, u2=u2, c=c, acc=v,
                             a1=u1+c, a2=u2+c))
        best = max(full, key=lambda k: full[k])
        diag = {k: v for k, v in full.items() if k[0] == k[1]}
        bd = max(diag, key=lambda k: diag[k])
        print(f"  best (u1,u2,c) = {best} -> {full[best]:.4f}   "
              f"[alpha=({best[0]+best[2]:.2f},{best[1]+best[2]:.2f}) "
              f"theta=({best[0]/max(best[0]+best[2],1e-9):.2f},{best[1]/max(best[1]+best[2],1e-9):.2f})]")
        print(f"  best u1=u2     = {bd} -> {diag[bd]:.4f}   two-branch gain "
              f"{full[best]-diag[bd]:+.4f}", flush=True)
        print("  best accuracy at each c (two-branch | uniform):")
        for c in C:
            f_ = {k: v for k, v in full.items() if k[2] == c}
            d_ = {k: v for k, v in f_.items() if k[0] == k[1]}
            kb, kd = max(f_, key=lambda k: f_[k]), max(d_, key=lambda k: d_[k])
            print(f"     c={c:.2f}   {f_[kb]:.4f} at u=({kb[0]:g},{kb[1]:g})   |   "
                  f"{d_[kd]:.4f} at u={kd[0]:g}   -> {f_[kb]-d_[kd]:+.4f}", flush=True)

        for rep in range(NSPLIT):
            h = rng.random(m) < 0.5
            fit, hel = np.where(h)[0], np.where(~h)[0]
            ff = {k: sc(v, fit) for k, v in Ts.items()}
            bf = max(ff, key=lambda k: ff[k])
            dd = {k: v for k, v in ff.items() if k[0] == k[1]}
            bdf = max(dd, key=lambda k: dd[k])
            rows.append(dict(section=SEC, rep=rep, u1=bf[0], u2=bf[1], c=bf[2],
                             du=bdf[0], dc=bdf[2],
                             held_2branch=sc(Ts[bf], hel), held_diag=sc(Ts[bdf], hel)))
            r_ = rows[-1]
            print(f"  split {rep}: chose u=({bf[0]:g},{bf[1]:g}) c={bf[2]:g} vs "
                  f"uniform u={bdf[0]:g} c={bdf[2]:g} | held-out {r_['held_2branch']:.4f} "
                  f"vs {r_['held_diag']:.4f} || {r_['held_2branch']-r_['held_diag']:+.4f}",
                  flush=True)

    pd.DataFrame(land).to_csv(HERE / "two_branch_theta_landscape.csv", index=False)
    r = pd.DataFrame(rows); r.to_csv(HERE / "two_branch_theta_heldout.csv", index=False)
    print("\n" + "=" * 60)
    for s_, g_ in r.groupby("section"):
        print(f"{s_:<8} held 2-branch {g_.held_2branch.mean():.4f}   "
              f"uniform {g_.held_diag.mean():.4f}   "
              f"gain {(g_.held_2branch-g_.held_diag).mean():+.4f}")
    print(f"\npositive in {int((r.held_2branch>r.held_diag).sum())}/{len(r)} splits")
