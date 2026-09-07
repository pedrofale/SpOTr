"""Is M3-1's two-branch gain about the TREE, or about having two free parameters?

Null: random 2-way partitions of the same SIZES (240 / 1205) as the real branch split. For each
partition -- real and random alike -- run the identical 5x5 (u1,u2) grid at c = 0.30 and take

    gain = best over the whole grid  -  best on the u1=u2 diagonal

which is a within-partition contrast, so it needs no cross-draw normalisation. Also scored on
the same three held-out spot splits used before.

NOTE the mask itself depends on the partition, so even the uniform-u diagonal has to be
re-solved for every draw. 25 solves per draw.
"""
import os, sys, time, itertools
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
SEC, CC, NDRAW, NWORK = "M3-1", 0.30, 40, 8
U = [0.0, 0.15, 0.30, 0.45, 0.60]
_S = {}


def load():
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1
    n, m = M.shape
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), br=br, n=n, m=m,
                p=np.ones(n)/n, q=np.ones(m)/m,
                cidx=np.array([clades.index(c) for c in cc]),
                COMP=np.load(HERE / f"comp_{SEC}.npy").astype(float))


def _init():
    _S["d"] = load()


def _solve(job):
    draw, u1, u2, lab = job
    S = _S["d"]
    u = np.where(lab == 0, u1, u2)
    al = u + CC
    th = np.where(al > 0, u / np.maximum(al, 1e-12), 1.0)
    T, _ = pe.perclade_fgw(S["M"], S["C1"], S["C2"], S["p"], S["q"], al,
                           theta=th, labels=lab, max_iter=200, tol=1e-9)
    return draw, u1, u2, T.argmax(1).astype(np.int32)


if __name__ == "__main__":
    S = load()
    n, m, COMP, cidx = S["n"], S["m"], S["COMP"], S["cidx"]
    n0 = int((S["br"] == 0).sum())
    rng = np.random.default_rng(0)
    splits = [rng.random(m) < 0.5 for _ in range(3)]

    labs = {-1: S["br"]}
    for d_ in range(NDRAW):
        perm = rng.permutation(n)
        lb = np.ones(n, dtype=int); lb[perm[:n0]] = 0
        labs[d_] = lb
    jobs = [(d_, u1, u2, lb) for d_, lb in labs.items() for u1, u2 in itertools.product(U, U)]
    print(f"{SEC}: real branch sizes {np.bincount(S['br'])}, c={CC}, "
          f"{NDRAW} random draws of the same sizes, {len(jobs)} solves", flush=True)

    A = {}
    t0 = time.time()
    with Pool(NWORK, initializer=_init) as pool:
        for i, (d_, u1, u2, am) in enumerate(pool.imap_unordered(_solve, jobs, 1)):
            A[(d_, u1, u2)] = am
            if (i + 1) % 100 == 0:
                print(f"   {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    def acc(am, cols):
        keep = np.isin(am, cols)
        j = np.where(keep, am, cols[0])
        return float(COMP[j, cidx].mean())

    def gains(d_, cols):
        f = {(u1, u2): acc(A[(d_, u1, u2)], cols) for u1, u2 in itertools.product(U, U)}
        dg = max(f[(u, u)] for u in U)
        return max(f.values()) - dg, max(f.values()), dg

    rows = []
    for d_ in labs:
        g_full, b_full, d_full = gains(d_, np.arange(m))
        hs = []
        for h in splits:
            fit, hel = np.where(h)[0], np.where(~h)[0]
            ff = {(u1, u2): acc(A[(d_, u1, u2)], fit) for u1, u2 in itertools.product(U, U)}
            bf = max(ff, key=lambda k: ff[k])
            bdf = max([(u, u) for u in U], key=lambda k: ff[k])
            hs.append(acc(A[(d_, *bf)], hel) - acc(A[(d_, *bdf)], hel))
        rows.append(dict(draw=d_, real=(d_ == -1), gain_full=g_full, best=b_full,
                         diag=d_full, gain_held=float(np.mean(hs)),
                         held_pos=int(sum(x > 0 for x in hs))))
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "branch_null_M3-1.csv", index=False)
    obs = R[R.real].iloc[0]
    nul = R[~R.real]
    print("\n" + "=" * 68)
    for tag, col in [("FULL-GRID gain", "gain_full"), ("HELD-OUT gain", "gain_held")]:
        o, v = float(obs[col]), nul[col].values
        p = (1 + int((v >= o).sum())) / (1 + len(v))
        print(f"{tag:<16} real {o:+.4f}   random {v.mean():+.4f} +- {v.std():.4f}   "
              f"z = {(o-v.mean())/max(v.std(),1e-9):+.2f}   p = {p:.3f}  "
              f"({int((v>=o).sum())}/{len(v)} draws >= real)")
    print(f"\nbest accuracy: real {obs['best']:.4f}   random {nul.best.mean():.4f} "
          f"+- {nul.best.std():.4f}   z = {(obs['best']-nul.best.mean())/max(nul.best.std(),1e-9):+.2f}")
    print(f"uniform-u best: real {obs['diag']:.4f}   random {nul['diag'].mean():.4f} "
          f"+- {nul['diag'].std():.4f}")
    print(f"held-out splits positive: real {int(obs['held_pos'])}/3   "
          f"random mean {nul.held_pos.mean():.2f}/3")
