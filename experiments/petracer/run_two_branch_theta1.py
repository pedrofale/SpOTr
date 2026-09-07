"""Two-branch alpha at theta=1 -- the regime where the mask fully decouples the branches.

The earlier two-branch grid used the UNMASKED cost, in which non-uniform alpha is not the
gradient of anything and the branches are coupled through every cross-pair. At theta=1 the
paper's Omega^w decomposition makes them separate problems, joined only by the shared spot
marginal, and per-branch alpha is a valid gradient for ANY alpha_1, alpha_2.

Same grid, same scoring, same split-half as run_two_branch_exact.py, so the numbers are
directly comparable. Also runs the uniform-alpha diagonal at theta=1 against the unmasked
diagonal, which asks a separate question: does dropping the cross-branch tree pairs help or
hurt on its own, before any per-branch alpha?
"""
import sys, time, itertools
import numpy as np, pandas as pd, ot
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, "/home/pedro/projects/spotr/phylospatial_mapping/src")
import spotr.perclade_exact as pe

HERE = Path(__file__).resolve().parent
G = [0.0, 0.15, 0.30, 0.45, 0.60, 0.80]
NSPLIT = 3
rng = np.random.default_rng(0)
rows, land = [], []

for SEC in ["M3-4", "M3-1"]:
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    X_sc, X_st = np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    cidx = np.array([clades.index(c) for c in cc])
    COMP = np.load(HERE / f"comp_{SEC}.npy").astype(float)
    n, m = len(cc), X_st.shape[0]
    M = ot.dist(X_sc, X_st); M = M / M.max()
    C1, C2 = CT / CT.mean(), CS / CS.mean()
    p, q = np.ones(n)/n, np.ones(m)/m
    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1
    cross = 2.0 * (br == 0).sum() * (br == 1).sum() / (n * n)
    print(f"\n=== {SEC} === branch sizes {np.bincount(br)}, "
          f"{100*cross:.0f}% of cell pairs are cross-branch and are DROPPED at theta=1", flush=True)

    def sc(T, cols):
        j = cols[T[:, cols].argmax(1)]
        return float(COMP[j, cidx].mean())

    th1 = np.ones(n)
    T_of, t0 = {}, time.time()
    for a1, a2 in itertools.product(G, G):
        ak = np.where(br == 0, a1, a2)
        T_of[(a1, a2)], _ = pe.perclade_fgw(M, C1, C2, p, q, ak, theta=th1, labels=br,
                                            max_iter=200, tol=1e-9)
    print(f"  {len(T_of)} solves in {time.time()-t0:.0f}s", flush=True)

    allc = np.arange(m)
    full = {k: sc(v, allc) for k, v in T_of.items()}
    for (a1, a2), v in full.items():
        land.append(dict(section=SEC, a1=a1, a2=a2, acc=v))
    best = max(full, key=lambda k: full[k])
    bd = max([(v, v) for v in G], key=lambda k: full[k])
    print(f"  full grid: best (a1,a2) = {best} -> {full[best]:.4f}")
    print(f"             best DIAGONAL a={bd[0]:g} -> {full[bd]:.4f}"
          f"   two-branch gain {full[best]-full[bd]:+.4f}   (alpha=0 {full[(0.0,0.0)]:.4f})")
    print("  uniform-alpha diagonal at theta=1 (masked) -- compare to the unmasked run:")
    print("     " + "  ".join(f"a={g:g}:{full[(g,g)]:.4f}" for g in G), flush=True)

    for rep in range(NSPLIT):
        h = rng.random(m) < 0.5
        fit, hel = np.where(h)[0], np.where(~h)[0]
        ff = {k: sc(v, fit) for k, v in T_of.items()}
        bf = max(ff, key=lambda k: ff[k])
        bdf = max([(v, v) for v in G], key=lambda k: ff[k])
        rows.append(dict(section=SEC, rep=rep, sel_a1=bf[0], sel_a2=bf[1], sel_diag=bdf[0],
                         held_2branch=sc(T_of[bf], hel), held_diag=sc(T_of[bdf], hel),
                         held_a0=sc(T_of[(0.0, 0.0)], hel)))
        r_ = rows[-1]
        print(f"  split {rep}: chose ({bf[0]:g},{bf[1]:g}) vs diagonal {bdf[0]:g} | "
              f"held-out 2-branch {r_['held_2branch']:.4f} diagonal {r_['held_diag']:.4f} "
              f"|| {r_['held_2branch']-r_['held_diag']:+.4f}", flush=True)

pd.DataFrame(land).to_csv(HERE / "two_branch_theta1_landscape.csv", index=False)
r = pd.DataFrame(rows); r.to_csv(HERE / "two_branch_theta1_heldout.csv", index=False)
print("\n" + "="*66)
print(f"{'section':<9}{'held 2-branch':>15}{'held diagonal':>15}{'gain':>9}{'alpha=0':>10}")
print("-"*58)
for s_, g_ in r.groupby("section"):
    print(f"{s_:<9}{g_.held_2branch.mean():>15.4f}{g_.held_diag.mean():>15.4f}"
          f"{(g_.held_2branch-g_.held_diag).mean():>+9.4f}{g_.held_a0.mean():>10.4f}")
print(f"\npositive in {int((r.held_2branch>r.held_diag).sum())}/{len(r)} splits")
