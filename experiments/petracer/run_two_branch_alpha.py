"""alpha_k with K=2: the two main phylogenetic branches, exhaustive 2D grid.

At 18 clades alpha_k needed coordinate ascent (an optimiser, with all its artefacts) and the
per-clade coherence did not replicate across tissue regions. At K=2 the whole landscape can be
enumerated -- 7x7 solves -- so there is no optimiser to blame, and the branch split is the most
robust feature of the tree.

Branch 1 = clades 1,2,3,4; branch 2 = the other 14. Verified identical in all three sections.
Within-branch tree-vs-space cell-pair r is higher for branch 1 in all three (+0.24/+0.22/+0.21
vs +0.15/+0.16/+0.20), so the prediction is alpha_1 > alpha_2.

Scored two ways: on all spots (the landscape), and split-half -- argmax chosen on half the
spots, accuracy reported on the other half, which is the honest number.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J, TAU = 0.02, 50, 3, 0.01
G = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75]
NSPLIT = 3
rng = np.random.default_rng(0)
rows, land = [], []

for SEC in ["M3-1", "M3-2", "M3-4"]:
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
    CT = np.asarray(d["C_tree"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    cidx = np.array([clades.index(c) for c in cc])
    COMP = np.load(HERE / f"comp_{SEC}.npy").astype(float)
    n, m = len(cc), X_st.shape[0]

    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1
    A = np.zeros((n, 2), np.float32); A[np.arange(n), br] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                             CT.astype(np.float32),
                                             np.asarray(d["C_space"], np.float32))
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

    def solve(a1, a2):
        ak = jnp.asarray(np.array([a1, a2], np.float32))
        def build(g):
            return build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                                equalize="measured")
        g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
        return np.asarray(g)

    def sc(g, cols):
        if not np.isfinite(g).all(): return np.nan
        return float(COMP[cols[np.asarray(g)[:, cols].argmax(1)], cidx].mean())

    allc = np.arange(m)
    Gam = {}
    for a1 in G:
        for a2 in G:
            Gam[(a1, a2)] = solve(a1, a2)
    full = {k: sc(v, allc) for k, v in Gam.items()}
    for (a1, a2), v in full.items():
        land.append(dict(section=SEC, a1=a1, a2=a2, acc=v))
    best = max(full, key=lambda k: full[k])
    bd = max([(v, v) for v in G], key=lambda k: full[k])
    print(f"\n=== {SEC} === branch sizes {np.bincount(br)}  "
          f"(branch 1 clades: {sorted(set(cc[br==0]), key=int)})")
    print(f"  full-grid best (a1,a2) = {best}  acc {full[best]:.4f}")
    print(f"  best DIAGONAL (a1=a2)  = {bd[0]:.2f}      acc {full[bd]:.4f}"
          f"   -> 2-branch gain {full[best]-full[bd]:+.4f}")
    print(f"  alpha=0 {full[(0.0,0.0)]:.4f}")

    for rep in range(NSPLIT):
        h = rng.random(m) < 0.5
        fit, hel = np.where(h)[0], np.where(~h)[0]
        ff = {k: sc(v, fit) for k, v in Gam.items()}
        bf = max(ff, key=lambda k: ff[k])
        bdf = max([(v, v) for v in G], key=lambda k: ff[k])
        rows.append(dict(section=SEC, rep=rep, sel_a1=bf[0], sel_a2=bf[1],
                         held_2branch=sc(Gam[bf], hel), held_global=sc(Gam[bdf], hel),
                         sel_global=bdf[0], held_a0=sc(Gam[(0.0, 0.0)], hel)))
        r_ = rows[-1]
        print(f"  split {rep}: chose (a1,a2)=({bf[0]:.2f},{bf[1]:.2f}) vs global {bdf[0]:.2f} "
              f"| held-out 2-branch {r_['held_2branch']:.4f} global {r_['held_global']:.4f} "
              f"|| {r_['held_2branch']-r_['held_global']:+.4f}")

pd.DataFrame(land).to_csv(HERE / "two_branch_landscape.csv", index=False)
r = pd.DataFrame(rows); r.to_csv(HERE / "two_branch_heldout.csv", index=False)
print("\n" + "=" * 72)
print(f"{'section':<9}{'held 2-branch':>15}{'held global':>13}{'gain':>9}{'alpha=0':>10}")
print("-" * 56)
for s_, g_ in r.groupby("section"):
    print(f"{s_:<9}{g_.held_2branch.mean():>15.4f}{g_.held_global.mean():>13.4f}"
          f"{(g_.held_2branch-g_.held_global).mean():>+9.4f}{g_.held_a0.mean():>10.4f}")
print("-" * 56)
print(f"{'ALL':<9}{r.held_2branch.mean():>15.4f}{r.held_global.mean():>13.4f}"
      f"{(r.held_2branch-r.held_global).mean():>+9.4f}{r.held_a0.mean():>10.4f}")
print(f"\npositive in {int((r.held_2branch>r.held_global).sum())}/{len(r)} splits")
print(f"selected alpha_1 (branch 1) mean {r.sel_a1.mean():.3f} | "
      f"alpha_2 (branch 2) mean {r.sel_a2.mean():.3f}  "
      f"-> a1>a2 in {int((r.sel_a1>r.sel_a2).sum())}/{len(r)} splits")
