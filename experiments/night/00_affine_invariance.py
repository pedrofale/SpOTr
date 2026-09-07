"""FACT 0.  Under fixed marginals, square-loss GW sees C_tree ONLY through a linear functional.

    sum_ijkl (C1_ik - C2_jl)^2 T_ij T_kl
      = sum_ik C1_ik^2 p_i p_k          (constant: depends on T only through the row marginal)
      - 2 <C1, T C2 T'>
      + sum_jl C2_jl^2 q_j q_l          (constant: column marginal)

so   argmin_T GW(T)  =  argmax_T <C1, T C2 T'>.

CONSEQUENCE 1 (affine invariance).  <aC1 + b, T C2 T'> = a<C1, T C2 T'> + b q'C2 q.
The additive part is constant and the multiplicative part is absorbed by alpha.  Rescaling or
shifting the tree metric CANNOT change the mapping -- only the alpha that selects it.

CONSEQUENCE 2 (what the tree is used for).  <C1, T C2 T'> = E_T[ tree distance x spatial
distance between the images ].  Square-loss GW maximises the (uncentred, hence after
Consequence 1 the centred) COVARIANCE between patristic distance and mapped spatial distance.
Everything the tree knows beyond that single scalar is invisible to the model.

This script checks both numerically on M3-4.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, ot
from pathlib import Path
HERE = Path(__file__).resolve().parent
PET = HERE.parent / "petracer"

d = np.load(PET / "petracerFAIR_M3-4.npz", allow_pickle=True)
M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
xy_sc = np.asarray(d["xy_sc"], float); xy_st = np.asarray(d["xy"], float)
n, m = M.shape
p, q = np.ones(n) / n, np.ones(m) / m
C1 = CT / CT.mean(); C2 = CS / CS.mean()

rng = np.random.default_rng(0)
T = rng.random((n, m)); T = T / T.sum()
T = ot.emd(p, q, rng.random((n, m)))          # a real (non-product) coupling


def gw(A, B, T):
    return float(((A[:, :, None, None] - B[None, None, :, :]) ** 2 * 0).sum()) if False else \
        float((A ** 2 @ T.sum(1)) @ T.sum(1)) - 2 * float((A * (T @ B @ T.T)).sum()) + \
        float((B ** 2 @ T.sum(0)) @ T.sum(0))


print("CHECK 1  the three-term expansion equals POT's gw_loss")
print(f"  ours {gw(C1, C2, T):.10f}   POT {ot.gromov.gwloss(*ot.gromov.init_matrix(C1, C2, p, q), T):.10f}")

print("\nCHECK 2  affine transforms of C_tree only rescale the cross term")
for a, b in [(1.0, 0.0), (2.36, -1.36), (0.4, 5.0), (1.0, 3.0)]:
    A = a * C1 + b
    cross_raw = float((C1 * (T @ C2 @ T.T)).sum())
    cross_aff = float((A * (T @ C2 @ T.T)).sum())
    pred = a * cross_raw + b * float(q @ C2 @ q)
    print(f"  a={a:<5} b={b:<6}  cross {cross_aff:.8f}   predicted {pred:.8f}   "
          f"diff {abs(cross_aff-pred):.2e}")

print("\nCHECK 3  the SOLUTION is identical under an affine transform, at the matched alpha")
print("         (alpha_new/(1-alpha_new) = (alpha/(1-alpha))/a)")


def solve(C1u, alpha):
    T = ot.gromov.fused_gromov_wasserstein(M, C1u, C2, p, q, loss_fun="square_loss",
                                           alpha=alpha, max_iter=200, tol_rel=1e-10,
                                           tol_abs=1e-10, G0=np.outer(p, q))
    am = np.asarray(T).argmax(1)
    return am, float(np.median(np.linalg.norm(xy_st[am] - xy_sc, axis=1)))


for a, b in [(1.0, 0.0), (2.36, -1.36), (0.4, 5.0)]:
    for alpha in [0.45, 0.75]:
        w = alpha / (1 - alpha) / a
        an = w / (1 + w)
        am0, e0 = solve(C1, alpha)
        am1, e1 = solve(np.clip(a * C1 + b, 0, None), an)
        print(f"  a={a:<5} b={b:<6} alpha {alpha:.2f} -> {an:.4f}   err {e0:7.2f} vs {e1:7.2f}"
              f"   assignments identical: {np.mean(am0 == am1):.1%}")

print("\nCHECK 4  what the tree contributes, as one number")
tt = CT[np.triu_indices(n, 1)]
ss = np.linalg.norm(xy_sc[:, None] - xy_sc[None], axis=2)[np.triu_indices(n, 1)]
print(f"  corr(patristic distance, TRUE spatial distance) over all {len(tt):,} cell pairs "
      f"= {np.corrcoef(tt, ss)[0,1]:+.4f}")
