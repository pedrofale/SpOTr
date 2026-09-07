"""GenFGW must reproduce POT's fused_gromov_wasserstein exactly in the rank-1 identity case,
and must descend monotonically with a learned rank-2 kernel.

The reparameterisation: standardising C1 and C2 is affine, hence free (Fact 0), and it maps
w = alpha/(1-alpha) on the raw matrices to w * sd1 * sd2 on the standardised ones.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, ot
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from spotr.genfgw import GenFGW, standardise, fit_kernel, build

rng = np.random.default_rng(0)
n, m = 60, 40
X1 = rng.normal(size=(n, 3)); X2 = rng.normal(size=(m, 3))
M = ot.dist(rng.normal(size=(n, 5)), rng.normal(size=(m, 5))); M /= M.max()
C1 = ot.dist(X1, X1); C2 = ot.dist(X2, X2)
p, q = np.ones(n) / n, np.ones(m) / m
P1, P2 = standardise(C1), standardise(C2)
s1 = C1[np.triu_indices(n, 1)].std(); s2 = C2[np.triu_indices(m, 1)].std()

print("TEST 1  coupling identical to POT (rank-1 identity kernel on standardised matrices)")
ok = True
for alpha in [0.1, 0.3, 0.5, 0.75, 0.9]:
    w = alpha / (1 - alpha) * s1 * s2
    an = w / (1 + w)
    Tg, _ = GenFGW(M, [P1], [P2], [1.0], an, dtype=np.float64).solve(p, q, 300, 1e-12)
    Tp = np.asarray(ot.gromov.fused_gromov_wasserstein(
        M, C1, C2, p, q, loss_fun="square_loss", alpha=alpha, max_iter=300,
        tol_rel=1e-12, tol_abs=1e-12, G0=np.outer(p, q)))
    d = np.abs(Tg - Tp).max()
    ok &= d < 1e-12
    print(f"   alpha {alpha:<5} -> {an:.5f}    max|dT| {d:.3e}")

print("\nTEST 2  monotone descent with a rank-2 learned kernel")
P1b = standardise(np.tanh(P1)); P2b = standardise(np.abs(P2))
_, info = GenFGW(M, [P1, P1b], [P2, P2b], [0.6, 0.25], 0.5, dtype=np.float64).solve(p, q, 200, 1e-12)
o = np.array(info["objective"])
mono = bool(np.all(np.diff(o) <= 1e-11))
print(f"   objective {o[0]:.6f} -> {o[-1]:.6f} in {info['n_iter']} iters; monotone {mono}")

print("\nTEST 3  fit_kernel/build round-trip on a planted relationship")
t = rng.gamma(2, 3, 200_000)
s = t * 40 + rng.normal(0, 60, len(t))          # planted linear link
K = fit_kernel(t, np.abs(s), ncomp=2)
print(f"   planted corr {np.corrcoef(t, np.abs(s))[0,1]:+.3f}   recovered rho_1 {K['rho'][0]:.3f}"
      f"   (should be close)")
Phi, Psi, rho = build(C1, C2, K)
print(f"   build -> {len(Phi)} components, Phi[0] shape {Phi[0].shape}, "
      f"finite {np.isfinite(Phi[0]).all()}")

print("\nALL PASS" if ok and mono else "\nFAILURE")
