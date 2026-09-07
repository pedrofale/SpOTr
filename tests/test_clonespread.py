"""Verify the clone-spread cost: gradient against finite differences, and monotone descent."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, ot
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from spotr.clonespread import CloneSpreadOT, fit_spread, spread_weights, clade_dispersion

rng = np.random.default_rng(0)
n, m = 40, 25
M = ot.dist(rng.normal(size=(n, 4)), rng.normal(size=(m, 4))); M /= M.max()
Y = rng.normal(size=(m, 2)) * 300.0
A = rng.gamma(3, 4, (n, n)); W = (A + A.T) / 2; np.fill_diagonal(W, 0)
W = W / W[np.triu_indices(n, 1)].mean()
p, q = np.ones(n) / n, np.ones(m) / m

print("TEST 1  analytic gradient vs finite differences")
g = CloneSpreadOT(M, W, Y, alpha=0.6)
T = ot.emd(p, q, rng.random((n, m)))
G = g.grad(T)
eps = 1e-6
err = []
for _ in range(40):
    a, b = rng.integers(n), rng.integers(m)
    Tp = T.copy(); Tp[a, b] += eps
    Tm = T.copy(); Tm[a, b] -= eps
    fd = (g.objective(Tp) - g.objective(Tm)) / (2 * eps)
    err.append(abs(fd - G[a, b]) / max(abs(fd), 1e-9))
print(f"   max relative error over 40 entries: {max(err):.3e}")

print("\nTEST 2  the objective really is the quadratic the line search assumes")
D = ot.emd(p, q, rng.random((n, m))) - T
b1 = float((G * D).sum())
a2 = g.alpha * g.scale * (float((D @ g.y2) @ (g.W @ D.sum(1)))
                                - float(((D @ g.Y) * (g.W @ (D @ g.Y))).sum()))
worst = max(abs(g.objective(T + t * D) - (g.objective(T) + t * b1 + t * t * a2))
            / max(abs(g.objective(T)), 1e-9) for t in [0.1, 0.3, 0.7, 1.0])
print(f"   max relative error of F(T+tD) = F(T) + t<G,D> + t^2 a2 : {worst:.3e}")

print("\nTEST 3  monotone descent")
T3, info = CloneSpreadOT(M, W, Y, alpha=0.6).solve(p, q, max_iter=200, tol=1e-12)
o = np.array(info["objective"])
print(f"   objective {o[0]:.6f} -> {o[-1]:.6f} in {info['n_iter']} iters; "
      f"monotone {bool(np.all(np.diff(o) <= 1e-10))}")

print("\nTEST 4  fit_spread recovers a planted r(a)")
a_true = rng.uniform(2, 35, 400_000)
r_true = 40 + 18 * a_true                                  # planted, microns
s_obs = np.sqrt(2) * r_true * np.sqrt(rng.exponential(1, len(a_true)) / 2) * np.sqrt(2)
sp = fit_spread(a_true * 2, s_obs)
pred = np.interp([5, 15, 30], sp["age"], sp["r"])
print(f"   planted r at a=5,15,30: {[40+18*x for x in (5,15,30)]}")
print(f"   recovered:              {[round(float(x)) for x in pred]}")

print("\nTEST 5  alpha = 0 reduces to plain OT on the expression cost")
T0, _ = CloneSpreadOT(M, W, Y, alpha=0.0).solve(p, q, max_iter=200, tol=1e-12)
Te = ot.emd(p, q, M)
print(f"   max|dT| vs ot.emd: {np.abs(np.asarray(T0) - Te).max():.3e}")
