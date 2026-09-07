"""Calibrate expression noise so alpha=0 leaves the same headroom as PEtracer.

The tree can only be shown to help where expression has NOT already solved the problem.
PEtracer: alpha=0 soft accuracy 0.1459 against a random-spot floor of 0.0977 -> 1.49x floor.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, ot
from pathlib import Path
sys.path.insert(0, ".")
import sim_branching

print(f"{'sigma_expr':>11}{'expr-NN':>10}{'FGW a=0':>10}{'chance':>9}{'x chance':>10}")
print("-" * 50)
for se in [1.3, 3.0, 6.0, 10.0, 16.0, 24.0]:
    d = sim_branching.simulate(0, bl_shape=8.0, sat_q=0.5, sigma_expr=se)
    M = ot.dist(d["X_sc"], d["X_st"]); M = M / M.max()
    n, m = d["n"], d["m"]
    nn = float((M.argmin(1) == d["true_spot"]).mean())
    T = ot.emd(np.ones(n)/n, np.ones(m)/m, M)
    a0 = float((T.argmax(1) == d["true_spot"]).mean())
    print(f"{se:>11.1f}{nn:>10.4f}{a0:>10.4f}{1/m:>9.4f}{a0*m:>10.1f}")
print("\nPEtracer M3-4 for reference: alpha=0 0.1459, floor 0.0977 -> 1.5x floor")
