"""Is there ANY per-cell alpha signal worth extracting, or only noise?

The per-cell argmin overfits by construction. Shrink it toward the global optimum:
    alpha_i(lam) = global + lam * (per-cell best - global),  lam in [0,1]
lam=0 is a single alpha; lam=1 is the full oracle. If any interior lam beats lam=0 on the
held-out half, there is real cell-level signal to model. If the curve is monotone downhill
from lam=0, there is not.
"""
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95])
LAM = np.linspace(0, 1, 11)

for SEC in ["M3-4", "M3-1"]:
    H = np.load(HERE / f"exact_halves_{SEC}.npy")
    n = H.shape[2]
    curves = []
    for rep in range(H.shape[0]):
        for A, B in [(0, 1), (1, 0)]:
            EA, EB = H[rep, A], H[rep, B]
            gi = EA.mean(0).argmin()
            bi = EA.argmin(1)                       # per-cell best alpha INDEX on half A
            row = []
            for lam in LAM:
                # interpolate in alpha, then read held-out error by linear interp over the grid
                a = GRID[gi] + lam * (GRID[bi] - GRID[gi])
                e = np.array([np.interp(a[i], GRID, EB[i]) for i in range(n)])
                row.append(e.mean())
            curves.append(row)
    C = np.array(curves).mean(0)
    print(f"\n=== {SEC} ===   held-out placement error (um) vs shrinkage")
    print("  lam   " + "".join(f"{l:>7.1f}" for l in LAM))
    print("  err   " + "".join(f"{v:>7.1f}" for v in C))
    k = int(C.argmin())
    print(f"  best lam = {LAM[k]:.1f} ({C[k]:.1f} um) vs single alpha {C[0]:.1f} um"
          f"  -> {C[0]-C[k]:+.1f} um")
