"""Uniform alpha sweep in steps of 0.1, exact solver."""
import numpy as np, ot, pandas as pd, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEC = "M3-4"
GRID = [round(0.1*i, 1) for i in range(11)]

d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float64), np.asarray(d["X_st"], np.float64)
CT = np.asarray(d["C_tree"], np.float64); CS = np.asarray(d["C_space"], np.float64)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
clades = list(np.asarray(d["clades"]).astype(str))
cidx = np.array([clades.index(c) for c in cc])
COMP = np.load(HERE / f"comp_{SEC}.npy").astype(np.float64)
n, m = len(cc), X_st.shape[0]
p, q = np.ones(n)/n, np.ones(m)/m
M = ot.dist(X_sc, X_st); M = M/M.max()
C1, C2 = CT/CT.mean(), CS/CS.mean()
floor = float(COMP.mean(0)[cidx].mean())

print(f"{SEC}: {n} cells x {m} spots, random-spot floor {floor:.4f}\n")
print(f"{'alpha':>7}{'soft accuracy':>16}{'median error (um)':>20}{'sec':>8}")
print("-"*51)
rows = []
for a in GRID:
    t0 = time.time()
    T = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun="square_loss", alpha=a,
                                           max_iter=200, tol_rel=1e-9, tol_abs=1e-9)
    j = T.argmax(1)
    acc = float(COMP[j, cidx].mean())
    err = float(np.median(np.linalg.norm(xy_st[j]-xy_sc, axis=1)))
    rows.append(dict(alpha=a, soft=acc, err=err))
    print(f"{a:>7.1f}{acc:>16.4f}{err:>20.1f}{time.time()-t0:>8.1f}", flush=True)
r = pd.DataFrame(rows); r.to_csv(HERE / f"exact_sweep_fine_{SEC}.csv", index=False)
b = r.loc[r.soft.idxmax()]
print(f"\nbest alpha = {b.alpha:g}  accuracy {b.soft:.4f}  "
      f"(alpha=0 {r.soft.iloc[0]:.4f}, floor {floor:.4f})")
