"""Does alpha have a meaningful optimum in the FGW cost, with NO entropic solver at all?

POT's fused_gromov_wasserstein is a reference implementation: exact conditional gradient with an
EMD solve each iteration. No eps, no exp_smooth_clipped, no alternation of ours. If the alpha
landscape from the shipped code is a solver artefact, this is where it shows.

POT minimises (1-alpha)*<M,T> + alpha*sum L(C1_ik, C2_jl) T_ij T_kl -- same alpha convention.
POT does not normalise C1/C2, and the GW loss compares them entrywise, so their relative scale
matters. Two normalisations are run to check the conclusion is not an artefact of that choice.
"""
import sys, time
import numpy as np, ot, anndata as ad
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEC = "M3-4"
GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]

d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc = np.asarray(d["X_sc"], np.float64); X_st = np.asarray(d["X_st"], np.float64)
CT = np.asarray(d["C_tree"], np.float64); CS = np.asarray(d["C_space"], np.float64)
cc = np.asarray(d["cell_clade"]).astype(str)
clades = list(np.asarray(d["clades"]).astype(str))
cidx = np.array([clades.index(c) for c in cc])
COMP = np.load(HERE / f"comp_{SEC}.npy").astype(np.float64)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
n, m = len(cc), X_st.shape[0]
p = np.ones(n)/n; q = np.ones(m)/m

M = ot.dist(X_sc, X_st); M = M/M.max()            # identical to prepare_ot_inputs
floor = float(COMP.mean(0)[cidx].mean())
print(f"{SEC}: {n} cells x {m} spots, random-spot floor {floor:.4f}\n")

def soft(T):
    j = T.argmax(1)
    return float(COMP[j, cidx].mean()), float(np.median(np.linalg.norm(xy_st[j]-xy_sc, axis=1)))

for tag, C1, C2 in [("max-norm", CT/CT.max(), CS/CS.max()),
                    ("mean-norm", CT/CT.mean(), CS/CS.mean())]:
    print(f"--- C_tree, C_space normalised by their {tag.split('-')[0]} ---")
    print(f"{'alpha':>7}{'soft acc':>11}{'median err (um)':>18}{'gw dist':>12}{'iters':>8}{'sec':>8}")
    rows = []
    for a in GRID:
        t0 = time.time()
        T, log = ot.gromov.fused_gromov_wasserstein(
            M, C1, C2, p, q, loss_fun="square_loss", alpha=a,
            log=True, max_iter=200, tol_rel=1e-9, tol_abs=1e-9)
        acc, err = soft(T)
        rows.append(dict(norm=tag, alpha=a, soft=acc, err=err, fgw=float(log["fgw_dist"])))
        print(f"{a:>7.2f}{acc:>11.4f}{err:>18.1f}{float(log['fgw_dist']):>12.5f}"
              f"{len(log.get('loss',[])):>8}{time.time()-t0:>8.1f}", flush=True)
    r = rows
    best = max(r, key=lambda x: x["soft"])
    print(f"  best alpha = {best['alpha']:g}  acc {best['soft']:.4f}  "
          f"(alpha=0 {r[0]['soft']:.4f}, floor {floor:.4f})   "
          f"gain over alpha=0 {best['soft']-r[0]['soft']:+.4f}\n")
    import pandas as pd
    pd.DataFrame(rows).to_csv(HERE / f"exact_fgw_{tag}_{SEC}.csv", index=False)
