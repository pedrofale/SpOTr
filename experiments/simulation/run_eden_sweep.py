"""Alpha sweep on the CALIBRATED Eden generator, scored three ways."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src")); sys.path.insert(0, str(HERE))
import sim_eden

GRID = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
SEEDS = [0, 1, 2]
_S = {}

def prep(seed, se):
    d = sim_eden.simulate(seed, sigma_expr=se)
    M = ot.dist(d["X_sc"], d["X_st"]); M = M / M.max()
    n, m, cl = d["n"], d["m"], d["clade"]
    K = cl.max() + 1
    COMP = np.zeros((m, K))
    for j in range(m):
        mem = cl[d["true_spot"] == j]
        COMP[j] = np.bincount(mem, minlength=K) / max(len(mem), 1)
    return dict(M=M, C1=d["C_tree"]/d["C_tree"].mean(), C2=d["C_space"]/d["C_space"].mean(),
                p=np.ones(n)/n, q=np.ones(m)/m, true=d["true_spot"], n=n, m=m,
                xy_st=d["xy_st"], COMP=COMP, clade=cl)

def _init(seed, se): _S["d"] = prep(seed, se)

def _solve(a):
    S = _S["d"]
    T = ot.gromov.fused_gromov_wasserstein(S["M"], S["C1"], S["C2"], S["p"], S["q"],
                                           loss_fun="square_loss", alpha=a,
                                           max_iter=200, tol_rel=1e-9, tol_abs=1e-9)
    am = T.argmax(1)
    return (a, float((am == S["true"]).mean()),
            float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_st"][S["true"]], axis=1))),
            float(S["COMP"][am, S["clade"]].mean()))

if __name__ == "__main__":
    rows = []
    print(f"{'expr':>6}{'a=0':>9}{'best':>9}{'at a':>7}{'rel':>8}"
          f"{'|  err best a':>15}{'soft best a':>13}")
    print("-" * 68)
    for se in [3.0, 6.0, 10.0, 16.0]:
        for seed in SEEDS:
            with Pool(8, initializer=_init, initargs=(seed, se)) as pool:
                for a, ac, er, sf in pool.imap_unordered(_solve, GRID):
                    rows.append(dict(sigma_expr=se, seed=seed, alpha=a, acc=ac, err=er, soft=sf))
        g = pd.DataFrame([r for r in rows if r["sigma_expr"] == se]).groupby("alpha").agg(
            acc=("acc", "mean"), err=("err", "mean"), soft=("soft", "mean"))
        ba = g.acc.idxmax()
        print(f"{se:>6.1f}{g.acc[0.0]:>9.4f}{g.acc[ba]:>9.4f}{ba:>7.2f}"
              f"{g.acc[ba]/g.acc[0.0]:>8.3f}{g.err.idxmin():>15.2f}{g.soft.idxmax():>13.2f}",
              flush=True)
    pd.DataFrame(rows).to_csv(HERE / "eden_sweep.csv", index=False)
