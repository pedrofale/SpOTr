"""Does square-loss FGW extract anything from a REALISTIC lineage tree?

Same generator, two trees: saturated (matches PEtracer) and unsaturated (control). If GW helps
on the unsaturated tree and not the saturated one, saturation is the culprit. If it fails on
both, the problem is the concave tree-space relation a branching process always produces --
and square-loss GW is simply the wrong reader for lineage data.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src")); sys.path.insert(0, str(HERE))
import sim_branching

GRID = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
SEEDS = [0, 1, 2]
_S = {}


def prep(seed, sat, se=1.3):
    d = sim_branching.simulate(seed, bl_shape=8.0, sat_q=sat, sigma_expr=se)
    M = ot.dist(d["X_sc"], d["X_st"]); M = M / M.max()
    n, m = d["n"], d["m"]
    # composition matrix: COMP[j, k] = fraction of spot j's true members in clade k
    cl = d["clade"]; K = cl.max() + 1
    COMP = np.zeros((m, K))
    for j in range(m):
        mem = cl[d["true_spot"] == j]
        COMP[j] = np.bincount(mem, minlength=K) / max(len(mem), 1)
    return dict(M=M, C1=d["C_tree"]/d["C_tree"].mean(), C2=d["C_space"]/d["C_space"].mean(),
                p=np.ones(n)/n, q=np.ones(m)/m, true=d["true_spot"], n=n, m=m,
                xy_st=d["xy_st"], COMP=COMP, clade=cl)


def _init(seed, sat, se):
    _S["d"] = prep(seed, sat, se)


def _solve(a):
    S = _S["d"]
    T = ot.gromov.fused_gromov_wasserstein(S["M"], S["C1"], S["C2"], S["p"], S["q"],
                                           loss_fun="square_loss", alpha=a,
                                           max_iter=200, tol_rel=1e-9, tol_abs=1e-9)
    am = T.argmax(1)
    err = np.linalg.norm(S["xy_st"][am] - S["xy_st"][S["true"]], axis=1)
    soft = float(S["COMP"][am, S["clade"]].mean())
    return a, float((am == S["true"]).mean()), float(np.median(err)), soft


if __name__ == "__main__":
    rows = []
    print(f"{'expr noise':>11}{'tree':>13}" + "".join(f"{h:>13}" for h in
          ["exact-spot", "placement", "SOFT (clade)"]))
    print(f"{'':>24}" + "".join(f"{'best a':>13}" for _ in range(3)))
    print("-" * 76)
    for se in [3.0, 6.0, 10.0, 16.0]:
        for sat, tag in [(0.5, "SATURATED"), (None, "unsaturated")]:
            acc = {a: [] for a in GRID}
            for seed in SEEDS:
                with Pool(8, initializer=_init, initargs=(seed, sat, se)) as pool:
                    for a, ac, er, sf in pool.imap_unordered(_solve, GRID):
                        acc[a].append(ac)
                        rows.append(dict(sat=str(sat), sigma_expr=se, seed=seed,
                                         alpha=a, acc=ac, err=er, soft=sf))
            R_ = pd.DataFrame([r for r in rows if r["sigma_expr"] == se
                               and r["sat"] == str(sat)])
            g = R_.groupby("alpha").agg(acc=("acc", "mean"), err=("err", "mean"),
                                        soft=("soft", "mean"))
            b_acc = g.acc.idxmax(); b_err = g.err.idxmin(); b_soft = g.soft.idxmax()
            print(f"{se:>11.1f}{tag:>13}"
                  f"{b_acc:>13.2f}{b_err:>13.2f}{b_soft:>13.2f}", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "branching_sweep.csv", index=False)
