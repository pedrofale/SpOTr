"""Is the weight selector's optimum interior, or does it run to the edge of the grid?

14_select_w.py used a grid centred on each section's known FGW optimum and reported the selector
as near-exact.  On M3-4 the KL was monotone over that grid, so the "choice" was its lower edge.
Extend the grid downward and see whether the criterion has a real minimum.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import GenFGW, fit_kernel, standardise, apply_lookup

SECS = ["M3-4", "M3-1", "M3-2"]
GRID = np.geomspace(0.002, 1.5, 16)
_S = {}


def _init(sec, R, half, seed):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float); xy_sc = np.asarray(d["xy_sc"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    CT = np.asarray(d["C_tree"], float); CS = squareform(pdist(xy_st))
    n = len(xy_sc)
    h = np.random.default_rng(seed).permutation(n)
    fit_i = h[:n // 2] if half == 0 else h[n // 2:]
    use_i = h[n // 2:] if half == 0 else h[:n // 2]
    iu = np.triu_indices(len(fit_i), 1)
    K = fit_kernel(CT[np.ix_(fit_i, fit_i)][iu], pdist(xy_sc[fit_i]), ncomp=2)
    CTu = CT[np.ix_(use_i, use_i)]
    if R == 0:
        Phi = [standardise(CTu)]; Psi = [standardise(CS)]; rho = np.array([1.0])
    else:
        Phi = [standardise(apply_lookup(CTu, K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(CS, K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    _S["d"] = dict(M=M[use_i], Phi=Phi, Psi=Psi, rho=rho, K=K, CTu=CTu, CS=CS,
                   xy_sc=xy_sc[use_i], xy_st=xy_st, n=len(use_i), m=M.shape[1])


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(p, q, 200, 1e-9)
    am = np.asarray(T).argmax(1)
    iu = np.triu_indices(S["n"], 1)
    H, _, _ = np.histogram2d(S["CTu"][iu], S["CS"][am[iu[0]], am[iu[1]]],
                             bins=[S["K"]["te"], S["K"]["se"]])
    H = H / H.sum(); P = S["K"]["P"]
    kl = float((H * np.log((H + 1e-10) / (P + 1e-10))).sum())
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1))), kl


if __name__ == "__main__":
    rows = []
    for sec in SECS:
        for R, name in [(0, "FGW"), (2, "learned")]:
            for seed in (0, 1, 2, 3):
                for half in (0, 1):
                    with Pool(7, initializer=_init, initargs=(sec, R, half, seed)) as pool:
                        for w, e, kl in pool.imap_unordered(_solve, list(GRID)):
                            rows.append(dict(section=sec, arm=name, seed=seed, half=half,
                                             w=w, err=e, kl=kl))
            g = pd.DataFrame(rows).query("section == @sec and arm == @name")
            gaps, sels, edges = [], [], 0
            for key, h_ in g.groupby(["seed", "half"]):
                wsel = h_.loc[h_.kl.idxmin()].w
                gaps.append(h_.loc[h_.kl.idxmin()].err - h_.err.min()); sels.append(wsel)
                edges += int(wsel in (GRID[0], GRID[-1]))
            print(f"{sec} {name:<8} gap to oracle over {len(gaps)} splits: "
                  f"mean {np.mean(gaps):+6.1f} um, median {np.median(gaps):+6.1f}, "
                  f"max {np.max(gaps):+6.1f}   |  exact in {sum(1 for x in gaps if x < 1e-9)}"
                  f"/{len(gaps)}   |  at a grid edge {edges}/{len(gaps)}", flush=True)
            pd.DataFrame(rows).to_csv(HERE / "selector_check.csv", index=False)
    print("done")
