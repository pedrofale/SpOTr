"""THE OTHER LEVER: the spot marginal.

Every result so far assumes each spot receives the same number of the cells being mapped.  That is
an assumption about the query set, not about the tissue, and it is false -- real occupancy varies
2-to-15 cells per spot (CV about 0.5).  Three settings, otherwise identical:

  uniform        q_j = 1/m                          (what everything else in this report uses)
  observed       q_j proportional to the number of cells the spot actually contains, which a
                 spatial assay gives you from nuclei counts -- no lineage or query information
  semi-relaxed   no spot marginal at all; every cell is placed, spot occupancy is free

Split protocol, both losses, 2 seeds x 2 halves per section.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import (GenFGW, fit_kernel, standardise, apply_lookup, solve_semirelaxed)

SECS = ["M3-4", "M3-1", "M3-2"]
CENTRE = {"M3-4": 0.1608, "M3-1": 0.0297, "M3-2": 0.0293}
MULT = np.array([0.0, 0.25, 0.5, 0.8, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5, 9.0])
_S = {}


def _init(sec, R, marg, half, seed):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float); xy_sc = np.asarray(d["xy_sc"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    CT = np.asarray(d["C_tree"], float); CS = squareform(pdist(xy_st))
    nc = np.asarray(d["spot_n_cells"], float)
    n = len(xy_sc)
    h = np.random.default_rng(seed).permutation(n)
    fit_i = h[:n // 2] if half == 0 else h[n // 2:]
    use_i = h[n // 2:] if half == 0 else h[:n // 2]
    CTu = CT[np.ix_(use_i, use_i)]
    if R == 0:
        Phi = [standardise(CTu)]; Psi = [standardise(CS)]; rho = np.array([1.0])
    else:
        iu = np.triu_indices(len(fit_i), 1)
        K = fit_kernel(CT[np.ix_(fit_i, fit_i)][iu], pdist(xy_sc[fit_i]), ncomp=R)
        Phi = [standardise(apply_lookup(CTu, K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(CS, K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    q = np.ones(len(nc)) / len(nc) if marg != "observed" else nc / nc.sum()
    _S["d"] = dict(M=M[use_i], Phi=Phi, Psi=Psi, rho=rho, q=q, marg=marg,
                   xy_sc=xy_sc[use_i], xy_st=xy_st, n=len(use_i), m=M.shape[1])


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]
    g = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w))
    if S["marg"] == "semirelaxed":
        T, _ = solve_semirelaxed(g, p, side="cells", max_iter=200, tol=1e-9)
    elif w == 0:
        T = ot.emd(p, S["q"], S["M"], numItermax=1_000_000)
    else:
        T, _ = g.solve(p, S["q"], max_iter=200, tol=1e-9)
    am = np.asarray(T).argmax(1)
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))


if __name__ == "__main__":
    rows = []
    for sec in SECS:
        for R, lname in [(0, "FGW"), (2, "learned")]:
            for marg in ["uniform", "observed", "semirelaxed"]:
                for seed in (0, 1):
                    for half in (0, 1):
                        with Pool(7, initializer=_init,
                                  initargs=(sec, R, marg, half, seed)) as pool:
                            for w, e in pool.imap_unordered(_solve, list(CENTRE[sec] * MULT)):
                                rows.append(dict(section=sec, loss=lname, marg=marg, seed=seed,
                                                 half=half, w=w, err=e))
        g = pd.DataFrame(rows).query("section == @sec and w > 0")
        b = g.groupby(["loss", "marg", "seed", "half"]).err.agg(["min", "median"])
        b = b.groupby(["loss", "marg"]).mean()
        print(f"\n{sec}")
        print(f"   {'':<10}{'uniform':>20}{'observed':>20}{'semi-relaxed':>20}")
        print(f"   {'':<10}" + "".join(f"{'best':>10}{'median':>10}" for _ in range(3)))
        for L in ["FGW", "learned"]:
            r = "".join(f"{b['min'][(L,m)]:>10.1f}{b['median'][(L,m)]:>10.1f}"
                        for m in ["uniform", "observed", "semirelaxed"])
            print(f"   {L:<10}" + r, flush=True)
        pd.DataFrame(rows).to_csv(HERE / "marginals.csv", index=False)
    print("\ndone")
