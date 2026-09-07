"""Robustness to the one hyperparameter of the estimator: how many bins.

The transforms are lookup tables over binned divergence time and binned spatial distance.  The
main results use 17 equal-width tree bins (plus the ceiling) and 24 spatial quantile bins.  If
the gain depends on that choice the estimator is not usable.  Same split protocol, 2 splits per
section, four bin counts.
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
CENTRE = {"M3-4": 0.1608, "M3-1": 0.0297, "M3-2": 0.0293}
MULT = np.array([0.25, 0.5, 0.8, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5, 9.0])
NBINS = [9, 13, 17, 25, 33]
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    return dict(M=M, CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st)


def _init(sec, nbin, half):
    S = load(sec)
    n = len(S["xy_sc"])
    h = np.random.default_rng(0).permutation(n)
    fit_i = h[:n // 2] if half == 0 else h[n // 2:]
    use_i = h[n // 2:] if half == 0 else h[:n // 2]
    CTu = S["CT"][np.ix_(use_i, use_i)]
    if nbin == 0:
        Phi = [standardise(CTu)]; Psi = [standardise(S["CS"])]; rho = np.array([1.0])
    else:
        iu = np.triu_indices(len(fit_i), 1)
        K = fit_kernel(S["CT"][np.ix_(fit_i, fit_i)][iu], pdist(S["xy_sc"][fit_i]),
                       ncomp=2, nbin=nbin)
        Phi = [standardise(apply_lookup(CTu, K["tc"], K["phi"][r])) for r in range(2)]
        Psi = [standardise(apply_lookup(S["CS"], K["sc"], K["psi"][r])) for r in range(2)]
        rho = K["rho"][:2]
    _S["d"] = dict(M=S["M"][use_i], Phi=Phi, Psi=Psi, rho=rho, xy_sc=S["xy_sc"][use_i],
                   xy_st=S["xy_st"], n=len(use_i), m=S["M"].shape[1])


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(p, q, 200, 1e-9)
    am = np.asarray(T).argmax(1)
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))


if __name__ == "__main__":
    rows = []
    for sec in SECS:
        for nbin in [0] + NBINS:
            for half in (0, 1):
                with Pool(7, initializer=_init, initargs=(sec, nbin, half)) as pool:
                    for w, e in pool.imap_unordered(_solve, list(CENTRE[sec] * MULT)):
                        rows.append(dict(section=sec, nbin=nbin, half=half, w=w, err=e))
        g = pd.DataFrame(rows).query("section == @sec")
        b = g.groupby(["nbin", "half"]).err.agg(["min", "median"]).groupby("nbin").mean()
        print(f"\n{sec}")
        print("   bins        " + "".join(f"{('FGW' if k==0 else k):>9}" for k in b.index))
        print("   best        " + "".join(f"{b['min'][k]:>9.1f}" for k in b.index))
        print("   median      " + "".join(f"{b['median'][k]:>9.1f}" for k in b.index), flush=True)
        pd.DataFrame(rows).to_csv(HERE / "binsens.csv", index=False)
    print("\ndone")
