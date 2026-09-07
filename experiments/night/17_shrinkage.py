"""If a transferred transform is a slightly WRONG version of the right shape, shrink it.

The structural term is a weighted sum of square-loss GW terms, so the square loss and the learned
components can simply sit in the same sum:

    Struct_lambda =  (1 - lambda) * GW( z(C_tree) , z(C_space) )        <- ordinary FGW
                   +      lambda  * sum_r rho_r GW( phi_r(C_tree) , psi_r(C_space) )

lambda = 0 is FGW exactly; lambda = 1 is the fully transferred loss.  An interior optimum means
the transferred transform is worth using but should not be trusted completely -- the textbook
bias-variance trade-off, expressed in the loss rather than in the parameters.

Run on all six ordered section pairs, and on the within-section split protocol for reference.
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
LAM = [0.0, 0.25, 0.5, 0.75, 1.0]
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    return dict(M=M, CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st)


def _init(target, source):
    T_ = load(target); S_ = load(source)
    iu = np.triu_indices(len(S_["xy_sc"]), 1)
    K = fit_kernel(S_["CT"][iu], pdist(S_["xy_sc"]), ncomp=2)
    Phi = [standardise(T_["CT"])] + [standardise(apply_lookup(T_["CT"], K["tc"], K["phi"][r]))
                                     for r in range(2)]
    Psi = [standardise(T_["CS"])] + [standardise(apply_lookup(T_["CS"], K["sc"], K["psi"][r]))
                                     for r in range(2)]
    _S["d"] = dict(M=T_["M"], Phi=Phi, Psi=Psi, rho=K["rho"][:2],
                   xy_sc=T_["xy_sc"], xy_st=T_["xy_st"], n=T_["M"].shape[0], m=T_["M"].shape[1])


def _solve(job):
    lam, w = job
    S = _S["d"]
    rho = np.r_[1.0 - lam, lam * S["rho"]]
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], rho, w / (1 + w)).solve(p, q, 200, 1e-9)
    am = np.asarray(T).argmax(1)
    return job, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))


if __name__ == "__main__":
    rows = []
    for tgt in SECS:
        for src in SECS:
            if src == tgt:
                continue
            jobs = [(l, w) for l in LAM for w in CENTRE[tgt] * MULT]
            with Pool(7, initializer=_init, initargs=(tgt, src)) as pool:
                for (lam, w), e in pool.imap_unordered(_solve, jobs, 3):
                    rows.append(dict(target=tgt, source=src, lam=lam, w=w, err=e))
            g = pd.DataFrame(rows).query("target == @tgt and source == @src")
            b = g.groupby("lam").err.agg(["min", "median"])
            print(f"{src} -> {tgt}")
            print("   lambda      " + "".join(f"{l:>9.2f}" for l in LAM))
            print("   best        " + "".join(f"{b['min'][l]:>9.1f}" for l in LAM))
            print("   median      " + "".join(f"{b['median'][l]:>9.1f}" for l in LAM), flush=True)
            pd.DataFrame(rows).to_csv(HERE / "shrinkage.csv", index=False)
    print("\ndone")
