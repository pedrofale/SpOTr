"""Demonstration couplings under the clone-spread cost, same protocol as everywhere else."""
import os, sys, pickle
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import GenFGW, standardise
from spotr.clonespread import CloneSpreadOT, fit_spread, spread_weights

SECS = ["M3-4", "M3-1", "M3-2"]
WGRID = list(np.geomspace(1e-3, 30.0, 14))
_S = {}


def prep(sec, arm):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float); xy_sc = np.asarray(d["xy_sc"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    CT = np.asarray(d["C_tree"], float); CS = squareform(pdist(xy_st))
    nc = np.asarray(d["spot_n_cells"], float)
    cl = np.unique(np.asarray(d["cell_clade"]).astype(str), return_inverse=True)
    n = len(xy_sc)
    h = np.random.default_rng(0).permutation(n)
    ref, qry = h[:n // 2], h[n // 2:]
    CTq = CT[np.ix_(qry, qry)]
    out = dict(M=M[qry], q=nc / nc.sum(), xy_sc=xy_sc[qry], xy_st=xy_st, arm=arm,
               n=len(qry), m=M.shape[1], clade=cl[1][qry], names=cl[0])
    if arm == "clonespread":
        iu = np.triu_indices(len(ref), 1)
        sp = fit_spread(CT[np.ix_(ref, ref)][iu], pdist(xy_sc[ref]))
        out["W"] = spread_weights(CTq, sp); out["Y"] = xy_st
    else:
        out["Phi"] = [standardise(CTq)]; out["Psi"] = [standardise(CS)]
        out["rho"] = np.array([1.0])
    return out


def _init(sec, arm):
    _S["d"] = prep(sec, arm)


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]
    a = w / (1 + w)
    if S["arm"] == "clonespread":
        T, _ = CloneSpreadOT(S["M"], S["W"], S["Y"], a).solve(p, S["q"], 200, 1e-9)
    else:
        T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], a).solve(p, S["q"], 200, 1e-9)
    am = np.asarray(T).argmax(1)
    return w, am.astype(np.int32), float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"],
                                                                  axis=1)))


if __name__ == "__main__":
    out = {}
    for sec in SECS:
        S = prep(sec, "clonespread")
        res = {}
        p = np.ones(S["n"]) / S["n"]
        res["expression only"] = np.asarray(ot.emd(p, S["q"], S["M"], numItermax=10**7)).argmax(1)
        for arm, tag in [("fgw", "GW"), ("clonespread", "clone spread")]:
            with Pool(7, initializer=_init, initargs=(sec, arm)) as pool:
                best = min(pool.imap_unordered(_solve, WGRID), key=lambda x: x[2])
            res[tag] = best[1]
            print(f"{sec} {tag}: w={best[0]:.4f}", flush=True)
        for k, am in res.items():
            e = np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)
            print(f"{sec:<6} {k:<16} median {np.median(e):7.1f}   within one spot "
                  f"{np.mean(e < 250):.1%}", flush=True)
        out[sec] = dict(res=res, xy_sc=S["xy_sc"], xy_st=S["xy_st"], clade=S["clade"],
                        clade_names=S["names"])
    pickle.dump(out, open(HERE / "final_maps_cs.pkl", "wb"))
    print("wrote final_maps_cs.pkl")
