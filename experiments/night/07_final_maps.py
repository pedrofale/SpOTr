"""Couplings for the demonstration figure, under the protocol used everywhere else:
transforms fitted on a disjoint half of the cells, the other half mapped, spot marginal from the
observed cell density, and the tree weight swept per arm on a common grid."""
import os, sys, pickle
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import GenFGW, fit_kernel, standardise, apply_lookup

CENTRE = {"M3-4": 0.1608, "M3-1": 0.0297, "M3-2": 0.0293}
MULT = np.array([0.25, 0.5, 0.8, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5, 9.0])
_S = {}


def prep(sec, R):
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
    if R == 0:
        Phi = [standardise(CTq)]; Psi = [standardise(CS)]; rho = np.array([1.0])
    else:
        iu = np.triu_indices(len(ref), 1)
        K = fit_kernel(CT[np.ix_(ref, ref)][iu], pdist(xy_sc[ref]), ncomp=R)
        Phi = [standardise(apply_lookup(CTq, K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(CS, K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    return dict(M=M[qry], Phi=Phi, Psi=Psi, rho=rho, q=nc / nc.sum(), xy_sc=xy_sc[qry],
                xy_st=xy_st, n=len(qry), m=M.shape[1], clade=cl[1][qry], names=cl[0])


def _init(sec, R):
    _S["d"] = prep(sec, R)


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]
    if w == 0:
        T = ot.emd(p, S["q"], S["M"], numItermax=1_000_000)
    else:
        T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(p, S["q"], 200, 1e-9)
    am = np.asarray(T).argmax(1)
    return w, am.astype(np.int32), float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"],
                                                                  axis=1)))


if __name__ == "__main__":
    out = {}
    for sec in CENTRE:
        S = prep(sec, 2)
        res = {}
        p = np.ones(S["n"]) / S["n"]
        res["expression only"] = np.asarray(ot.emd(p, S["q"], S["M"],
                                                   numItermax=1_000_000)).argmax(1)
        for R, tag in [(0, "FGW"), (2, "learned")]:
            with Pool(7, initializer=_init, initargs=(sec, R)) as pool:
                best = min(pool.imap_unordered(_solve, list(CENTRE[sec] * MULT)),
                           key=lambda x: x[2])
            res[tag] = best[1]
            print(f"{sec} {tag}: best w = {best[0]:.4f}", flush=True)
        for k, am in res.items():
            e = np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)
            print(f"{sec:<6} {k:<16} median {np.median(e):7.1f}   within one spot (250 um) "
                  f"{np.mean(e < 250):.1%}", flush=True)
        out[sec] = dict(res=res, xy_sc=S["xy_sc"], xy_st=S["xy_st"], clade=S["clade"],
                        clade_names=S["names"])
    pickle.dump(out, open(HERE / "final_maps.pkl", "wb"))
    print("wrote final_maps.pkl")
