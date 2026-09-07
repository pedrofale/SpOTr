"""Does a LEARNED structural loss beat square-loss FGW?

Same solver, same alpha grid, same everything -- only the structural loss changes:

  alpha = 0            expression only
  FGW                  square loss on the raw tree and spatial distances (rank-1, identity)
  learned rank R       square loss on the canonical transforms phi_r, psi_r, weighted by rho_r
  phi only / psi only  ablations: transform one geometry, leave the other raw

The transforms are fitted here on the SAME section's cells using their true positions.  This run
therefore answers the ORACLE question -- is there anything to gain -- and says nothing about
learnability.  Cross-section transfer is 03_transfer.py.

The alpha grid is expressed as w = alpha/(1-alpha) on STANDARDISED matrices, where by Fact 0 the
standardisation is free and w is comparable across transforms.  w = 0.16 reproduces the
mean-normalised alpha = 0.75 that the raw sweep selects.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import GenFGW, fit_kernel, build, standardise, apply_lookup

SECS = ["M3-4", "M3-1", "M3-2"]
# Per-section grid, centred on the w that reproduces each section's known FGW optimum
# (alpha 0.75 / 0.45 / 0.45 in the mean-normalised parameterisation).  The SAME grid is used for
# every structural loss within a section, so no arm gets a finer search than another.
CENTRE = {"M3-4": 0.1608, "M3-1": 0.0297, "M3-2": 0.0293}
MULT = np.array([0.0, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.3, 3.2, 5.0, 8.0])
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    xy_st = np.asarray(d["xy"], float)
    return dict(M=M, CT=np.asarray(d["C_tree"], float),
                CS=squareform(pdist(xy_st)),                 # spot-spot distances in um
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st)


def make_cost(S, kind, ncomp=2, K=None):
    """Return (Phi, Psi, rho) for the requested structural loss."""
    if K is None:
        iu = np.triu_indices(len(S["xy_sc"]), 1)
        K = fit_kernel(S["CT"][iu], pdist(S["xy_sc"]), ncomp=max(ncomp, 1))
    if kind == "fgw":
        return [standardise(S["CT"])], [standardise(S["CS"])], np.array([1.0])
    if kind == "phi_only":
        A = standardise(apply_lookup(S["CT"], K["tc"], K["phi"][0]))
        return [A], [standardise(S["CS"])], np.array([1.0])
    if kind == "psi_only":
        B = standardise(apply_lookup(S["CS"], K["sc"], K["psi"][0]))
        return [standardise(S["CT"])], [B], np.array([1.0])
    Phi, Psi, rho = build(S["CT"], S["CS"], K, ncomp=ncomp)
    return Phi, Psi, rho


def _init(sec, kind, ncomp):
    S = load(sec)
    S["Phi"], S["Psi"], S["rho"] = make_cost(S, kind, ncomp)
    S["n"], S["m"] = S["M"].shape
    _S["d"] = S


def _solve(w):
    S = _S["d"]
    a = w / (1.0 + w)
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    t0 = time.time()
    T, info = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], a).solve(p, q, max_iter=200, tol=1e-9)
    am = np.asarray(T).argmax(1)
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1))), \
        info["n_iter"], time.time() - t0


ARMS = [("fgw", 1, "FGW (square loss, raw)"),
        ("phi_only", 1, "learned phi (tree side)"),
        ("psi_only", 1, "learned psi (space side)"),
        ("learned", 1, "learned rank 1"),
        ("learned", 2, "learned rank 2"),
        ("learned", 3, "learned rank 3"),
        ("learned", 4, "learned rank 4")]

if __name__ == "__main__":
    rows = []
    for sec in SECS:
        for kind, nc, name in ARMS:
            WGRID = list(CENTRE[sec] * MULT)
            with Pool(7, initializer=_init, initargs=(sec, kind, nc)) as pool:
                for w, e, it, dt in pool.imap_unordered(_solve, WGRID):
                    rows.append(dict(section=sec, cost=name, w=w, err=e, iters=it, secs=dt))
            g = pd.DataFrame(rows).query("section == @sec and cost == @name")
            b = g.loc[g.err.idxmin()]
            print(f"{sec}  {name:<26} best {b.err:7.1f} um at w={b.w:<7.4f}   [{g.secs.sum():.0f}s]",
                  flush=True)
            pd.DataFrame(rows).to_csv(HERE / "learned_loss.csv", index=False)
        print(flush=True)
