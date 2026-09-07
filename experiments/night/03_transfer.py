"""THE HONEST QUESTION: can the relationship be LEARNED in one section and USED in another?

This is the setting the method is for.  Section A is a matched reference -- cells with
expression, a lineage tree AND their positions (PEtracer in situ).  Section B is a new section
whose cells are to be placed: expression and a tree, no positions.  Everything about the
structural loss is estimated on A; nothing on B is used except what a real application would
have.

  1. fit phi_r, psi_r on A's cell pairs                       (A's positions -- allowed)
  2. transfer the lookup tables to B's tree and spot distances
  3. solve, and score against B's held-out true positions     (evaluation only)

Both arms are swept over the SAME grid of the expression/tree trade-off w on the target, and the
whole curve is reported, so the comparison does not depend on how w is chosen.  A single number
under the strict protocol (w also picked on A) is reported alongside.

CONTROL: `shuffled` refits the transforms on A after permuting the pairing between tree distance
and spatial distance.  The marginals, the bin counts, the number of parameters and the rank are
all preserved -- only the relationship is destroyed.
"""
import os, sys, time
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
MULT = np.array([0.0, 0.25, 0.5, 0.8, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5, 9.0, 13.0])
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    xy_st = np.asarray(d["xy"], float)
    return dict(M=M, CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st)


def kernel_of(secs, ncomp, shuffle_seed=None):
    """Fit the transforms on one reference section, or on several pooled into an atlas."""
    if isinstance(secs, str):
        secs = [secs]
    T, X = [], []
    for sec in secs:
        S = load(sec)
        iu = np.triu_indices(len(S["xy_sc"]), 1)
        T.append(S["CT"][iu]); X.append(pdist(S["xy_sc"]))
    t = np.concatenate(T); s = np.concatenate(X)
    if shuffle_seed is not None:
        s = np.random.default_rng(shuffle_seed).permutation(s)
    return fit_kernel(t, s, ncomp=ncomp)


def _init(target, arm, ncomp, source, seed):
    S = load(target)
    if arm == "fgw":
        S["Phi"] = [standardise(S["CT"])]; S["Psi"] = [standardise(S["CS"])]
        S["rho"] = np.array([1.0])
    else:
        K = kernel_of(source, ncomp, seed)
        S["Phi"] = [standardise(apply_lookup(S["CT"], K["tc"], K["phi"][r])) for r in range(ncomp)]
        S["Psi"] = [standardise(apply_lookup(S["CS"], K["sc"], K["psi"][r])) for r in range(ncomp)]
        S["rho"] = K["rho"][:ncomp]
    S["n"], S["m"] = S["M"].shape
    _S["d"] = S


def _solve(w):
    S = _S["d"]
    a = w / (1.0 + w)
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], a).solve(p, q, max_iter=200, tol=1e-9)
    am = np.asarray(T).argmax(1)
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))


if __name__ == "__main__":
    jobs = [(b, "fgw", 1, None, None, "FGW (no transfer)") for b in SECS]
    for b in SECS:
        others = [a for a in SECS if a != b]
        for a in others:
            for R in (2, 3):
                jobs.append((b, "learned", R, a, None, f"learned rank {R} from {a}"))
        for R in (2, 3):
            jobs.append((b, "learned", R, others, None, f"learned rank {R} from BOTH others"))
        jobs.append((b, "learned", 3, others, 12345, "SHUFFLED control"))
    rows = []
    for target, arm, R, source, seed, name in jobs:
        t0 = time.time()
        grid = list(CENTRE[target] * MULT)
        with Pool(7, initializer=_init, initargs=(target, arm, R, source, seed)) as pool:
            for w, e in pool.imap_unordered(_solve, grid):
                rows.append(dict(target=target, arm=name, source=str(source), ncomp=R,
                                 shuffled=seed is not None, w=w, err=e))
        g = pd.DataFrame(rows).query("target == @target and arm == @name")
        print(f"target {target}   {name:<28} best over w: {g.err.min():7.1f} um   "
              f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "transfer.csv", index=False)
    print("\ndone")
