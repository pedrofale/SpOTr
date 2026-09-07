"""HEAD TO HEAD: the interpretable clone-spread cost against GW and against the fitted transforms.

Identical protocol for every arm: r(a) / the transforms are estimated on a DISJOINT HALF of the
cells, the other half is mapped and scored, the spot marginal comes from observed cell density, and
the tree weight is swept over ONE common wide grid so no arm gets a finer search.
5 seeds x 2 halves x 3 sections = 30 splits.

  expression only     alpha = 0
  FGW                 square loss on the raw tree and spatial distances
  learned rank 2      correspondence-analysis transforms (two fitted lookup tables)
  clone spread        d^2 / 2 r(a)^2  -- one monotone function in microns
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
from spotr.clonespread import CloneSpreadOT, fit_spread, spread_weights

SECS = ["M3-4", "M3-1", "M3-2"]
WGRID = np.r_[0.0, np.geomspace(1e-3, 30.0, 14)]
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    nc = np.asarray(d["spot_n_cells"], float)
    return dict(M=M, CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st, q=nc / nc.sum())


def _init(sec, arm, half, seed):
    S = load(sec)
    n = len(S["xy_sc"])
    h = np.random.default_rng(seed).permutation(n)
    ref = h[:n // 2] if half == 0 else h[n // 2:]
    use = h[n // 2:] if half == 0 else h[:n // 2]
    CTu = S["CT"][np.ix_(use, use)]
    iu = np.triu_indices(len(ref), 1)
    t_ref = S["CT"][np.ix_(ref, ref)][iu]; s_ref = pdist(S["xy_sc"][ref])
    d = dict(M=S["M"][use], q=S["q"], xy_sc=S["xy_sc"][use], xy_st=S["xy_st"],
             n=len(use), m=S["M"].shape[1], arm=arm)
    if arm == "clonespread":
        sp = fit_spread(t_ref, s_ref)
        d["W"] = spread_weights(CTu, sp)
        d["Y"] = S["xy_st"]
    elif arm == "fgw":
        d["Phi"] = [standardise(CTu)]; d["Psi"] = [standardise(S["CS"])]
        d["rho"] = np.array([1.0])
    else:
        K = fit_kernel(t_ref, s_ref, ncomp=2)
        d["Phi"] = [standardise(apply_lookup(CTu, K["tc"], K["phi"][r])) for r in range(2)]
        d["Psi"] = [standardise(apply_lookup(S["CS"], K["sc"], K["psi"][r])) for r in range(2)]
        d["rho"] = K["rho"][:2]
    _S["d"] = d


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]
    a = w / (1.0 + w)
    t0 = time.time()
    if w == 0:
        T = ot.emd(p, S["q"], S["M"], numItermax=10 ** 7)
    elif S["arm"] == "clonespread":
        T, _ = CloneSpreadOT(S["M"], S["W"], S["Y"], a).solve(p, S["q"], 200, 1e-9)
    else:
        T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], a).solve(p, S["q"], 200, 1e-9)
    am = np.asarray(T).argmax(1)
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1))), \
        time.time() - t0


ARMS = [("fgw", "FGW"), ("learned", "learned rank 2"), ("clonespread", "clone spread")]

if __name__ == "__main__":
    rows = []
    for sec in SECS:
        for arm, name in ARMS:
            for seed in range(5):
                for half in (0, 1):
                    with Pool(6, initializer=_init, initargs=(sec, arm, half, seed)) as pool:
                        for w, e, dt in pool.imap_unordered(_solve, list(WGRID)):
                            rows.append(dict(section=sec, arm=name, seed=seed, half=half,
                                             w=w, err=e, secs=dt))
            pd.DataFrame(rows).to_csv(HERE / "headtohead.csv", index=False)
            g = pd.DataFrame(rows).query("section == @sec and arm == @name and w > 0")
            st = g.groupby(["seed", "half"]).err.agg(["min", "median"]).mean()
            print(f"{sec}  {name:<16} best {st['min']:7.1f}   median over grid {st['median']:7.1f}"
                  f"   [{g.secs.mean():.1f}s per solve]", flush=True)
        e0 = pd.DataFrame(rows).query("section == @sec and w == 0").err.mean()
        print(f"{sec}  {'expression only':<16} {e0:7.1f}\n", flush=True)
    print("done")
