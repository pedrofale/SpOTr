"""CONTROLLED TEST OF THE PRE-FLIGHT CHECK.

Across datasets, rho_2 of the reference tracks whether the learned loss helps.  That is a
correlation over six points.  Turn it into a manipulation: take PEtracer, where rho_2 = 0.15-0.23
and the learned loss helps, and DEGRADE the reference to clone resolution -- exactly what eTRACER
gives -- keeping everything else identical.  The prediction is that rho_2 collapses and the gain
disappears.

Degraded reference: replace each reference cell's position by its clade's centroid and each pair's
tree distance by its clade pair's mean, i.e. keep only what a clone-labelled in-situ readout knows.
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
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    cl = np.unique(np.asarray(d["cell_clade"]).astype(str), return_inverse=True)[1]
    nc = np.asarray(d["spot_n_cells"], float)
    return dict(M=M, CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st, clade=cl, q=nc / nc.sum())


def reference_pairs(S, idx, resolution, rng):
    """(tree distance, spatial distance) draws from the reference cells."""
    CT = S["CT"][np.ix_(idx, idx)]; xy = S["xy_sc"][idx]; cl = S["clade"][idx]
    iu = np.triu_indices(len(idx), 1)
    if resolution == "cell":
        return CT[iu], pdist(xy)
    # CLONE RESOLUTION, exactly the information eTRACER has: the spatial distances are real (they
    # are between labelled bins), but the tree distance of a pair is not known -- only its clade
    # pair is, so t is a DRAW from the empirical distribution of patristic distances between those
    # two clades.  Both marginals are preserved; what is destroyed is the pair-level link.
    s = pdist(xy)
    a, b = cl[iu[0]], cl[iu[1]]
    K = cl.max() + 1
    pool_ = {}
    for u in range(K):
        for v in range(u, K):
            iu_, iv_ = np.where(cl == u)[0], np.where(cl == v)[0]
            if len(iu_) == 0 or len(iv_) == 0:
                continue
            D = CT[np.ix_(iu_, iv_)]
            val = D[np.triu_indices(len(iu_), 1)] if u == v else D.ravel()
            if len(val):
                pool_[(u, v)] = pool_[(v, u)] = val
    t = np.empty(len(s)); ok = np.zeros(len(s), bool)
    for key in set(zip(a.tolist(), b.tolist())):
        m = (a == key[0]) & (b == key[1])
        v = pool_.get(key)
        if v is None or not m.any():
            continue
        t[m] = rng.choice(v, m.sum()); ok |= m
    return t[ok], s[ok]


def _init(sec, R, resolution, half, seed):
    S = load(sec)
    n = len(S["xy_sc"])
    rng = np.random.default_rng(seed)
    h = rng.permutation(n)
    fit_i = h[:n // 2] if half == 0 else h[n // 2:]
    use_i = h[n // 2:] if half == 0 else h[:n // 2]
    CTu = S["CT"][np.ix_(use_i, use_i)]
    if R == 0:
        Phi = [standardise(CTu)]; Psi = [standardise(S["CS"])]; rho = np.array([1.0])
        K = None
    else:
        t, s = reference_pairs(S, fit_i, resolution, rng)
        K = fit_kernel(t, s, ncomp=R, nbin=17)
        Phi = [standardise(apply_lookup(CTu, K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(S["CS"], K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    _S["d"] = dict(M=S["M"][use_i], Phi=Phi, Psi=Psi, rho=rho, q=S["q"],
                   xy_sc=S["xy_sc"][use_i], xy_st=S["xy_st"], n=len(use_i), m=S["M"].shape[1])


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]
    T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(p, S["q"], 200, 1e-9)
    am = np.asarray(T).argmax(1)
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))


if __name__ == "__main__":
    print("rho of the reference, by resolution")
    for sec in SECS:
        S = load(sec)
        n = len(S["xy_sc"]); rng = np.random.default_rng(0)
        h = rng.permutation(n); fit_i = h[:n // 2]
        for res in ("cell", "clone"):
            t, s = reference_pairs(S, fit_i, res, rng)
            K = fit_kernel(t, s, ncomp=3, nbin=17)
            print(f"   {sec} {res:<6} corr {np.corrcoef(t, s)[0,1]:+.3f}   "
                  f"rho = {K['rho'].round(3)}", flush=True)

    rows = []
    for sec in SECS:
        for R, res, name in [(0, "cell", "FGW"), (2, "cell", "learned, cell reference"),
                             (2, "clone", "learned, clone reference")]:
            for seed in (0, 1):
                for half in (0, 1):
                    with Pool(7, initializer=_init,
                              initargs=(sec, R, res, half, seed)) as pool:
                        for w, e in pool.imap_unordered(_solve, list(CENTRE[sec] * MULT)):
                            rows.append(dict(section=sec, arm=name, seed=seed, half=half,
                                             w=w, err=e))
            g = pd.DataFrame(rows).query("section == @sec and arm == @name")
            st = g.groupby(["seed", "half"]).err.agg(["min", "median"]).mean()
            print(f"{sec} {name:<26} best {st['min']:7.1f}   median over grid {st['median']:7.1f}",
                  flush=True)
            pd.DataFrame(rows).to_csv(HERE / "reference_resolution.csv", index=False)
    print("done")
