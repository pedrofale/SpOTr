"""CAN THE TRADE-OFF WEIGHT BE CHOSEN WITHOUT ANY GROUND TRUTH ON THE TARGET?

Every earlier attempt in this project failed because the criteria were goodness-of-fit on
EXPRESSION, and adding tree structure necessarily moves cells away from their best expression
match.  The reference gives a criterion that is not about expression at all.

The reference section tells us what the joint distribution of (tree distance, spatial distance)
looks like in this tumour.  A mapping induces its own such joint distribution: take every pair of
mapped cells, read off their tree distance and the distance between the spots they were sent to.
Choose the weight whose induced joint distribution is closest to the reference one.

    w* = argmin_w  D( H_w  ||  P_reference )

No positions of the mapped cells are used.  Reported against the weight the ground truth would
have chosen.
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
    return dict(M=M, CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st)


def _init(sec, R, half, seed):
    S = load(sec)
    n = len(S["xy_sc"])
    h = np.random.default_rng(seed).permutation(n)
    fit_i = h[:n // 2] if half == 0 else h[n // 2:]
    use_i = h[n // 2:] if half == 0 else h[:n // 2]
    iu = np.triu_indices(len(fit_i), 1)
    K = fit_kernel(S["CT"][np.ix_(fit_i, fit_i)][iu], pdist(S["xy_sc"][fit_i]), ncomp=6)
    CTu = S["CT"][np.ix_(use_i, use_i)]
    if R == 0:
        Phi = [standardise(CTu)]; Psi = [standardise(S["CS"])]; rho = np.array([1.0])
    else:
        Phi = [standardise(apply_lookup(CTu, K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(S["CS"], K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    _S["d"] = dict(M=S["M"][use_i], Phi=Phi, Psi=Psi, rho=rho, K=K, CTu=CTu, CS=S["CS"],
                   xy_sc=S["xy_sc"][use_i], xy_st=S["xy_st"], n=len(use_i), m=S["M"].shape[1])


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(p, q, 200, 1e-9)
    am = np.asarray(T).argmax(1)
    err = float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))
    # induced joint distribution of (tree distance, mapped spot distance)
    iu = np.triu_indices(S["n"], 1)
    t = S["CTu"][iu]; sm = S["CS"][am[iu[0]], am[iu[1]]]
    K = S["K"]
    H, _, _ = np.histogram2d(t, sm, bins=[K["te"], K["se"]])
    H = H / H.sum()
    P = K["P"]
    eps = 1e-10
    kl = float((H * np.log((H + eps) / (P + eps))).sum())
    tv = float(0.5 * np.abs(H - P).sum())
    # per-component agreement: how strongly the realised joint expresses each canonical
    # component, against how strongly the reference does (rho_r).  Components the loss does NOT
    # use give an unbiased check -- the loss cannot inflate them by construction.
    a = P.sum(1); b = P.sum(0)
    c_ref, c_map = [], []
    for r in range(len(K["rho"])):
        f = K["phi"][r]; g = K["psi"][r]
        c_ref.append(float((P * np.outer(f, g)).sum()))
        c_map.append(float((H * np.outer(f, g)).sum()))
    return w, err, kl, tv, np.array(c_ref), np.array(c_map)


if __name__ == "__main__":
    rows = []
    for sec in SECS:
        for seed in (0, 1):
            for half in (0, 1):
                for R, name in [(0, "FGW"), (2, "learned rank 2")]:
                    with Pool(6, initializer=_init, initargs=(sec, R, half, seed)) as pool:
                        for w, e, kl, tv, cr, cm in pool.imap_unordered(
                                _solve, list(CENTRE[sec] * MULT)):
                            d_ = dict(section=sec, seed=seed, half=half, arm=name,
                                      w=w, err=e, kl=kl, tv=tv, ncomp=R)
                            for r in range(len(cr)):
                                d_[f"cref{r}"] = cr[r]; d_[f"cmap{r}"] = cm[r]
                            rows.append(d_)
        D = pd.DataFrame(rows).query("section == @sec")
        print(f"\n{sec}")
        for name in ["FGW", "learned rank 2"]:
            g = D[D.arm == name]
            picked, oracle, best = [], [], []
            for (sd, hf), h in g.groupby(["seed", "half"]):
                picked.append(h.loc[h.kl.idxmin()].err)
                oracle.append(h.err.min())
                best.append(h.loc[h.err.idxmin()].w)
            print(f"   {name:<16} criterion picks -> {np.mean(picked):7.1f} um   "
                  f"oracle best {np.mean(oracle):7.1f} um   gap {np.mean(picked)-np.mean(oracle):+6.1f}")
        pd.DataFrame(rows).to_csv(HERE / "select_w.csv", index=False)
