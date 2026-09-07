"""IS THE IN-SECTION GAIN REAL, OR IS IT FITTING THE VERY CELLS BEING PLACED?

In 02 the transforms were fitted on the same cells whose positions were then predicted.  In 03
they were fitted on a different SECTION and most of the gain disappeared.  Two explanations:

  (a) sections genuinely differ, and the transform has to be re-estimated per tissue context;
  (b) the in-section gain was estimation on the answer.

Separate them by splitting the query cells of ONE section in half.  Fit phi, psi on half 1's
pairs (their tree distances and true positions); map half 2 onto the same spots and score it.
Same tissue, same section, disjoint cells.  If the gain survives here but not across sections,
(a) holds; if it dies here too, (b) does.

FGW is run on exactly the same half, so the comparison is like for like.
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
MULT = np.array([0.0, 0.25, 0.5, 0.8, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5, 9.0, 13.0])
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    return dict(M=M, CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st)


def _init(sec, arm, R, half, seed):
    S = load(sec)
    n = len(S["xy_sc"])
    rng = np.random.default_rng(seed)
    h = rng.permutation(n)
    fit_i = h[:n // 2] if half == 0 else h[n // 2:]
    use_i = h[n // 2:] if half == 0 else h[:n // 2]
    CTu = S["CT"][np.ix_(use_i, use_i)]
    Mu = S["M"][use_i]
    if arm == "fgw":
        Phi = [standardise(CTu)]; Psi = [standardise(S["CS"])]; rho = np.array([1.0])
    else:
        iu = np.triu_indices(len(fit_i), 1)
        K = fit_kernel(S["CT"][np.ix_(fit_i, fit_i)][iu], pdist(S["xy_sc"][fit_i]), ncomp=R)
        Phi = [standardise(apply_lookup(CTu, K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(S["CS"], K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    _S["d"] = dict(M=Mu, Phi=Phi, Psi=Psi, rho=rho, xy_sc=S["xy_sc"][use_i], xy_st=S["xy_st"],
                   n=len(use_i), m=S["M"].shape[1])


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    if w == 0:
        T = ot.emd(p, q, S["M"], numItermax=1_000_000)
    else:
        T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(p, q, 200, 1e-9)
    am = np.asarray(T).argmax(1)
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))


if __name__ == "__main__":
    rows = []
    for sec in SECS:
        for seed in (0, 1):
            for half in (0, 1):
                for arm, R, name in [("fgw", 1, "FGW"), ("learned", 2, "learned rank 2"),
                                     ("learned", 3, "learned rank 3")]:
                    with Pool(6, initializer=_init, initargs=(sec, arm, R, half, seed)) as pool:
                        for w, e in pool.imap_unordered(_solve, list(CENTRE[sec] * MULT)):
                            rows.append(dict(section=sec, seed=seed, half=half, arm=name,
                                             w=w, err=e))
                    pd.DataFrame(rows).to_csv(HERE / "split_estimation.csv", index=False)
        R_ = pd.DataFrame(rows).query("section == @sec")
        b = R_[R_.w > 0].groupby(["arm", "seed", "half"]).err.min().reset_index()
        print(f"\n{sec}   (transforms fitted on the other half of the same section)")
        for arm in ["FGW", "learned rank 2", "learned rank 3"]:
            v = b[b.arm == arm].err.values
            print(f"   {arm:<16} {np.mean(v):7.1f} um   per split: "
                  + " ".join(f"{x:.0f}" for x in v), flush=True)
