"""Is the FITTED transform the right one, or would any reshaping do?

Sweep a two-parameter family of monotone power transforms

    phi(t) = t^gamma      psi(s) = s^delta

which contains ordinary FGW at gamma = delta = 1, and compare it with the transform the
correspondence analysis fits from the reference half.  If the fitted transform lands near the
best of the family, the estimator is finding the right shape; if some arbitrary power beats it,
the gain is just "any nonlinearity helps".

Protocol is the split one: transforms estimated on half the cells, mapping scored on the other
half, best over the same alpha grid.
"""
import os, sys, itertools
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
GAM = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    return dict(M=M, CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st)


def _init(sec, half, seed):
    S = load(sec)
    n = len(S["xy_sc"])
    h = np.random.default_rng(seed).permutation(n)
    fit_i = h[:n // 2] if half == 0 else h[n // 2:]
    use_i = h[n // 2:] if half == 0 else h[:n // 2]
    S["CTu"] = S["CT"][np.ix_(use_i, use_i)]
    S["Mu"] = S["M"][use_i]
    S["xyu"] = S["xy_sc"][use_i]
    iu = np.triu_indices(len(fit_i), 1)
    S["K"] = fit_kernel(S["CT"][np.ix_(fit_i, fit_i)][iu], pdist(S["xy_sc"][fit_i]), ncomp=2)
    S["n"] = len(use_i); S["m"] = S["M"].shape[1]
    _S["d"] = S


def _solve(job):
    kind, a, b, w = job
    S = _S["d"]
    if kind == "power":
        Phi = [standardise(S["CTu"] ** a)]; Psi = [standardise(S["CS"] ** b)]
        rho = np.array([1.0])
    else:
        K = S["K"]; R = 2
        Phi = [standardise(apply_lookup(S["CTu"], K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(S["CS"], K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    T, _ = GenFGW(S["Mu"], Phi, Psi, rho, w / (1 + w)).solve(p, q, 200, 1e-9)
    am = np.asarray(T).argmax(1)
    return job, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xyu"], axis=1)))


if __name__ == "__main__":
    rows = []
    for sec in SECS:
        jobs = [("power", a, b, w) for a, b in itertools.product(GAM, GAM)
                for w in CENTRE[sec] * MULT]
        jobs += [("fitted", 0, 0, w) for w in CENTRE[sec] * MULT]
        for half in (0,):
            with Pool(7, initializer=_init, initargs=(sec, half, 0)) as pool:
                for job, e in pool.imap_unordered(_solve, jobs, 4):
                    rows.append(dict(section=sec, kind=job[0], gamma=job[1], delta=job[2],
                                     w=job[3], err=e))
        R = pd.DataFrame(rows).query("section == @sec")
        best = R.groupby(["kind", "gamma", "delta"]).err.min().reset_index()
        pw = best[best.kind == "power"].sort_values("err")
        fit = best[best.kind == "fitted"].err.iloc[0]
        fgw = pw[(pw.gamma == 1.0) & (pw.delta == 1.0)].err.iloc[0]
        print(f"\n{sec}   FGW (gamma=delta=1) {fgw:.1f} um   |  fitted transform {fit:.1f} um")
        print(f"   best 5 power transforms:")
        for _, r in pw.head(5).iterrows():
            print(f"      t^{r.gamma:<5} s^{r.delta:<5} {r.err:7.1f}")
        print(f"   the fitted transform beats {np.mean(pw.err.values > fit)*100:.0f}% "
              f"of the {len(pw)} power transforms", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "family.csv", index=False)
