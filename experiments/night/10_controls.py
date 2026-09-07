"""Two things the split result needs.

CONTROL.  Refit the transforms on the other half after PERMUTING the pairing between tree
distance and spatial distance.  Identical estimation, identical rank, identical number of knots,
identical alpha grid -- only the relationship is gone.  If the shuffled arm also beats FGW, the
gain is extra capacity rather than lineage information.

DIAGNOSIS.  Measure how much the fitted transform itself moves between two halves of one section
versus between two sections.  That says whether the failure of cross-section transfer is
estimation noise or a genuine difference between sections.
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


def _init(sec, R, half, seed, shuffle):
    S = load(sec)
    n = len(S["xy_sc"])
    h = np.random.default_rng(seed).permutation(n)
    fit_i = h[:n // 2] if half == 0 else h[n // 2:]
    use_i = h[n // 2:] if half == 0 else h[:n // 2]
    CTu = S["CT"][np.ix_(use_i, use_i)]
    iu = np.triu_indices(len(fit_i), 1)
    t = S["CT"][np.ix_(fit_i, fit_i)][iu]; s = pdist(S["xy_sc"][fit_i])
    if shuffle:
        s = np.random.default_rng(1000 + seed * 10 + half).permutation(s)
    K = fit_kernel(t, s, ncomp=R)
    Phi = [standardise(apply_lookup(CTu, K["tc"], K["phi"][r])) for r in range(R)]
    Psi = [standardise(apply_lookup(S["CS"], K["sc"], K["psi"][r])) for r in range(R)]
    _S["d"] = dict(M=S["M"][use_i], Phi=Phi, Psi=Psi, rho=K["rho"][:R],
                   xy_sc=S["xy_sc"][use_i], xy_st=S["xy_st"], n=len(use_i), m=S["M"].shape[1])


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
    print("=== DIAGNOSIS: how far does the fitted transform move? ===")
    g = np.linspace(3, 70, 200); gs = np.linspace(60, 2100, 200)
    fits = {}
    for sec in SECS:
        S = load(sec); n = len(S["xy_sc"])
        for seed in range(4):
            h = np.random.default_rng(seed).permutation(n)
            for half in (0, 1):
                idx = h[:n // 2] if half == 0 else h[n // 2:]
                iu = np.triu_indices(len(idx), 1)
                K = fit_kernel(S["CT"][np.ix_(idx, idx)][iu], pdist(S["xy_sc"][idx]), ncomp=2)
                fits[(sec, seed, half)] = (np.interp(g, K["tc"], K["phi"][0]),
                                           np.interp(gs, K["sc"], K["psi"][0]))

    def zc(v):
        return (v - v.mean()) / v.std()

    within, between = [], []
    ks = list(fits)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            d1 = np.abs(zc(fits[a][0]) - zc(fits[b][0])).mean()
            (within if a[0] == b[0] else between).append(d1)
    print(f"  mean |phi_A - phi_B| (z-scored) : halves of one section {np.mean(within):.3f}"
          f"   different sections {np.mean(between):.3f}"
          f"   ratio {np.mean(between)/np.mean(within):.2f}x")

    print("\n=== CONTROL: shuffled-pairing transforms, same estimation ===")
    rows = []
    for sec in SECS:
        for seed in (0, 1):
            for half in (0, 1):
                with Pool(6, initializer=_init, initargs=(sec, 2, half, seed, True)) as pool:
                    for w, e in pool.imap_unordered(_solve, list(CENTRE[sec] * MULT)):
                        rows.append(dict(section=sec, seed=seed, half=half, arm="SHUFFLED",
                                         w=w, err=e))
        R = pd.DataFrame(rows).query("section == @sec and w > 0")
        v = R.groupby(["seed", "half"]).err.min().values
        print(f"  {sec}: shuffled control {np.mean(v):7.1f} um   per split: "
              + " ".join(f"{x:.0f}" for x in v), flush=True)
        pd.DataFrame(rows).to_csv(HERE / "split_shuffled.csv", index=False)
