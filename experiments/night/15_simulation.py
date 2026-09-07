"""INDEPENDENT TESTBED: does the learned loss beat FGW on data from a different process?

sim_eden.py generates a tumour by Eden growth under volume exclusion, adds local mixing
(motility/packing) and tree-reconstruction error, then cuts spots by a k-d partition.  It was
calibrated to PEtracer's signature (clade coherence, global and local tree-space concordance,
territoriality, fraction of pairs at the saturation ceiling) BEFORE any of tonight's work, so it
is an independent test of the method rather than a restatement of it.

Two questions:
  1. split protocol, as on the real data: transforms fitted on half the cells, other half mapped
  2. cross-TUMOUR transfer: transforms fitted on one simulated tumour, applied to another.
     The real data cannot answer this -- it has one tumour.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
sys.path.insert(0, str(HERE.parent / "simulation"))
from spotr.genfgw import GenFGW, fit_kernel, standardise, apply_lookup
import sim_eden

SEEDS = [0, 1, 2]
MULT = np.array([0.0, 0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 0.6, 1.0, 1.7, 3.0])
_S = {}


def gen(seed):
    return sim_eden.simulate(seed=seed, n=1200, m_spots=128, dim=3, sigma_expr=6.0)


def _init(seed, arm, src_seed, shuffle, half, split_seed):
    S = gen(seed)
    n = S["n"]
    h = np.random.default_rng(split_seed).permutation(n)
    fit_i = h[:n // 2] if half == 0 else h[n // 2:]
    use_i = h[n // 2:] if half == 0 else h[:n // 2]
    CTu = S["C_tree"][np.ix_(use_i, use_i)]
    M = ot.dist(np.asarray(S["X_sc"], float), np.asarray(S["X_st"], float)); M /= M.max()
    if arm == "fgw":
        Phi = [standardise(CTu)]; Psi = [standardise(S["C_space"])]; rho = np.array([1.0])
    else:
        if src_seed is None:                       # fit on the other half of this tumour
            R = gen(seed); idx = fit_i
        else:                                      # fit on a DIFFERENT tumour
            R = gen(src_seed); idx = np.arange(R["n"])
        iu = np.triu_indices(len(idx), 1)
        t = R["C_tree"][np.ix_(idx, idx)][iu]; s = pdist(R["xy_sc"][idx])
        if shuffle:
            s = np.random.default_rng(7).permutation(s)
        K = fit_kernel(t, s, ncomp=2)
        Phi = [standardise(apply_lookup(CTu, K["tc"], K["phi"][r])) for r in range(2)]
        Psi = [standardise(apply_lookup(S["C_space"], K["sc"], K["psi"][r])) for r in range(2)]
        rho = K["rho"][:2]
    _S["d"] = dict(M=M[use_i], Phi=Phi, Psi=Psi, rho=rho, xy_sc=S["xy_sc"][use_i],
                   xy_st=S["xy_st"], n=len(use_i), m=S["m"])


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
    s0 = gen(0)
    iu = np.triu_indices(s0["n"], 1)
    t0 = s0["C_tree"][iu]; sp0 = pdist(s0["xy_sc"])
    K0 = fit_kernel(t0, sp0, ncomp=3)
    print(f"simulated signature: corr(tree,space) {np.corrcoef(t0, sp0)[0,1]:+.3f}   "
          f"maximal {K0['rho'][0]:.3f}   rho2 {K0['rho'][1]:.3f}   "
          f"pairs at ceiling {np.mean(t0 >= t0.max()-1e-9):.1%}", flush=True)
    rows = []
    for seed in SEEDS:
        for half in (0, 1):
            for arm, src, sh, name in [("fgw", None, False, "FGW"),
                                       ("learned", None, False, "learned (same tumour)"),
                                       ("learned", None, True, "SHUFFLED control"),
                                       ("learned", (seed + 1) % 3, False, "learned (other tumour)")]:
                with Pool(6, initializer=_init,
                          initargs=(seed, arm, src, sh, half, 100 + half)) as pool:
                    for w, e in pool.imap_unordered(_solve, list(MULT)):
                        rows.append(dict(seed=seed, half=half, arm=name, w=w, err=e))
        D = pd.DataFrame(rows).query("seed == @seed and w > 0")
        print(f"\ntumour seed {seed}   (expression only "
              f"{pd.DataFrame(rows).query('seed == @seed and w == 0').err.mean():.1f})")
        for name in ["FGW", "learned (same tumour)", "learned (other tumour)", "SHUFFLED control"]:
            g = D[D.arm == name].groupby("half").err
            print(f"   {name:<24} best {g.min().mean():7.1f}   median over grid "
                  f"{g.median().mean():7.1f}", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "simulation.csv", index=False)
