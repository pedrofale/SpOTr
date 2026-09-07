"""THE APPLICATION FIGURE: place one section's cells into a different section, and check the
result against what that section actually contains.

Scored on clade territory: for each clade, the predicted density over the target's spots against
the density of the target's OWN cells of that clade.  Two reference points are needed to read the
number:

  NULL      the same map with the query cells' clade labels permuted -- if the tree is doing
            nothing, this is what you get
  CEILING   the same score for the target's own cells mapped into the target, which is the best
            this metric can reach with a perfect within-section map
"""
import os, sys, pickle
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import GenFGW, fit_kernel, standardise, apply_lookup

SECS = ["M3-4", "M3-1", "M3-2"]
CENTRE = {"M3-4": 0.1608, "M3-1": 0.0297, "M3-2": 0.0293}
MULT = np.array([0.0, 0.5, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5])
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    nc = np.asarray(d["spot_n_cells"], float)
    return dict(X_sc=np.asarray(d["X_sc"], float), X_st=np.asarray(d["X_st"], float),
                CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st, q=nc / nc.sum(),
                cell_clade=np.asarray(d["cell_clade"]).astype(str))


def true_density(B, clades):
    m = len(B["xy_st"])
    own = np.argmin(((B["xy_sc"][:, None, :] - B["xy_st"][None, :, :]) ** 2).sum(2), axis=1)
    D = np.zeros((len(clades), m))
    for k, c in enumerate(clades):
        for j in own[B["cell_clade"] == c]:
            D[k, j] += 1
    return D


def _init(src, tgt, arm):
    A, B = load(src), load(tgt)
    M = ot.dist(A["X_sc"], B["X_st"]); M /= M.max()
    if arm == "fgw":
        Phi = [standardise(A["CT"])]; Psi = [standardise(B["CS"])]; rho = np.array([1.0])
    else:
        iu = np.triu_indices(len(A["xy_sc"]), 1)
        K = fit_kernel(A["CT"][iu], pdist(A["xy_sc"]), ncomp=2)
        Phi = [standardise(apply_lookup(A["CT"], K["tc"], K["phi"][r])) for r in range(2)]
        Psi = [standardise(apply_lookup(B["CS"], K["sc"], K["psi"][r])) for r in range(2)]
        rho = K["rho"][:2]
    _S["d"] = dict(M=M, Phi=Phi, Psi=Psi, rho=rho, A=A, B=B, q=B["q"],
                   n=M.shape[0], m=M.shape[1])


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]; q = S["q"]
    if w == 0:
        T = ot.emd(p, q, S["M"], numItermax=1_000_000)
    else:
        T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(p, q, 200, 1e-9)
    return w, np.asarray(T, float)


def score(P, TRUE, wts=None):
    rs, ws = [], []
    for k in range(len(TRUE)):
        if TRUE[k].sum() < 15 or P[k].sum() <= 0 or TRUE[k].std() == 0:
            continue
        rs.append(pearsonr(P[k] / P[k].sum(), TRUE[k] / TRUE[k].sum()).statistic)
        ws.append(TRUE[k].sum())
    return float(np.average(rs, weights=ws))


def densities(T, labels, clades):
    return np.array([T[labels == c].sum(0) for c in clades])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out, rows = {}, []
    for src, tgt in [("M3-1", "M3-4"), ("M3-2", "M3-1"), ("M3-4", "M3-2")]:
        A, B = load(src), load(tgt)
        clades = sorted(set(A["cell_clade"]) & set(B["cell_clade"]))
        TRUE = true_density(B, clades)
        best = {}
        for arm, name in [("fgw", "FGW"), ("learned", "learned")]:
            with Pool(7, initializer=_init, initargs=(src, tgt, arm)) as pool:
                res = dict(pool.imap_unordered(_solve, list(CENTRE[tgt] * MULT)))
            for w, T in res.items():
                P = densities(T, A["cell_clade"], clades)
                r = score(P, TRUE)
                rows.append(dict(src=src, tgt=tgt, arm=name if w > 0 else "expression only",
                                 w=w, r=r))
                if w == 0:
                    best["expression only"] = (r, T)
                elif name not in best or r > best[name][0]:
                    best[name] = (r, T)
            # null: permute the query cells' clade labels, keep the same coupling
            T = best[name][1]
            nulls = [score(densities(T, rng.permutation(A["cell_clade"]), clades), TRUE)
                     for _ in range(40)]
            rows.append(dict(src=src, tgt=tgt, arm=f"{name} NULL (clades permuted)",
                             w=-1, r=float(np.mean(nulls))))
            print(f"{src} -> {tgt}  {name:<9} r = {best[name][0]:+.3f}   "
                  f"null {np.mean(nulls):+.3f} +- {np.std(nulls):.3f}   "
                  f"(expression only {best['expression only'][0]:+.3f})", flush=True)
        out[(src, tgt)] = dict(best=best, clades=clades, TRUE=TRUE,
                               xy_st=B["xy_st"], labels=A["cell_clade"])
    # ceiling: the same score for a within-section map
    for sec in SECS:
        B = load(sec)
        clades = sorted(set(B["cell_clade"]))
        TRUE = true_density(B, clades)
        iu = np.triu_indices(len(B["xy_sc"]), 1)
        K = fit_kernel(B["CT"][iu], pdist(B["xy_sc"]), ncomp=2)
        M = ot.dist(B["X_sc"], B["X_st"]); M /= M.max()
        Phi = [standardise(apply_lookup(B["CT"], K["tc"], K["phi"][r])) for r in range(2)]
        Psi = [standardise(apply_lookup(B["CS"], K["sc"], K["psi"][r])) for r in range(2)]
        n, m = M.shape
        w = CENTRE[sec] * 1.5
        T, _ = GenFGW(M, Phi, Psi, K["rho"][:2], w / (1 + w)).solve(
            np.ones(n) / n, B["q"], 200, 1e-9)
        r = score(densities(np.asarray(T, float), B["cell_clade"], clades), TRUE)
        rows.append(dict(src=sec, tgt=sec, arm="CEILING (within section)", w=w, r=r))
        print(f"CEILING  {sec} -> itself: r = {r:+.3f}", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "application.csv", index=False)
    pickle.dump(out, open(HERE / "application.pkl", "wb"))
    print("done")
