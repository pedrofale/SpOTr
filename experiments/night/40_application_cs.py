"""Cross-section application under the clone-spread cost: one section's cells into another's spots.

r(a) is fitted on the SOURCE section (the matched reference) and applied to the target's tree and
spots.  No cell is shared and no position in the target is used to build the cost.
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
from spotr.genfgw import GenFGW, standardise
from spotr.clonespread import CloneSpreadOT, fit_spread, spread_weights

SECS = ["M3-4", "M3-1", "M3-2"]
WGRID = [0.0] + list(np.geomspace(1e-3, 30.0, 12))
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
    d = dict(M=M, q=B["q"], n=M.shape[0], m=M.shape[1], arm=arm)
    if arm == "clonespread":
        iu = np.triu_indices(len(A["xy_sc"]), 1)
        sp = fit_spread(A["CT"][iu], pdist(A["xy_sc"]))       # fitted on the SOURCE
        d["W"] = spread_weights(A["CT"], sp); d["Y"] = B["xy_st"]
    else:
        d["Phi"] = [standardise(A["CT"])]; d["Psi"] = [standardise(B["CS"])]
        d["rho"] = np.array([1.0])
    _S["d"] = d


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]
    a = w / (1 + w)
    if w == 0:
        T = ot.emd(p, S["q"], S["M"], numItermax=10 ** 7)
    elif S["arm"] == "clonespread":
        T, _ = CloneSpreadOT(S["M"], S["W"], S["Y"], a).solve(p, S["q"], 200, 1e-9)
    else:
        T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], a).solve(p, S["q"], 200, 1e-9)
    return w, np.asarray(T, float)


def score(P, TRUE):
    rs, ws = [], []
    for k in range(len(TRUE)):
        if TRUE[k].sum() < 15 or P[k].sum() <= 0 or TRUE[k].std() == 0:
            continue
        rs.append(pearsonr(P[k] / P[k].sum(), TRUE[k] / TRUE[k].sum()).statistic)
        ws.append(TRUE[k].sum())
    return float(np.average(rs, weights=ws))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, out = [], {}
    for src, tgt in [("M3-1", "M3-4"), ("M3-2", "M3-1"), ("M3-4", "M3-2")]:
        A, B = load(src), load(tgt)
        clades = sorted(set(A["cell_clade"]) & set(B["cell_clade"]))
        TRUE = true_density(B, clades)
        best = {}
        for arm, name in [("fgw", "GW"), ("clonespread", "clone spread")]:
            with Pool(7, initializer=_init, initargs=(src, tgt, arm)) as pool:
                res = dict(pool.imap_unordered(_solve, WGRID))
            for w, T in res.items():
                P = np.array([T[A["cell_clade"] == c].sum(0) for c in clades])
                r = score(P, TRUE)
                nm = "expression only" if w == 0 else name
                rows.append(dict(src=src, tgt=tgt, arm=nm, w=w, r=r))
                if w == 0:
                    best["expression only"] = (r, T)
                elif nm not in best or r > best[nm][0]:
                    best[nm] = (r, T)
            nulls = [score(np.array([best[name][1][rng.permutation(A["cell_clade"]) == c].sum(0)
                                     for c in clades]), TRUE) for _ in range(30)]
            rows.append(dict(src=src, tgt=tgt, arm=f"{name} NULL", w=-1, r=float(np.mean(nulls))))
            print(f"{src} -> {tgt}  {name:<14} r = {best[name][0]:+.3f}   "
                  f"null {np.mean(nulls):+.3f}   (expression only {best['expression only'][0]:+.3f})",
                  flush=True)
        out[(src, tgt)] = dict(best=best, clades=clades, TRUE=TRUE,
                               xy_st=B["xy_st"], labels=A["cell_clade"])
    pd.DataFrame(rows).to_csv(HERE / "application_cs.csv", index=False)
    pickle.dump(out, open(HERE / "application_cs.pkl", "wb"))
    print("done")
