"""THE APPLICATION: map the cells of one section onto the SPOTS OF A DIFFERENT SECTION.

This is the unmatched setting the method exists for.  Section A is a matched PEtracer reference:
its cells have expression, a lineage tree, and positions.  Section B is a different section of
the same tumour; all we use from it is what a spatial assay gives you -- spot expression and spot
coordinates.  No cell of A has ever been seen in B, and no position in B is used to build the
cost.

  query   = every cell of A, carrying its expression profile and its place in the tree
  target  = the spots of B
  M       = expression distance between A's cells and B's spots
  C1      = tree distances among A's cells
  C2      = spot distances in B
  phi,psi = fitted on A, the matched reference

There is no cell-level ground truth (A's cells are not in B), so the map is scored on what CAN be
checked: for each clade, does the predicted spatial density over B's spots agree with the density
of B's OWN cells of that clade?  Both are normalised per clade, so the score is unaffected by the
two sections containing different numbers of cells per clade.
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
MULT = np.array([0.0, 0.25, 0.5, 0.8, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5, 9.0])
CENTRE = {"M3-4": 0.1608, "M3-1": 0.0297, "M3-2": 0.0293}
_S = {}


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    return dict(X_sc=np.asarray(d["X_sc"], float), X_st=np.asarray(d["X_st"], float),
                CT=np.asarray(d["C_tree"], float), CS=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st, cell_clade=cc,
                spot_clade=np.asarray(d["spot_clade"]).astype(str))


def true_density(B, clades):
    """Density over B's spots of B's OWN cells of each clade (its held-out half is the query
    half; use the spot-side composition, which is what a spatial assay would show)."""
    m = len(B["xy_st"])
    own = np.argmin(((B["xy_sc"][:, None, :] - B["xy_st"][None, :, :]) ** 2).sum(2), axis=1)
    D = np.zeros((len(clades), m))
    for k, c in enumerate(clades):
        idx = np.where(B["cell_clade"] == c)[0]
        for j in own[idx]:
            D[k, j] += 1
    return D


def _init(src, tgt, arm, R):
    A, B = load(src), load(tgt)
    M = ot.dist(A["X_sc"], B["X_st"]); M /= M.max()
    if arm == "fgw":
        Phi = [standardise(A["CT"])]; Psi = [standardise(B["CS"])]; rho = np.array([1.0])
    else:
        iu = np.triu_indices(len(A["xy_sc"]), 1)
        K = fit_kernel(A["CT"][iu], pdist(A["xy_sc"]), ncomp=R)
        Phi = [standardise(apply_lookup(A["CT"], K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(B["CS"], K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    _S["d"] = dict(M=M, Phi=Phi, Psi=Psi, rho=rho, A=A, B=B, n=M.shape[0], m=M.shape[1])


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    if w == 0:
        T = ot.emd(p, q, S["M"], numItermax=1_000_000)
    else:
        T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(
            p, q, max_iter=200, tol=1e-9)
    return w, np.asarray(T, float)


def score(T, A, B, clades, TRUE):
    """per-clade spatial correlation between the predicted density and B's own cells"""
    P = np.zeros_like(TRUE)
    for k, c in enumerate(clades):
        P[k] = T[A["cell_clade"] == c].sum(0)
    rs, wts = [], []
    for k in range(len(clades)):
        if TRUE[k].sum() < 15 or P[k].sum() <= 0 or TRUE[k].std() == 0:
            continue
        rs.append(pearsonr(P[k] / P[k].sum(), TRUE[k] / TRUE[k].sum()).statistic)
        wts.append(TRUE[k].sum())
    return float(np.average(rs, weights=wts)), len(rs)


if __name__ == "__main__":
    rows = []
    for src in SECS:
        for tgt in SECS:
            if src == tgt:
                continue
            A, B = load(src), load(tgt)
            clades = sorted(set(A["cell_clade"]) & set(B["cell_clade"]))
            TRUE = true_density(B, clades)
            for arm, R, name in [("fgw", 1, "FGW"), ("learned", 2, "learned rank 2"),
                                 ("learned", 3, "learned rank 3")]:
                grid = list(CENTRE[tgt] * MULT)
                with Pool(6, initializer=_init, initargs=(src, tgt, arm, R)) as pool:
                    for w, T in pool.imap_unordered(_solve, grid):
                        r, nk = score(T, A, B, clades, TRUE)
                        rows.append(dict(src=src, tgt=tgt, arm=name, w=w, r=r, nclades=nk))
                g = pd.DataFrame(rows).query("src==@src and tgt==@tgt and arm==@name")
                print(f"{src} -> {tgt}  {name:<16} best clade-density r = {g.r.max():+.3f} "
                      f"(at w={g.loc[g.r.idxmax()].w:.4f}); at w=0 (expression only) "
                      f"{g[g.w==0].r.iloc[0]:+.3f}", flush=True)
                pd.DataFrame(rows).to_csv(HERE / "cross_section.csv", index=False)
            print(flush=True)
