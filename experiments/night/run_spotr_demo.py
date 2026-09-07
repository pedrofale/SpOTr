"""END TO END: the recommended recipe, from the stored inputs to the headline number.

Half the cells of a section are the matched reference -- they have a tree AND positions, as
barcoded cells in an imaged section do.  The other half are treated as dissociated: expression and
a tree, no position.  Nothing about the second half is used to fit anything.

Two ways to fix the one remaining free parameter are shown:
  swept      the tree weight chosen on the grid (the footing on which alpha is always reported)
  automatic  the weight chosen by matching the induced (tree, space) joint to the reference,
             using no ground truth at all
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, ot
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import learn_relationship, map_cells, select_weight

GRID = {"M3-4": 0.1608, "M3-1": 0.0297, "M3-2": 0.0293}
MULT = np.array([0.25, 0.5, 0.8, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5, 9.0])

for SEC, c in GRID.items():
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    Xc = np.asarray(d["X_sc"], float); Xs = np.asarray(d["X_st"], float)
    CT = np.asarray(d["C_tree"], float)
    xy_sc = np.asarray(d["xy_sc"], float); xy_st = np.asarray(d["xy"], float)
    CS = squareform(pdist(xy_st))
    nc = np.asarray(d["spot_n_cells"], float)
    q = nc / nc.sum()                       # observed spot density, not uniform
    n = len(xy_sc)
    h = np.random.default_rng(0).permutation(n)
    ref, qry = h[:n // 2], h[n // 2:]

    K = learn_relationship(CT[np.ix_(ref, ref)], xy_sc[ref])           # step 1
    M = ot.dist(Xc[qry], Xs); M /= M.max()
    CTq = CT[np.ix_(qry, qry)]
    p = np.ones(len(qry)) / len(qry)

    def err(am):
        return float(np.median(np.linalg.norm(xy_st[am] - xy_sc[qry], axis=1)))

    out = {"expression only": err(np.asarray(ot.emd(p, q, M, numItermax=10**6)).argmax(1))}
    for tag, kern in [("FGW", None), ("learned", K)]:
        best = min((err(map_cells(M, CTq, CS, kernel=kern, w=w, q=q)[0].argmax(1)), w)
                   for w in c * MULT)
        out[f"{tag} (swept)"] = best[0]
    w_auto = select_weight(M, CTq, CS, K, use_kernel=True, q=q)
    out["learned (automatic weight)"] = err(map_cells(M, CTq, CS, kernel=K, w=w_auto,
                                                      q=q)[0].argmax(1))
    print(f"{SEC}:  " + "   ".join(f"{k} {v:.1f} um" for k, v in out.items())
          + f"   [automatic w = {w_auto:.4f}]", flush=True)
