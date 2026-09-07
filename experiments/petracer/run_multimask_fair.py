import os, sys, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.multimask as mm
W = [0.0, 0.25, 0.5, 1.0, 2.0]
_S = {}
def prep():
    d = np.load(HERE / "petracerFAIR_M3-4.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    cc = np.asarray(d["cell_clade"]).astype(str); _, clade = np.unique(cc, return_inverse=True)
    n, m = M.shape
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), clade=clade, CT=CT,
                p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float))
def _init():
    S = prep()
    S["bf"] = mm.BlockFGW(S["M"], S["C1"], S["C2"], S["clade"],
                          mm.clade_resolved_matrix(S["CT"], S["clade"]))
    _S["d"] = S
def _solve(w):
    S = _S["d"]
    T, _ = S["bf"].set_weights(*w).solve(S["p"], S["q"], max_iter=200, tol=1e-9)
    am = np.asarray(T).argmax(1)
    return w, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))
if __name__ == "__main__":
    rows = []
    with Pool(8, initializer=_init) as pool:
        for w, e in pool.imap_unordered(_solve, list(itertools.product(W, W, W)), 1):
            rows.append(dict(w_near=w[0], w_mid=w[1], w_far=w[2], err=e))
    R = pd.DataFrame(rows); R.to_csv(HERE / "multimask_fair.csv", index=False)
    uni = R[(R.w_near == R.w_mid) & (R.w_mid == R.w_far) & (R.w_near > 0)]
    bu = uni.loc[uni.err.idxmin()]; bb = R.loc[R.err.idxmin()]
    print(f"best UNIFORM  w={bu.w_near:g}  {bu.err:.1f} um")
    print(f"best SPLIT    ({bb.w_near:g}, {bb.w_mid:g}, {bb.w_far:g})  {bb.err:.1f} um"
          f"   gain {bu.err - bb.err:+.1f} um")
