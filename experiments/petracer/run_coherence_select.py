"""Can clade coherence SELECT the resolution weights, with no ground truth?

Matching the tissue's true clade coherence selects a global alpha to within 4 um of optimal.
If it can also select (w_near, w_mid, w_far), the decomposition becomes learnable -- and each
weight is interpretable as how far to trust the tree at that resolution level.

For every weight triple: the true error (evaluation only) and the map's clade coherence
(computable without positions). Selection = pick the triple whose coherence is closest to target.
"""
import os, sys, itertools
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.multimask as mm

W = [0.0, 0.5, 1.0, 2.0]
_S = {}


def prep(sec):
    d = np.load(HERE / f"petracer_{sec}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    cc = np.asarray(d["cell_clade"]).astype(str); _, clade = np.unique(cc, return_inverse=True)
    n, m = M.shape
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), clade=clade, CT=CT, cc=cc,
                p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float))


def _init(sec):
    S = prep(sec)
    S["bf"] = mm.BlockFGW(S["M"], S["C1"], S["C2"], S["clade"],
                          mm.clade_resolved_matrix(S["CT"], S["clade"]))
    _S["d"] = S


def coherence(pos, cc):
    g = np.median(np.linalg.norm(pos - np.median(pos, 0), axis=1))
    return float(np.mean([np.median(np.linalg.norm(pos[cc == c] - np.median(pos[cc == c], 0),
                                                   axis=1)) / g
                          for c in np.unique(cc) if (cc == c).sum() >= 30]))


def _solve(w):
    S = _S["d"]
    T, _ = S["bf"].set_weights(*w).solve(S["p"], S["q"], max_iter=200, tol=1e-9)
    pos = S["xy_st"][np.asarray(T).argmax(1)]
    return (w, float(np.median(np.linalg.norm(pos - S["xy_sc"], axis=1))),
            coherence(pos, S["cc"]))


if __name__ == "__main__":
    for SEC in ["M3-4", "M3-1", "M3-2"]:
        S = prep(SEC)
        target = coherence(S["xy_sc"], S["cc"])
        rows = []
        with Pool(8, initializer=_init, initargs=(SEC,)) as pool:
            for w, err, coh in pool.imap_unordered(_solve, list(itertools.product(W, W, W)), 1):
                rows.append(dict(w=str(w), w_near=w[0], w_mid=w[1], w_far=w[2],
                                 err=err, coh=coh))
        R = pd.DataFrame(rows); R.to_csv(HERE / f"cohsel_{SEC}.csv", index=False)
        act = R[(R.w_near > 0) | (R.w_mid > 0) | (R.w_far > 0)]
        pick = act.iloc[(act.coh - target).abs().argmin()]
        bestw = act.iloc[act.err.argmin()]
        uni = act[(act.w_near == act.w_mid) & (act.w_mid == act.w_far)]
        upick = uni.iloc[(uni.coh - target).abs().argmin()]
        print(f"\n=== {SEC} ===  target clade coherence {target:.4f}")
        print(f"  coherence-matching picks   {pick.w:<16} err {pick.err:.1f} um "
              f"(coh {pick.coh:.3f})")
        print(f"  the best weights would be  {bestw.w:<16} err {bestw.err:.1f} um")
        print(f"  best UNIFORM by coherence  {upick.w:<16} err {upick.err:.1f} um")
        print(f"  no tree at all             {'(0,0,0)':<16} "
              f"err {R[R.w == '(0.0, 0.0, 0.0)'].err.iloc[0]:.1f} um")
        print(f"  --> coherence selection costs {pick.err - bestw.err:+.1f} um vs the oracle "
              f"best", flush=True)
