"""Does CLADE COHERENCE of the map track mapping quality?

This is Pedro's actual proposal and I never tested it -- the "coherence" in the earlier loss
probe was expression agreement within a spot, a different quantity. I asserted clade coherence
would rise monotonically with alpha (so be useless as an objective) without checking.

Clade coherence is computable with NO ground truth: take the mapped positions of a clade's
cells and ask how tightly they cluster. If it has an interior optimum near the truth's, alpha
is selectable without positions -- which was the original goal.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
GRID = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
_S = {}


def prep():
    d = np.load(HERE / "petracer_M3-4.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    n, m = M.shape
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float),
                cc=np.asarray(d["cell_clade"]).astype(str))


def _init(): _S["d"] = prep()


def _solve(a):
    S = _S["d"]
    T = np.asarray(ot.gromov.fused_gromov_wasserstein(
        S["M"], S["C1"], S["C2"], S["p"], S["q"], loss_fun="square_loss", alpha=a,
        max_iter=200, tol_rel=1e-9, tol_abs=1e-9))
    pos = S["xy_st"][T.argmax(1)]
    err = float(np.median(np.linalg.norm(pos - S["xy_sc"], axis=1)))
    # CLADE COHERENCE OF THE MAP -- no ground truth used
    glob = np.median(np.linalg.norm(pos - np.median(pos, 0), axis=1))
    coh = []
    for c in np.unique(S["cc"]):
        s = S["cc"] == c
        if s.sum() < 30: continue
        coh.append(np.median(np.linalg.norm(pos[s] - np.median(pos[s], 0), axis=1)) / glob)
    # how much of the tissue the map actually uses (collapse detector)
    spread = float(np.median(np.linalg.norm(pos - np.median(pos, 0), axis=1)))
    used = float(len(np.unique(T.argmax(1))))
    return a, err, float(np.mean(coh)), spread, used


if __name__ == "__main__":
    rows = []
    with Pool(8, initializer=_init) as pool:
        for a, err, coh, spread, used in pool.imap_unordered(_solve, GRID):
            rows.append(dict(alpha=a, err=err, clade_coh=coh, spread=spread, spots_used=used))
    R = pd.DataFrame(rows).sort_values("alpha")
    R.to_csv(HERE / "coherence_loss.csv", index=False)
    print(f"{'alpha':>7}{'true err':>10}{'clade coherence':>18}{'map spread':>12}{'spots used':>12}")
    print("-" * 60)
    for _, r in R.iterrows():
        print(f"{r.alpha:>7.2f}{r.err:>10.1f}{r.clade_coh:>18.4f}{r.spread:>12.0f}"
              f"{int(r.spots_used):>12}")
    r = spearmanr(R.clade_coh, R.err)
    print(f"\nclade coherence vs true error: rho = {r.statistic:+.3f}, p = {r.pvalue:.3f}")
    print(f"  coherence picks alpha = {R.alpha[R.clade_coh.idxmin()]:g}")
    print(f"  truth     picks alpha = {R.alpha[R.err.idxmin()]:g}")
