"""Self-consistent rogueness: recompute it from the CURRENT MAPPING, not from any input score.

  1. solve the map with the current per-cell trust
  2. for each cell, measure how far it landed from where its own clade landed
  3. down-weight the cells that disagree with their clade
  4. re-solve, and repeat

Nothing is taken on input except the tree and the expression -- no clade-call confidence, no
ground truth. The rogueness is a property of the map, so it can change as the map improves.

The obvious hazard is positive feedback: a down-weighted cell feels no tree pull, drifts further
from its clade, looks more rogue, and gets down-weighted again. Damping is applied to the trust
update, and both the trust distribution and the (unused-by-the-scheme) true error are tracked
per iteration so a runaway is visible rather than silent.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.trustfgw as tf

NITER, DAMP = 8, 0.5
SETTINGS = [(lam, sc) for lam in (0.6, 1.0) for sc in (0.0, 0.25, 0.5)]
_S = {}


def prep():
    d = np.load(HERE / "petracer_M3-4.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    n, m = M.shape
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float),
                cc=cc, n=n, clades=np.unique(cc))


def _init(): _S["d"] = prep()


def _run(job):
    lam, scale = job
    S = _S["d"]
    a = np.ones(S["n"])
    out = []
    for it in range(NITER):
        T, _ = tf.solve(S["M"], S["C1"], S["C2"], S["p"], S["q"], a, lam,
                        max_iter=200, tol=1e-9)
        pos = S["xy_st"][np.asarray(T).argmax(1)]
        err = float(np.median(np.linalg.norm(pos - S["xy_sc"], axis=1)))   # evaluation only
        # rogueness GIVEN THE MAP: how far the cell landed from where its clade landed
        r = np.zeros(S["n"])
        for c in S["clades"]:
            s = S["cc"] == c
            cen = np.median(pos[s], 0)
            dd = np.linalg.norm(pos[s] - cen, axis=1)
            r[s] = dd / max(np.median(dd), 1e-9)
        out.append(dict(lam=lam, scale=scale, it=it, err=err, mean_alpha=float(a.mean()),
                        frac_zero=float((a <= 1e-9).mean()), rogue_spread=float(r.std())))
        if scale > 0:
            z = (r - r.mean()) / (r.std() + 1e-9)
            a = (1 - DAMP) * a + DAMP * np.clip(1.0 - scale * z, 0.0, 1.0)
    return out


if __name__ == "__main__":
    rows = []
    with Pool(6, initializer=_init) as pool:
        for res in pool.imap_unordered(_run, SETTINGS):
            rows += res
    R = pd.DataFrame(rows); R.to_csv(HERE / "iterative_rogue.csv", index=False)
    for (lam, sc), g in R.groupby(["lam", "scale"]):
        g = g.sort_values("it")
        tag = "uniform (no reweighting)" if sc == 0 else f"scale {sc}"
        print(f"\nlam {lam}, {tag}")
        print("   iter " + " ".join(f"{i:>7}" for i in g.it))
        print("   err  " + " ".join(f"{v:>7.1f}" for v in g.err))
        print("   <a>  " + " ".join(f"{v:>7.2f}" for v in g.mean_alpha))
        print("   a=0  " + " ".join(f"{v:>7.1%}" for v in g.frac_zero))
