"""The alpha sweep, current expression cost vs the fair one, everything else identical."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
GRID = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
_S = {}


def prep(sec, fair):
    f = f"petracerFAIR_{sec}.npz" if fair else f"petracer_{sec}.npz"
    d = np.load(HERE / f, allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    n, m = M.shape
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    return dict(M=M, C1=CT/CT.mean(), C2=CS/CS.mean(), p=np.ones(n)/n, q=np.ones(m)/m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float),
                cidx=np.array([clades.index(c) for c in cc]),
                COMP=np.load(HERE / f"comp_{sec}.npy").astype(float))


def _init(sec, fair):
    _S["d"] = prep(sec, fair)


def _solve(a):
    S = _S["d"]
    T = np.asarray(ot.gromov.fused_gromov_wasserstein(
        S["M"], S["C1"], S["C2"], S["p"], S["q"], loss_fun="square_loss", alpha=a,
        max_iter=200, tol_rel=1e-9, tol_abs=1e-9))
    am = T.argmax(1)
    return (a, float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1))),
            float(S["COMP"][am, S["cidx"]].mean()))


if __name__ == "__main__":
    rows = []
    for sec in ["M3-4", "M3-1", "M3-2"]:
        for fair in (False, True):
            with Pool(8, initializer=_init, initargs=(sec, fair)) as pool:
                for a, err, soft in pool.imap_unordered(_solve, GRID):
                    rows.append(dict(section=sec, fair=fair, alpha=a, err=err, soft=soft))
            print(f"  done {sec} fair={fair}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(HERE / "fair_sweep.csv", index=False)
    for sec, g0 in R.groupby("section"):
        print(f"\n=== {sec} ===")
        print(f"{'alpha':>7}{'current err':>14}{'FAIR err':>11}    "
              f"{'current soft':>13}{'FAIR soft':>11}")
        print("-" * 60)
        cur = g0[~g0.fair].set_index("alpha").sort_index()
        fai = g0[g0.fair].set_index("alpha").sort_index()
        for a in GRID:
            print(f"{a:>7.2f}{cur.err[a]:>14.1f}{fai.err[a]:>11.1f}    "
                  f"{cur.soft[a]:>13.4f}{fai.soft[a]:>11.4f}")
        for tag, t in [("current", cur), ("FAIR", fai)]:
            b = t.err.idxmin()
            print(f"  {tag:<8} no tree {t.err[0.0]:7.1f} um -> best {t.err[b]:7.1f} um at "
                  f"alpha={b:g}   gain {t.err[0.0]-t.err[b]:+7.1f} um "
                  f"({100*(t.err[0.0]-t.err[b])/t.err[0.0]:4.0f}%)")
