"""ORACLE question: is there a weighted decomposition of the tree-space cost that beats
global-alpha FGW?  Ground truth is used to FIT the weights -- learnability is a separate
question and is deliberately not asked here.

Every decomposition whose classes are unions of clade-pairs is a K x K clade-level weight
matrix. Tested, in increasing generality:

  1 bin    global alpha (ordinary FGW)                       -- the baseline
  2,3,5,8  weights binned by clade-pair TREE DISTANCE        -- reweight the tree by resolution
  perclade one weight per clade (a row/column scheme)        -- the paper's alpha_tau

Weights are fitted by coordinate descent directly on median placement error. Both the in-sample
ceiling and a held-out score (fit on half the spots, score on the other half) are reported, so
extra capacity cannot flatter the answer.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.multimask as mm

GRID = np.array([0.0, 0.15, 0.3, 0.6, 1.0, 1.7, 3.0])
SEC = "M3-4"
_S = {}


def prep():
    d = np.load(HERE / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M = M / M.max()
    CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
    cc = np.asarray(d["cell_clade"]).astype(str); _, clade = np.unique(cc, return_inverse=True)
    n, m = M.shape
    K = clade.max() + 1
    Dkk = np.array([[CT[np.ix_(clade == a, clade == b)].mean() for b in range(K)]
                    for a in range(K)])
    return dict(M=M, C1=CT / CT.mean(), C2=CS / CS.mean(), clade=clade, Dkk=Dkk, K=K,
                p=np.ones(n) / n, q=np.ones(m) / m, m=m,
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=np.asarray(d["xy"], float))


def _init():
    S = prep()
    S["bf"] = mm.BlockFGW(S["M"], S["C1"], S["C2"], S["clade"], np.ones((S["K"], S["K"])))
    _S["d"] = S


def bin_map(Dkk, nbin):
    """assign each clade PAIR to a tree-distance bin (bin 0 = the diagonal, same clade)"""
    K = len(Dkk)
    lab = np.zeros((K, K), int)
    off = ~np.eye(K, dtype=bool)
    v = Dkk[off]
    if nbin > 1:
        edges = np.quantile(v, np.linspace(0, 1, nbin)[1:-1]) if nbin > 2 else []
        lab[off] = np.digitize(v, edges) + 1
    else:
        lab[off] = 0
    return lab


def _solve(job):
    scheme, w, tag = job
    S = _S["d"]
    K = S["K"]
    if scheme == "bins":
        lab = bin_map(S["Dkk"], len(w))
        Wkk = np.asarray(w)[lab]
    else:                                     # per-clade: W[a,b] = (w_a + w_b)/2
        w = np.asarray(w)
        Wkk = 0.5 * (w[:, None] + w[None, :])
    T, _ = S["bf"].set_weight_matrix(Wkk).solve(S["p"], S["q"], max_iter=200, tol=1e-9)
    return tag, np.asarray(T).argmax(1).astype(np.int32)


def coord_descent(pool, scheme, nw, score, passes=3, init=1.0):
    w = [init] * nw
    best = None
    for it in range(passes):
        for k in range(nw):
            jobs = []
            for g in GRID:
                ww = list(w); ww[k] = float(g)
                jobs.append((scheme, ww, (it, k, float(g))))
            res = {}
            for tag, am in pool.imap_unordered(_solve, jobs, 1):
                res[tag[2]] = am
            errs = {g: score(res[g]) for g in res}
            gb = min(errs, key=lambda x: errs[x])
            w[k] = gb; best = (errs[gb], list(w), res[gb])
    return best


if __name__ == "__main__":
    S = prep()
    rng = np.random.default_rng(0)
    h = rng.random(S["m"]) < 0.5
    fit, hel = np.where(h)[0], np.where(~h)[0]

    def err(am, cols=None):
        if cols is None:
            return float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"], axis=1)))
        k = np.isin(am, cols)
        return float(np.median(np.linalg.norm(S["xy_st"][am[k]] - S["xy_sc"][k], axis=1)))

    out = []
    with Pool(8, initializer=_init) as pool:
        for name, scheme, nw in [("global alpha (FGW)", "bins", 1),
                                 ("2 tree-distance bins", "bins", 2),
                                 ("3 tree-distance bins", "bins", 3),
                                 ("5 tree-distance bins", "bins", 5),
                                 ("8 tree-distance bins", "bins", 8),
                                 ("per-clade alpha (18)", "clade", S["K"])]:
            t0 = time.time()
            e_all, w_all, am_all = coord_descent(pool, scheme, nw, lambda a: err(a))
            e_fit, w_fit, am_fit = coord_descent(pool, scheme, nw, lambda a: err(a, fit))
            out.append(dict(scheme=name, n_par=nw, insample=e_all,
                            heldout=err(am_fit, hel), secs=time.time() - t0,
                            weights=np.round(w_all, 2).tolist()))
            r = out[-1]
            print(f"{name:<24} params {nw:>3}   in-sample {r['insample']:7.1f}   "
                  f"held-out {r['heldout']:7.1f}   ({r['secs']:.0f}s)", flush=True)
            print(f"{'':<24} fitted weights: {r['weights']}", flush=True)
    pd.DataFrame(out).to_csv(HERE / "decomposition_ceiling.csv", index=False)
