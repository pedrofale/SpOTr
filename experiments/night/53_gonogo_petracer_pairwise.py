"""GO/NO-GO (2026-10-12): does a decomposed pairwise (GW-type) term between UNBARCODED cells recover
lineage better than a spatial neighbour vote on PEtracer M3?

Every regime-B arm so far scored query cells one at a time (linear terms).  The formulation's
quadratic term -- two query cells, both tree endpoints latent -- was never run.  This is it:

    min_T  <U, T> + lam * Q(T),   T >= 0, rows sum to 1, clade marginal free (semi-relaxed)
    Q(T) = (1/n) sum_{i != i'} sum_{k,k'} T_ik T_i'k' * l_b(k,k')(d_ii')
    l_b(d) = (d - mu_b)^2 / (2 sig_b^2) + log sig_b

U is the per-clade mixture (48_, the best tree-free arm, -log p(x_i | k) with LOO-ML bandwidths).
The pair loss is a square loss with a per-class target and weight -- a DECOMPOSED GW term -- whose
classes b(k,k') are either
    POTTS  (tree-free): same clade / different clade
    TREE   (divergence time): same clade / different clade with recent, older, or founding LCA
mu_b and sig_b are estimated from reference cell pairs only.  lam is the one fitted weight.
Solver: Frank-Wolfe with a row-wise argmin linear oracle and exact line search, from argmin U.

Metrics, chosen in advance: macro clade accuracy, top-1, and LINEAGE ERROR = divergence age (days)
of the LCA of the true and predicted clade, 0 when correct.  The vote gets a tree-aware decision
too (argmin of expected LCA age under its neighbour distribution), so the tree arms are not
compared against a baseline that ignores the metric.

Pre-declared rule: GO only if, at some reference density, the TREE arm with lam chosen
leave-one-section-out beats BOTH the tuned kNN vote and the POTTS arm in a majority of splits.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
SECS = ["M3-4", "M3-1", "M3-2"]
DENS = [0.45, 0.20, 0.05]
SEEDS = [0, 1, 2]
LAMS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]
KCAND = [1, 3, 5, 9, 15, 25]
HGRID = np.array([10., 15., 22., 32., 46., 68., 100., 150., 220., 320., 470., 700.])
NIT = 40


def mix(D2, s):
    v = 2.0 * s ** 2
    return -(logsumexp(-D2 / v - np.log(np.pi * v), axis=1) - np.log(D2.shape[1]))


def loo_h(Dr2, mem):
    if len(mem) < 3:
        return np.nan
    S = Dr2[np.ix_(mem, mem)].copy(); np.fill_diagonal(S, np.inf)
    ll = [np.mean(logsumexp(-S / (2 * h * h) - np.log(2 * np.pi * h * h), axis=1)) for h in HGRID]
    return float(HGRID[int(np.argmax(ll))])


def solve(U, D, D2, A2, A1, A0, lam):
    n, K = U.shape
    T = np.zeros_like(U); T[np.arange(n), U.argmin(1)] = 1.0
    if lam == 0:
        return T

    def qprod(X):          # returns D2@X@A2 - D@X@A1 + (colsum - X)@A0, the un-scaled bilinear map
        return D2 @ X @ A2 - D @ X @ A1 + (X.sum(0)[None, :] - X) @ A0

    for _ in range(NIT):
        G = U + lam * (2.0 / n) * qprod(T)
        S = np.zeros_like(T); S[np.arange(n), G.argmin(1)] = 1.0
        Dl = S - T
        slope = float((G * Dl).sum())
        if slope > -1e-9:
            break
        curv = lam * float((qprod(Dl) * Dl).sum()) / n
        g = 1.0 if curv <= 0 else min(1.0, -slope / (2 * curv))
        T = T + g * Dl
    return T


def run(job):
    sec, f, seed = job
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    XY = np.asarray(z["xy_sc"], float); CT = np.asarray(z["C_tree"], float) / 2.0   # LCA age, days
    cl = np.asarray(z["clades"]); ix = {c: i for i, c in enumerate(cl)}
    Y = np.array([ix[c] for c in np.asarray(z["cell_clade"])]); K = len(cl); n = len(XY)
    perm = np.random.default_rng(seed).permutation(n)
    nr = int(round(f * n)); ref, qry = np.sort(perm[:nr]), np.sort(perm[nr:])
    xr, xq, yr, yq = XY[ref], XY[qry], Y[ref], Y[qry]

    # clade-pair LCA ages (ultrametric tree, clades are subtrees: one age per clade pair)
    rep = np.array([np.where(Y == k)[0][0] for k in range(K)])
    AGE = CT[np.ix_(rep, rep)].copy(); np.fill_diagonal(AGE, 0.0)
    amax = AGE.max()

    ok = np.array([(yr == k).sum() >= 2 for k in range(K)])
    Dq2 = cdist(xq, xr, "sqeuclidean")
    Dr2 = squareform(pdist(xr) ** 2)
    hk = np.array([loo_h(Dr2, np.where(yr == k)[0]) for k in range(K)])
    hg = loo_h(Dr2, np.arange(len(ref)))
    U = np.full((len(qry), K), 1e3)
    for k in np.where(ok)[0]:
        mk = np.where(yr == k)[0]
        U[:, k] = mix(Dq2[:, mk], np.full(len(mk), hk[k] if np.isfinite(hk[k]) else hg))

    # pair classes over clade pairs
    res_ages = AGE[(AGE > 0) & (AGE < amax - 1e-6)]
    cut = np.median(res_ages) if len(res_ages) else amax
    cls_potts = np.where(np.eye(K, dtype=bool), 0, 1)
    cls_tree = np.where(np.eye(K, dtype=bool), 0,
                        np.where(AGE >= amax - 1e-6, 3, np.where(AGE <= cut, 1, 2)))

    # per-class separation statistics from REFERENCE pairs only
    iu = np.triu_indices(len(ref), 1)
    dref = np.sqrt(Dr2[iu]); kr1, kr2 = yr[iu[0]], yr[iu[1]]

    def class_mats(cls):
        nb = cls.max() + 1
        bc = cls[kr1, kr2]
        mu = np.array([dref[bc == b].mean() if (bc == b).sum() > 20 else dref.mean() for b in range(nb)])
        sg = np.array([dref[bc == b].std() if (bc == b).sum() > 20 else dref.std() for b in range(nb)])
        A2 = 1.0 / (2 * sg[cls] ** 2); A1 = mu[cls] / sg[cls] ** 2
        A0 = mu[cls] ** 2 / (2 * sg[cls] ** 2) + np.log(sg[cls])
        return A2, A1, A0, mu, sg

    D = squareform(pdist(xq)); D2 = D ** 2
    rows = []

    def record(arm, lam, pred):
        pres = [k for k in range(K) if (yq == k).sum() > 0]
        rows.append(dict(sec=sec, dens=f, seed=seed, arm=arm, lam=lam,
                         top1=float((pred == yq).mean()),
                         macro=float(np.mean([(pred[yq == k] == k).mean() for k in pres])),
                         lin_err=float(AGE[pred, yq].mean())))

    # tree-free baselines
    nn = np.argsort(Dq2, axis=1)[:, :max(KCAND)]
    Srr = Dr2.copy(); np.fill_diagonal(Srr, np.inf); nnr = np.argsort(Srr, axis=1)[:, :max(KCAND)]

    def votes(lab, kk):
        v = np.zeros((len(lab), K))
        for c in range(kk):
            np.add.at(v, (np.arange(len(lab)), lab[:, c]), 1.0 + 1e-6 * (kk - c))
        return v

    def macro_of(pred, y):
        return np.mean([(pred[y == k] == k).mean() for k in np.unique(y)])
    kk = max([c for c in KCAND if c < len(ref)],
             key=lambda c: macro_of(votes(yr[nnr], c).argmax(1), yr))
    V = votes(yr[nn], kk)
    record("kNN vote (k LOO)", np.nan, V.argmax(1))
    P = V / V.sum(1, keepdims=True)
    record("kNN vote, tree-aware decision", np.nan, (P @ AGE.T).argmin(1))  # argmin_k E[age(k,c)]
    record("15-NN vote", np.nan, votes(yr[nn], min(15, len(ref) - 1)).argmax(1))

    # unary alone, then the two pairwise arms
    for name, cls in [("POTTS pairwise", cls_potts), ("TREE pairwise (divergence time)", cls_tree)]:
        A2, A1, A0, mu, sg = class_mats(cls)
        for lam in LAMS:
            T = solve(U, D, D2, A2, A1, A0, lam)
            record("mixture (unary only)" if lam == 0 else name, lam, T.argmax(1))
    return rows


if __name__ == "__main__":
    jobs = [(s, f, sd) for s in SECS for f in DENS for sd in SEEDS]
    out = []
    with Pool(14) as pool:
        for r in pool.imap_unordered(run, jobs):
            out += r
            print(f"{r[0]['sec']} dens={r[0]['dens']} seed={r[0]['seed']} done", flush=True)
    df = pd.DataFrame(out).drop_duplicates(subset=["sec", "dens", "seed", "arm", "lam"])
    df.to_csv(HERE / "gonogo_petracer_pairwise.csv", index=False)
    print("written gonogo_petracer_pairwise.csv")
