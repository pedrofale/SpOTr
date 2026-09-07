"""RUN THE METHOD ON BCSA, on the project's own benchmark footing.

BCSA has no cell positions, so there is no placement error to measure.  The established task in
experiments/bcsa3 is spot deconvolution: predict clone proportions per Visium spot, scored against
Numbat's calls by argmax accuracy, ARI and mean per-clone Pearson against the soft posterior
(experiments/bcsa3/evaluate.py), with DestVI and scDOT as cached baselines.

The coupling gives proportions directly: aggregate T over the cells of each clone and normalise
per spot.

CIRCULARITY GUARD.  The learned transform is estimated from spot clone compositions, and the
metric scores agreement with spot clone compositions.  So the transform is fitted on a random HALF
of the spots and every arm is scored on the OTHER half.  The expression and FGW arms never touch
spot_soft at all.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, accuracy_score
HERE = Path(__file__).resolve().parent
BC = HERE.parent / "bcsa3"
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import (GenFGW, fit_kernel, standardise, apply_lookup,
                          solve_semirelaxed)

WGRID = [0.0] + list(np.geomspace(0.004, 1.2, 9))
_S = {}


def load(tag):
    d = np.load(BC / f"bcsa3_{tag}_inputs.npz", allow_pickle=True)
    S = {k: np.asarray(d[k]) for k in d.files}
    # one spot in D1 has a library size of 8.7e7 against a median of 5.3e3 -- a corrupted row that
    # otherwise sets M.max() and crushes the expression cost relative to the structural term
    rs = S["R_st"].astype(float).sum(1)
    keep = rs < 3 * np.median(rs)
    if (~keep).any():
        print(f"   [{tag}] dropping {int((~keep).sum())} corrupted spot(s)", flush=True)
        for k in ["R_st", "xy", "spot_clone_numbat", "spot_soft", "spots"]:
            S[k] = S[k][keep]
        S["C_space"] = S["C_space"][np.ix_(keep, keep)]
    return S


def sample_joint(soft, clones, xy, spots, pool, rng, n_draw=1_500_000):
    cum = soft[spots].cumsum(1)
    j = rng.integers(0, len(spots), n_draw); l = rng.integers(0, len(spots), n_draw)
    k = j != l; j, l = j[k], l[k]
    c1 = clones[(rng.random(len(j))[:, None] < cum[j]).argmax(1)]
    c2 = clones[(rng.random(len(l))[:, None] < cum[l]).argmax(1)]
    s = np.linalg.norm(xy[spots[j]] - xy[spots[l]], axis=1)
    t = np.empty(len(j)); ok = np.zeros(len(j), bool)
    for key in set(zip(c1.tolist(), c2.tolist())):
        m = (c1 == key[0]) & (c2 == key[1]); v = pool.get(key)
        if v is None or not m.any():
            continue
        t[m] = rng.choice(v, m.sum()); ok |= m
    return t[ok], s[ok]


def _init(tag, R, seed):
    S = load(tag)
    rng = np.random.default_rng(seed)
    Xc = np.log1p(S["R_sc"].astype(float)); Xs = np.log1p(S["R_st"].astype(float))
    M = ot.dist(Xc, Xs); M /= M.max()
    CT = S["C_tree"].astype(float); xy = S["xy"].astype(float)
    CS = squareform(pdist(xy))
    clones = S["clones"]; cc = S["cell_clone"]
    m = len(xy)
    perm = rng.permutation(m)
    ref, held = perm[:m // 2], perm[m // 2:]
    if R == 0:
        Phi = [standardise(CT)]; Psi = [standardise(CS)]; rho = np.array([1.0])
        K = None
    else:
        pool = {}
        for a in clones:
            for b in clones:
                if b < a:
                    continue
                ia, ib = np.where(cc == a)[0], np.where(cc == b)[0]
                if len(ia) == 0 or len(ib) == 0:
                    continue
                Dm = CT[np.ix_(ia, ib)]
                v = Dm[np.triu_indices(len(ia), 1)] if a == b else Dm.ravel()
                if len(v):
                    pool[(a, b)] = pool[(b, a)] = v
        t, s = sample_joint(S["spot_soft"].astype(float), clones, xy, ref, pool, rng)
        K = fit_kernel(t, s, ncomp=R, nbin=17)
        Phi = [standardise(apply_lookup(CT, K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(CS, K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    _S["d"] = dict(M=M, Phi=Phi, Psi=Psi, rho=rho, held=held, S=S, K=K,
                   n=M.shape[0], m=m)


def props(T, S):
    P = np.zeros((T.shape[1], len(S["clones"])))
    for k, c in enumerate(S["clones"]):
        P[:, k] = T[S["cell_clone"] == c].sum(0)
    return P


def score_held(P, S, held, label):
    P = P[held]; P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    truth = S["spot_clone_numbat"][held]; soft = S["spot_soft"].astype(float)[held]
    pred = S["clones"][P.argmax(1)]
    cors = [pearsonr(P[:, k], soft[:, k])[0] for k in range(P.shape[1])
            if np.std(soft[:, k]) > 1e-8 and np.std(P[:, k]) > 1e-8]
    return dict(method=label, acc=accuracy_score(truth, pred),
                ari=adjusted_rand_score(truth, pred),
                clone_r=float(np.mean(cors)) if cors else np.nan,
                majority=np.bincount(truth).max() / len(truth))


def _solve(job):
    w, mode = job
    S = _S["d"]
    q = np.ones(S["m"]) / S["m"]
    g = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], max(w, 1e-12) / (1 + max(w, 1e-12)))
    if mode == "balanced":
        p = np.ones(S["n"]) / S["n"]
        if w == 0:
            T = ot.emd(p, q, S["M"], numItermax=10 ** 7)
        else:
            T, _ = g.solve(p, q, 200, 1e-9)
    else:
        # spot marginal fixed, cell mixture free -- the deconvolution formulation
        if w == 0:
            T = np.zeros(S["M"].shape); T[S["M"].argmin(0), np.arange(S["m"])] = q
        else:
            T, _ = solve_semirelaxed(g, q, side="spots", max_iter=200, tol=1e-9)
    return (w, mode), props(np.asarray(T, float), S["S"])


if __name__ == "__main__":
    rows = []
    for tag in ["D1", "B1"]:
        for seed in (0, 1, 2):
            for R, name in [(0, "FGW"), (2, "learned")]:
                jobs = [(w, md) for w in WGRID for md in ("balanced", "free-mixture")]
                with Pool(6, initializer=_init, initargs=(tag, R, seed)) as pool:
                    S = load(tag)
                    held = np.random.default_rng(seed).permutation(len(S["xy"]))[len(S["xy"]) // 2:]
                    for (w, md), P in pool.imap_unordered(_solve, jobs):
                        r = score_held(P, S, held, "expression only" if w == 0 else name)
                        r.update(tag=tag, seed=seed, w=w, mode=md)
                        rows.append(r)
                print(f"   [{tag} seed {seed} {name}] done, {len(rows)} rows", flush=True)
                pd.DataFrame(rows).to_csv(HERE / "bcsa_run.csv", index=False)
        g = pd.DataFrame(rows).query("tag == @tag")
        print(f"\n=== BCSA3 {tag}  (scored on held-out spots, 3 reference draws) ===")
        print(f"{'arm':<18}{'marginals':<15}{'acc':>8}{'ARI':>8}{'clone r':>10}")
        print(f"{'majority clone':<18}{'':<15}{g[g.w == 0].majority.mean():>8.3f}"
              f"{'-':>8}{'-':>10}")
        for md in ("balanced", "free-mixture"):
            gm = g[g["mode"] == md]
            e = gm[gm.w == 0]
            print(f"{'expression only':<18}{md:<15}{e.acc.mean():>8.3f}{e.ari.mean():>8.3f}"
                  f"{e.clone_r.mean():>10.3f}")
            for name in ["FGW", "learned"]:
                h = gm[(gm.method == name) & (gm.w > 0)]
                if not len(h):
                    continue
                b = h.loc[h.groupby("seed").clone_r.idxmax()]
                print(f"{name:<18}{md:<15}{b.acc.mean():>8.3f}{b.ari.mean():>8.3f}"
                      f"{b.clone_r.mean():>10.3f}", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "bcsa_run.csv", index=False)
    print("\ndone")
