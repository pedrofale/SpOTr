"""DECOMPOSITION CHECK (2026-10-12): does a decomposed tree term, weighted by divergence-time class,
beat plain FGW in unmatched mapping -- and can its weights come from a growth law instead of ground
truth?

Structural term, one family for every arm (exact Frank-Wolfe, EMD linear step, exact line search):

    S(T) = sum_{i,k,j,l} W_ik (Phi_ik - d_jl)^2 T_ij T_kl          d in um, spots
    F(T) = (1 - a) <M, T> + a S(T) / scale,  a = w / (1 + w)

Phi_ik is the separation the tree EXPECTS for cells i, k and W_ik how much to trust it.

  FGW                 W = 1, Phi = patristic distance. By Fact 0 this is exactly square-loss FGW
                      (with W constant only <C_tree, T D T'> survives the fixed marginals).
  decomposed, ground  4 divergence-time classes (recent <20 d, 20-30 d, 30-35 d, founding = 35 d);
    truth             per class Phi_b = mean separation, W_b = 1/variance, measured on REFERENCE
                      pairs (disjoint half of the cells, true positions).  Interpretable upper bound.
  decomposed, law     the same 4 classes, Phi_b and W_b from the growth law
                      r(a)^2 = L^2 + c^2 n(a) / (2 pi rho)
                      n(a): clone size from the TREE of the cells being mapped;  rho: cells per um^2
                      from the SPOT side;  (L, c): fitted on the OTHER TWO sections only.
                      Uses no position of any cell in the target section.
  law, continuous     per-pair Rayleigh moments of the law instead of class averages.

Protocol as 36_headtohead.py: disjoint halves, observed spot marginal, median placement error of
argmax.  3 seeds x 2 halves x 3 sections = 18 splits (a subset of the 30).  Tree weight w chosen
LEAVE-ONE-SECTION-OUT, as in exp-h2h-reanalysis.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, ot
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
SECS = ["M3-4", "M3-1", "M3-2"]
WGRID = np.r_[0.0, np.geomspace(0.03, 30.0, 12)]
EDGES = [0.0, 20.0, 30.0, 35.0 - 1e-6, 35.0 + 1e-6]
ONLY = None
CLASSES = ["recent <20 d", "20-30 d", "30-35 d", "founding"]


def load(sec):
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy_st = np.asarray(d["xy"], float)
    M = ot.dist(np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)); M /= M.max()
    nc = np.asarray(d["spot_n_cells"], float)
    return dict(M=M, AGE=np.asarray(d["C_tree"], float) / 2.0, D=squareform(pdist(xy_st)),
                xy_sc=np.asarray(d["xy_sc"], float), xy_st=xy_st, q=nc / nc.sum(),
                area=ConvexHull(xy_st).volume)


def clone_size(AGE):
    """n_ik = number of cells in the clone founded at the LCA of i and k (incl. both)."""
    srt = np.sort(AGE, axis=1)
    return np.vstack([np.searchsorted(srt[i], AGE[i], side="right") for i in range(len(AGE))])


def measured_r(age, sep, nbin=22, min_pairs=200):
    cap = age.max(); below = age[age < cap - 1e-9]
    edges = np.r_[np.linspace(below.min(), below.max(), nbin), cap - 1e-9, cap + 1e-9]
    out = []
    for i in range(len(edges) - 1):
        m = (age >= edges[i]) & (age < edges[i + 1])
        if m.sum() >= min_pairs:
            out.append((age[m].mean(), np.sqrt((sep[m] ** 2).mean() / 2.0)))
    return np.array(out)


def law_points(sec):
    """(measured r per age bin, packing r per age bin) on a section's full ground truth."""
    S = load(sec); AGE = S["AGE"]; n = len(AGE); rho = n / S["area"]
    iu = np.triu_indices(n, 1)
    Mr = measured_r(AGE[iu], pdist(S["xy_sc"]))
    NC = clone_size(AGE)
    pack = []
    for a, _ in Mr:
        nc = (AGE <= a + 1e-9).sum(1)
        pack.append(np.sqrt(np.average(nc / (2 * np.pi * rho), weights=nc)))
    return Mr[:, 1], np.array(pack)


def fit_law(points):
    """(L, c) minimising squared log error of sqrt(L^2 + c^2 pack^2) over all bins given."""
    lr = np.concatenate([np.log(m) for m, _ in points]); pk = np.concatenate([p for _, p in points])
    best = (np.inf, 0, 1)
    for L in np.linspace(0, 600, 121):
        for c in np.linspace(0.2, 3.0, 141):
            e = np.sum((lr - 0.5 * np.log(L * L + c * c * pk ** 2 + 1e-9)) ** 2)
            if e < best[0]:
                best = (e, L, c)
    return best[1], best[2]


def classes(AGE):
    return np.digitize(AGE, EDGES[1:-1])          # 0..3


def solve(M, Wm, WPhi, D, D2, p, q, a, iters=300, tol=1e-9):
    T = np.outer(p, q)
    for _ in range(iters):
        G = (1 - a) * M + a * 2.0 * (Wm @ T @ D2 - 2.0 * WPhi @ T @ D)
        S = ot.emd(p, q, G, numItermax=10 ** 7)
        Dl = S - T
        slope = float((G * Dl).sum())
        if slope > -tol:
            break
        quad = a * (float(((Wm @ Dl @ D2) * Dl).sum()) - 2.0 * float(((WPhi @ Dl @ D) * Dl).sum()))
        g = 1.0 if quad <= 0 else min(1.0, -slope / (2.0 * quad))
        T = T + g * Dl
    return T


def dc_std(X):
    return (X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()).std()


def run(job):
    sec, seed, half, law = job
    S = load(sec)
    n = len(S["AGE"]); h = np.random.default_rng(seed).permutation(n)
    ref = h[:n // 2] if half == 0 else h[n // 2:]
    use = h[n // 2:] if half == 0 else h[:n // 2]
    AGE = S["AGE"][np.ix_(use, use)]; M = S["M"][use]; D = S["D"]; D2 = D ** 2; q = S["q"]
    p = np.ones(len(use)) / len(use)
    cls = classes(AGE)

    # ---- ground-truth class moments from REFERENCE pairs
    iu = np.triu_indices(len(ref), 1)
    ra = S["AGE"][np.ix_(ref, ref)][iu]; rs = pdist(S["xy_sc"][ref]); rc = classes(ra)
    gt_mu = np.array([rs[rc == b].mean() for b in range(4)])
    gt_w = np.array([1.0 / rs[rc == b].var() for b in range(4)])

    # ---- growth law: n(a) from the mapped cells' tree, rho from the spot side, (L, c) transferred
    L, c = law
    rho = len(use) / S["area"]
    R2 = L * L + c * c * clone_size(AGE) / (2 * np.pi * rho)       # r_law^2 per pair
    mean_pair = np.sqrt(np.pi / 2 * R2); var_pair = (4 - np.pi) / 2 * R2
    law_mu = np.array([mean_pair[cls == b].mean() for b in range(4)])
    law_var = np.array([(2 * R2[cls == b]).mean() - law_mu[b] ** 2 for b in range(4)])

    arms = {
        "FGW": (np.ones_like(AGE), 2.0 * AGE),
        "decomposed, ground truth": (gt_w[cls], gt_mu[cls]),
        "decomposed, law": ((1.0 / law_var)[cls], law_mu[cls]),
        "law, continuous": (1.0 / var_pair, mean_pair),
    }
    rows = []
    for name, (Wm, Phi) in arms.items():
        if ONLY and name not in ONLY:
            continue
        np.fill_diagonal(Wm, 0.0)
        WPhi = Wm * Phi
        T0 = np.outer(p, q)
        g0 = 2.0 * (Wm @ T0 @ D2 - 2.0 * WPhi @ T0 @ D)
        sc = dc_std(M) / max(dc_std(g0), 1e-300)
        for w in WGRID:
            if w == 0 and name != "FGW":
                continue
            a = w / (1 + w)
            T = solve(M, Wm * sc, WPhi * sc, D, D2, p, q, a)
            am = T.argmax(1)
            err = float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"][use], axis=1)))
            rows.append(dict(sec=sec, seed=seed, half=half,
                             arm="expression only" if w == 0 else name, w=w, err=err))
    meta = dict(sec=sec, seed=seed, half=half, L=L, c=c,
                **{f"gt_mu_{b}": gt_mu[b] for b in range(4)},
                **{f"gt_sd_{b}": gt_w[b] ** -0.5 for b in range(4)},
                **{f"law_mu_{b}": law_mu[b] for b in range(4)},
                **{f"law_sd_{b}": np.sqrt(law_var[b]) for b in range(4)},
                **{f"frac_{b}": float((cls == b).mean()) for b in range(4)})
    return rows, meta


if __name__ == "__main__":
    pts = {s: law_points(s) for s in SECS}
    laws = {s: fit_law([pts[o] for o in SECS if o != s]) for s in SECS}
    for s, (L, c) in laws.items():
        print(f"law for {s}, fitted on the other two sections: L = {L:.0f} um, c = {c:.2f}", flush=True)
    jobs = [(s, sd, hf, laws[s]) for s in SECS for sd in (0, 1, 2) for hf in (0, 1)]
    rows, metas = [], []
    with Pool(18) as pool:
        for r, m in pool.imap_unordered(run, jobs):
            rows += r; metas.append(m)
            print(f"{m['sec']} seed {m['seed']} half {m['half']} done", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "decomposition_check.csv", index=False)
    pd.DataFrame(metas).to_csv(HERE / "decomposition_check_classes.csv", index=False)
    print("written")
