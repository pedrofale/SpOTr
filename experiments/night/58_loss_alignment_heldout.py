"""LADDER STEP 1 (2026-10-12): does the niche loss still peak at the accuracy optimum when cell states are
defined on the scRNA-seq arm ONLY and their program genes are REMOVED from the mapping cost?

57_ used PEtracer's own Hotspot modules (built with spatial information) and all 175 genes in the cost.
Here, as in a genuinely unmatched dataset:
  1. programs: NMF (K=4) on the dissociated cells' log-expression; per program the top genes by
     loading x specificity; their union G_P is the PROGRAM GENE SET
  2. states: NMF refitted on G_P only; dissociated cells -> argmax usage; spot-side cells scored with the
     SAME programs (transform), so both sides use one state definition learned from the scRNA arm
  3. niche model p(state | niche) trained on the spot side only (as 57_)
  4. mapping cost on the REMAINING genes (held-out) -- and, as a control, on all genes
Exact FGW path (55_'s solver), seeds 0 and 1 per section = 6 splits.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import importlib.util
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, treedata as td, ot
from scipy.stats import spearmanr
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
K_PROG, TOP = 4, 8
_C = {}


def base():
    if "t" not in _C:
        t = td.read_h5td("/home/pedro/.treedata/datasets/koblan25_tumor.h5td")
        o = t.obs.copy()
        nf = np.c_[np.asarray(t.obsm["subtype_density"]), o.tumor_boundary_dist.values,
                   o.lung_boundary_dist.fillna(o.lung_boundary_dist.max()).values]
        X = t.X.tocsr().astype(float)
        lib = np.asarray(X.sum(1)).ravel()
        _C["t"] = (o, pd.DataFrame(nf, index=o.index), X, lib, {c: i for i, c in enumerate(o.index)})
    return _C["t"]


def logexpr(cells):
    o, NF, X, lib, ix = base()
    r = np.array([ix[c] for c in cells])
    return np.log1p(X[r].multiply(1e4 / np.maximum(lib[r], 1)[:, None]).toarray())


def prep(sec, seed):
    key = (sec, seed)
    if key in _C:
        return _C[key]
    o, NF, X, lib, ix = base()
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    S = m55.load(sec)
    allc = np.asarray(z["cell_names"]); n = len(allc)
    use = np.random.default_rng(seed).permutation(n)[n // 2:]
    scells = np.asarray(z["spot_cell_names"]); sof = np.asarray(z["spot_of_cell"]); m = S["M"].shape[1]
    # 1. programs on the dissociated arm only
    Ed = logexpr(allc)
    nmf = NMF(K_PROG, init="nndsvda", max_iter=1000, random_state=0).fit(Ed)
    H = nmf.components_; spec_ = H / np.maximum(H.sum(0, keepdims=True), 1e-12)
    GP = sorted({g for k in range(K_PROG) for g in np.argsort(-(H[k] * spec_[k]))[:TOP]})
    rest = np.setdiff1d(np.arange(Ed.shape[1]), GP)
    # 2. states from programs refitted on G_P; same programs scored on the spot side
    nmf2 = NMF(K_PROG, init="nndsvda", max_iter=1000, random_state=0).fit(Ed[:, GP])
    st_d = pd.Series(nmf2.transform(Ed[:, GP]).argmax(1), index=allc)
    st_s = nmf2.transform(logexpr(scells)[:, GP]).argmax(1)
    # 3. niche model, spot side only
    sc_ = StandardScaler().fit(NF.loc[scells].values)
    pca = PCA(10, random_state=0).fit(sc_.transform(NF.loc[scells].values))
    f = lambda A: pca.transform(sc_.transform(A))
    clf = LogisticRegression(max_iter=3000).fit(f(NF.loc[scells].values), st_s)
    spotNF = np.vstack([NF.loc[scells[sof == j]].values.mean(0) for j in range(m)])
    P_spot = clf.predict_proba(f(spotNF)); cls = list(clf.classes_)
    cells = allc[use]
    ic = np.array([cls.index(v) if v in cls else -1 for v in st_d.loc[cells].values])
    P_own = clf.predict_proba(f(NF.loc[cells].values))
    spot_tbd = np.array([o.loc[scells[sof == j], "tumor_boundary_dist"].mean() for j in range(m)])
    # 4. costs: held-out genes and all genes (fair aggregation from the FAIR file)
    Xsc = np.asarray(z["X_sc"], float)[use]; Xst = np.asarray(z["X_st"], float)
    Mh = ot.dist(Xsc[:, rest], Xst[:, rest]); Mh /= Mh.max()
    Ma = S["M"][use]
    _C[key] = dict(S=S, use=use, cells=cells, P_spot=P_spot, P_own=P_own, ic=ic,
                   fit=o.loc[cells, "fitness"].values, tbd_own=o.loc[cells, "tumor_boundary_dist"].values,
                   spot_tbd=spot_tbd, M={"held-out genes": Mh, "all genes": Ma}, nGP=len(GP))
    return _C[key]


def niche_ll(P, ic):
    ok = ic >= 0
    return float(np.mean(np.log(P[np.arange(len(ic))[ok], ic[ok]] + 1e-12)))


def job(args):
    sec, seed, variant, w = args
    D = prep(sec, seed); S = D["S"]; use = D["use"]
    if w is None:
        return [dict(sec=sec, seed=seed, variant="truth", w=np.nan, err=0.0, nGP=D["nGP"],
                     niche=niche_ll(D["P_own"], D["ic"]), growth=spearmanr(D["fit"], D["tbd_own"])[0])]
    AGE = S["AGE"][np.ix_(use, use)]; M = D["M"][variant]; Dm = S["D"]; q = S["q"]
    p = np.ones(len(use)) / len(use)
    Wm = np.ones_like(AGE); np.fill_diagonal(Wm, 0.0); WPhi = Wm * 2.0 * AGE
    T0 = np.outer(p, q); g0 = 2.0 * (Wm @ T0 @ Dm ** 2 - 2.0 * WPhi @ T0 @ Dm)
    sc = m55.dc_std(M) / max(m55.dc_std(g0), 1e-300)
    T = m55.solve(M, Wm * sc, WPhi * sc, Dm, Dm ** 2, p, q, w / (1 + w))
    am = T.argmax(1)
    err = float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"][use], axis=1)))
    return [dict(sec=sec, seed=seed, variant=variant, w=w, err=err, nGP=D["nGP"],
                 niche=niche_ll(D["P_spot"][am], D["ic"]), growth=spearmanr(D["fit"], D["spot_tbd"][am])[0])]


if __name__ == "__main__":
    jobs = [(s, sd, None, None) for s in m55.SECS for sd in (0, 1)]
    jobs += [(s, sd, v, float(w)) for s in m55.SECS for sd in (0, 1)
             for v in ("held-out genes", "all genes") for w in m55.WGRID]
    out = []
    with Pool(24) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            out += r
    df = pd.DataFrame(out).sort_values(["sec", "seed", "variant", "w"])
    df.to_csv(HERE / "loss_alignment_heldout.csv", index=False)
    print(f"program genes removed from the cost: {sorted(df.nGP.unique())} of 175\n")
    rows = []
    for (sec, seed, var), g in df[df.variant != "truth"].groupby(["sec", "seed", "variant"]):
        bi, ni, gi = g.err.idxmin(), g.niche.idxmax(), g.growth.idxmin()
        rows.append(dict(split=f"{sec} s{seed}", variant=var, expr_only=g[g.w == 0].err.iloc[0],
                         w_best=g.loc[bi, "w"], err_best=g.loc[bi, "err"],
                         w_niche=g.loc[ni, "w"], err_niche=g.loc[ni, "err"],
                         w_growth=g.loc[gi, "w"], err_growth=g.loc[gi, "err"],
                         rho_err_niche=spearmanr(g.err, -g.niche)[0], niche_at_w0_is_max=bool(g.loc[ni, "w"] == 0)))
    r = pd.DataFrame(rows)
    print(r.round(3).to_string(index=False))
    for var, g in r.groupby("variant"):
        print(f"\n{var}: mean error expr-only {g.expr_only.mean():.0f} | at accuracy optimum {g.err_best.mean():.0f} | "
              f"at niche pick {g.err_niche.mean():.0f} | at growth pick {g.err_growth.mean():.0f} um;  "
              f"niche picks w=0 in {g.niche_at_w0_is_max.sum()}/{len(g)};  mean rho(err, -niche) {g.rho_err_niche.mean():+.2f}")
