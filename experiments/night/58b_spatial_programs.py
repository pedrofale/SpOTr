"""LADDER STEP 1, variant (2026-10-12): niche-responsive programs defined on the SPATIAL ARM ONLY.

58_ defined states by NMF on the dissociated arm and the niche loss lost precision (+52 to +68 um at its
pick).  PEtracer's own hotspot modules (57_, +0 to +12 um) were found by spatial autocorrelation over the
whole section -- including the cells we pretend are dissociated, a leak.  The honest equivalent: find
spatially autocorrelated genes and modules on the spot-side cells only, then score the dissociated cells
with the same gene sets.  Everything else as 58_ (spot-side niche model, program genes removed from the
cost, exact FGW path, 6 splits).

Programs: Moran's I per gene on a 15-NN spatial graph of spot-side cells; top 40 genes; modules by
average-linkage clustering of the correlation of SPATIALLY SMOOTHED expression (Hotspot's idea); K = 4.
State of a cell = argmax over modules of the mean z-score (z within each arm) of the module's genes.
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m58", HERE / "58_loss_alignment_heldout.py")
m58 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m58)
K_PROG, NTOP, KNN = 4, 40, 15


def programs(E, xy):
    nn = cKDTree(xy).query(xy, KNN + 1)[1][:, 1:]
    Z = (E - E.mean(0)) / np.maximum(E.std(0), 1e-9)
    moran = (Z * Z[nn].mean(1)).mean(0)                    # Moran's I with row-normalised kNN weights
    top = np.argsort(-moran)[:NTOP]
    Sm = Z[nn][:, :, top].mean(1)                          # spatially smoothed expression
    C = np.corrcoef(Sm.T)
    lab = fcluster(linkage(1 - C[np.triu_indices(len(top), 1)], "average"), K_PROG, "maxclust")
    return [top[lab == k] for k in np.unique(lab)], top


def states(E, mods):
    Z = (E - E.mean(0)) / np.maximum(E.std(0), 1e-9)
    return np.vstack([Z[:, g].mean(1) for g in mods]).T.argmax(1)


_orig_prep = m58.prep


def prep(sec, seed):
    key = ("sp", sec, seed)
    if key in m58._C:
        return m58._C[key]
    D = _orig_prep(sec, seed)                               # reuse costs, niche features, fitness, etc.
    o, NF, X, lib, ix = m58.base()
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    allc = np.asarray(z["cell_names"]); scells = np.asarray(z["spot_cell_names"])
    sof = np.asarray(z["spot_of_cell"]); m = D["S"]["M"].shape[1]
    Es = m58.logexpr(scells); xy_s = o.loc[scells, ["centroid_x", "centroid_y"]].values
    mods, top = programs(Es, xy_s)
    GP = np.unique(np.concatenate(mods))
    st_s = states(Es, mods); st_d = pd.Series(states(m58.logexpr(allc), mods), index=allc)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    import ot
    sc_ = StandardScaler().fit(NF.loc[scells].values)
    pca = PCA(10, random_state=0).fit(sc_.transform(NF.loc[scells].values))
    f = lambda A: pca.transform(sc_.transform(A))
    clf = LogisticRegression(max_iter=3000).fit(f(NF.loc[scells].values), st_s)
    spotNF = np.vstack([NF.loc[scells[sof == j]].values.mean(0) for j in range(m)])
    cls = list(clf.classes_); cells = D["cells"]
    ic = np.array([cls.index(v) if v in cls else -1 for v in st_d.loc[cells].values])
    rest = np.setdiff1d(np.arange(Es.shape[1]), GP)
    Xsc = np.asarray(z["X_sc"], float)[D["use"]]; Xst = np.asarray(z["X_st"], float)
    Mh = ot.dist(Xsc[:, rest], Xst[:, rest]); Mh /= Mh.max()
    out = dict(D); out.update(P_spot=clf.predict_proba(f(spotNF)), P_own=clf.predict_proba(f(NF.loc[cells].values)),
                              ic=ic, nGP=len(GP), M={"held-out genes": Mh, "all genes": D["M"]["all genes"]})
    m58._C[key] = out
    return out


m58.prep = prep


def job(args):
    return m58.job(args)

if __name__ == "__main__":
    import sys
    from multiprocessing import Pool
    from scipy.stats import spearmanr
    m55 = m58.m55
    jobs = [(s, sd, None, None) for s in m55.SECS for sd in (0, 1)]
    jobs += [(s, sd, v, float(w)) for s in m55.SECS for sd in (0, 1)
             for v in ("held-out genes", "all genes") for w in m55.WGRID]
    out = []
    with Pool(24) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            out += r
    df = pd.DataFrame(out).sort_values(["sec", "seed", "variant", "w"])
    df.to_csv(HERE / "loss_alignment_spatialprog.csv", index=False)
    print(f"program genes removed from the cost: {sorted(df.nGP.unique())} of 175\n")
    rows = []
    for (sec, seed, var), g in df[df.variant != "truth"].groupby(["sec", "seed", "variant"]):
        bi, ni = g.err.idxmin(), g.niche.idxmax()
        rows.append(dict(split=f"{sec} s{seed}", variant=var, expr_only=g[g.w == 0].err.iloc[0],
                         w_best=g.loc[bi, "w"], err_best=g.loc[bi, "err"], w_niche=g.loc[ni, "w"],
                         err_niche=g.loc[ni, "err"], rho=spearmanr(g.err, -g.niche)[0]))
    r = pd.DataFrame(rows); print(r.round(3).to_string(index=False))
    for var, g in r.groupby("variant"):
        gain = (g.expr_only - g.err_niche).sum() / (g.expr_only - g.err_best).sum()
        print(f"\n{var}: expr-only {g.expr_only.mean():.0f} | optimum {g.err_best.mean():.0f} | niche pick "
              f"{g.err_niche.mean():.0f} um ({100*gain:.0f}% of the tree's gain); niche picks w=0 in "
              f"{(g.w_niche == 0).sum()}/6; mean rho {g.rho.mean():+.2f}")
