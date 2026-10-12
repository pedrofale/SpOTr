"""LOSS ALIGNMENT (2026-10-12): do biology-motivated, label-free criteria peak where placement accuracy
peaks along the FGW weight path?  PEtracer construction, exact FGW (55_'s solver), one split per section.

Candidate criteria, each computable in a genuinely unmatched dataset:
  growth   Spearman(tree-derived fitness of cell i, tumour-boundary distance of the spot it is mapped to)
           -- fitter lineages should sit nearer the growing edge (measured on truth: rho -0.10..-0.15)
  niche    mean log p(state module of cell i | niche of its spot), the model p(state | niche) trained on
           the SPOT side only (spot cells' own modules and niche features: local non-tumour composition,
           tumour- and lung-boundary distances) -- states are induced by niches
The truth row gives each criterion at the cells' true positions.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import importlib.util
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, treedata as td
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
T_OBS = None


def obs():
    global T_OBS
    if T_OBS is None:
        t = td.read_h5td("/home/pedro/.treedata/datasets/koblan25_tumor.h5td")
        o = t.obs.copy()
        nf = np.c_[np.asarray(t.obsm["subtype_density"]), o.tumor_boundary_dist.values,
                   o.lung_boundary_dist.fillna(o.lung_boundary_dist.max()).values]
        T_OBS = (o, pd.DataFrame(nf, index=o.index))
    return T_OBS


def job(args):
    sec, w = args
    o, NF = obs()
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    S = m55.load(sec)
    n = len(S["AGE"]); h = np.random.default_rng(0).permutation(n); use = h[n // 2:]
    cells = np.asarray(z["cell_names"])[use]
    scells = np.asarray(z["spot_cell_names"]); sof = np.asarray(z["spot_of_cell"]); m = S["M"].shape[1]
    # spot-side niche model
    sc_ = StandardScaler().fit(NF.loc[scells].values); pca = PCA(10, random_state=0).fit(sc_.transform(NF.loc[scells].values))
    f = lambda X: pca.transform(sc_.transform(X))
    ys = o.loc[scells, "hotspot_module"].astype(str).values
    clf = LogisticRegression(max_iter=3000).fit(f(NF.loc[scells].values), ys)
    spotNF = np.vstack([NF.loc[scells[sof == j]].values.mean(0) for j in range(m)])
    spot_tbd = np.array([o.loc[scells[sof == j], "tumor_boundary_dist"].mean() for j in range(m)])
    P_spot = clf.predict_proba(f(spotNF)); cls = list(clf.classes_)
    yc = o.loc[cells, "hotspot_module"].astype(str).values; ic = np.array([cls.index(v) for v in yc])
    fit = o.loc[cells, "fitness"].values
    rows = []
    if w is None:   # truth: each cell at its own position / own niche
        P_own = clf.predict_proba(f(NF.loc[cells].values))
        rows.append(dict(sec=sec, w=np.nan, err=0.0,
                         growth=spearmanr(fit, o.loc[cells, "tumor_boundary_dist"].values)[0],
                         niche=float(np.mean(np.log(P_own[np.arange(len(cells)), ic] + 1e-12)))))
        return rows
    AGE = S["AGE"][np.ix_(use, use)]; M = S["M"][use]; D = S["D"]; q = S["q"]
    p = np.ones(len(use)) / len(use)
    Wm = np.ones_like(AGE); np.fill_diagonal(Wm, 0.0); WPhi = Wm * 2.0 * AGE
    T0 = np.outer(p, q); g0 = 2.0 * (Wm @ T0 @ (D ** 2) - 2.0 * WPhi @ T0 @ D)
    sc = m55.dc_std(M) / max(m55.dc_std(g0), 1e-300)
    T = m55.solve(M, Wm * sc, WPhi * sc, D, D ** 2, p, q, w / (1 + w))
    am = T.argmax(1)
    err = float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"][use], axis=1)))
    rows.append(dict(sec=sec, w=w, err=err, growth=spearmanr(fit, spot_tbd[am])[0],
                     niche=float(np.mean(np.log(P_spot[am, ic] + 1e-12)))))
    return rows


if __name__ == "__main__":
    jobs = [(s, None) for s in m55.SECS] + [(s, float(w)) for s in m55.SECS for w in m55.WGRID]
    out = []
    with Pool(20) as pool:
        for r in pool.imap_unordered(job, jobs):
            out += r
    df = pd.DataFrame(out).sort_values(["sec", "w"], na_position="first")
    df.to_csv(HERE / "loss_alignment.csv", index=False)
    for sec, g in df.groupby("sec"):
        tru = g[g.w.isna()].iloc[0]; g = g[g.w.notna()]
        print(f"\n{sec}  (truth: growth rho {tru.growth:+.3f}, niche logL {tru.niche:.3f})")
        print(g[["w", "err", "growth", "niche"]].round(3).to_string(index=False))
        best = g.loc[g.err.idxmin()]
        print(f"   accuracy optimum w = {best.w:.3g} ({best.err:.0f} um) | growth most negative at w = "
              f"{g.loc[g.growth.idxmin()].w:.3g} | niche logL max at w = {g.loc[g.niche.idxmax()].w:.3g}"
              f" | spearman over w: err~growth {spearmanr(g.err, g.growth)[0]:+.2f}, err~niche {spearmanr(g.err, -g.niche)[0]:+.2f}")
