"""FIRST LOOK (2026-10-12): which biological links between lineage, state and niche exist in PEtracer M3?
Ground truth (positions, tree, clades) is used here only to MEASURE the links a label-free loss could rely on.

 (a) growth: does tree-derived fitness track position (tumour-boundary distance)?  per cell and per clade
 (b) inheritance vs induction: how much of a cell's state (hotspot module) is explained by its clade
     (inherited) vs its niche (local non-tumour composition + boundary distances), cross-validated;
     and per clade, does the niche share vary more than a size-matched random grouping would?
"""
import numpy as np, pandas as pd, treedata as td
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

t = td.read_h5td("/home/pedro/.treedata/datasets/koblan25_tumor.h5td")
o = t.obs.copy()
sd = pd.DataFrame(np.asarray(t.obsm["subtype_density"]), index=o.index)
tr = o[o.fitness.notna() & o.clade.notna() & o.hotspot_module.notna()].copy()
print("traced cells with fitness, clade, module:", len(tr), " per section:", tr["sample"].value_counts().to_dict())

print("\n(a) GROWTH: fitness vs distance to tumour boundary (negative = fitter at the edge)")
for sec, g in tr.groupby("sample", observed=True):
    r, p = spearmanr(g.fitness, g.tumor_boundary_dist)
    cl = g.groupby("clade", observed=True).agg(f=("fitness", "mean"), d=("tumor_boundary_dist", "mean"), n=("fitness", "size"))
    cl = cl[cl.n >= 30]
    rc, pc = spearmanr(cl.f, cl.d)
    print(f"   {sec}: per cell rho = {r:+.3f} (p={p:.1e}, n={len(g)})   per clade rho = {rc:+.3f} (p={pc:.2f}, n={len(cl)} clades)")

print("\n(b) INHERITANCE vs INDUCTION of state (hotspot module, 4 classes), 5-fold CV accuracy")
rng = np.random.default_rng(0)
for sec, g in tr.groupby("sample", observed=True):
    y = g.hotspot_module.astype(str).values
    C = pd.get_dummies(g.clade.astype(str)).values.astype(float)
    niche_raw = np.c_[sd.loc[g.index].values, g[["tumor_boundary_dist"]].values,
                      g[["lung_boundary_dist"]].fillna(g.lung_boundary_dist.max()).values]
    N = PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(niche_raw))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    def acc(X):
        return (cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv) == y).mean()
    base = pd.Series(y).value_counts(normalize=True).max()
    a_c, a_n, a_b = acc(C), acc(N), acc(np.c_[C, N])
    print(f"   {sec}: majority {base:.3f} | clade {a_c:.3f} | niche {a_n:.3f} | both {a_b:.3f}")
    # per-clade niche dependence: within each clade, how well does niche predict state (CV), vs random groups
    pred_n = cross_val_predict(LogisticRegression(max_iter=2000), np.c_[C, N], y, cv=cv)
    pred_c = cross_val_predict(LogisticRegression(max_iter=2000), C, y, cv=cv)
    gain = pd.Series((pred_n == y).astype(float) - (pred_c == y).astype(float), index=g.index)
    per = gain.groupby(g.clade.astype(str)).mean(); sizes = g.clade.astype(str).value_counts()
    keep = sizes[sizes >= 30].index
    obs_sd = per[keep].std()
    lab = g.clade.astype(str).values; null = []
    for _ in range(500):
        perm = rng.permutation(lab)
        pp = gain.groupby(perm).mean(); null.append(pp[keep].std())
    null = np.array(null)
    print(f"        niche gain over clade, per clade: range {per[keep].min():+.3f}..{per[keep].max():+.3f}, "
          f"sd {obs_sd:.3f} vs size-matched random groups {null.mean():.3f} +- {null.std():.3f} "
          f"(p = {(null >= obs_sd).mean():.3f})")
