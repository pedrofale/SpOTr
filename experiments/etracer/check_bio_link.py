"""THE CHECK: on real eTRACER data, does any scRNA-seq-derived clone feature predict whether
that clone is spatially dispersed?

This is the decision criterion for the transfer idea. If some feature computable WITHOUT a
mapping predicts spatial dispersion at AUC ~0.8, a rule fitted here can be carried to a new
tumour and should return roughly a third of the oracle gain (simulation, q23). At ~0.6 it
carries nothing and the direction is not worth building.

Labels come from the barcodes: each Stereo-seq bin has a clone, so each clone's true spatial
spread is known. Features use ONLY the scRNA-seq matrix and the lineage tree -- nothing
spatial, no mapping -- because those are the only things available on an unmapped tumour.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "etracer_inputs.npz", allow_pickle=True)
X, Ct = np.asarray(d["X_sc"], float), np.asarray(d["C_tree"], float)
cell_clone = np.asarray(d["cell_clone"]).astype(str)
bin_clone = np.asarray(d["bin_clone"]).astype(str)
xy, clones = np.asarray(d["xy"], float), np.asarray(d["clones"]).astype(str)
diam = np.linalg.norm(xy.max(0) - xy.min(0))
D = cdist(xy, xy)
KNN = 12
nbr = np.argsort(D, axis=1)[:, 1:KNN + 1]

rows = []
for c in clones:
    bsel, csel = bin_clone == c, cell_clone == c
    if bsel.sum() < 20 or csel.sum() < 15:
        continue
    # ---- LABEL (barcodes, spatial): how clustered are this clone's bins? ----
    same = (bin_clone[nbr[bsel]] == c).mean()
    chance = bsel.mean()
    P = xy[bsel]
    rows.append(dict(
        clone=c, n_bins=int(bsel.sum()), n_cells=int(csel.sum()),
        coherence=float(same - chance),                       # kNN excess over chance
        spread=float(np.linalg.norm(P - P.mean(0), axis=1).mean() / diam),
        # ---- FEATURES (scRNA-seq + tree only, no spatial, no mapping) ----
        size=int(csel.sum()),
        het=float(X[csel].std(0).mean()),
        het_pair=float(np.linalg.norm(X[csel] - X[csel].mean(0), axis=1).mean()),
        expr_sep=float(cross_val_score(
            LogisticRegression(max_iter=800, class_weight="balanced"),
            X, csel.astype(int), cv=3, scoring="balanced_accuracy").mean()),
        tree_within=float(Ct[np.ix_(csel, csel)][np.triu_indices(csel.sum(), 1)].mean()),
        tree_ratio=float(Ct[np.ix_(csel, csel)].mean() / max(Ct[np.ix_(csel, ~csel)].mean(), 1e-12)),
    ))
F = pd.DataFrame(rows)
print(f"{len(F)} clones with >=20 bins and >=15 cells (of {len(clones)})\n")
print(F[["clone", "n_bins", "n_cells", "coherence", "spread", "het", "expr_sep", "tree_ratio"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

FEATS = ["size", "het", "het_pair", "expr_sep", "tree_within", "tree_ratio"]
print(f"\n{'feature':>14}{'Spearman vs coherence':>24}{'p':>9}{'  vs spread':>14}{'p':>9}")
print("-" * 72)
for f_ in FEATS:
    r1, p1 = stats.spearmanr(F[f_], F.coherence)
    r2, p2 = stats.spearmanr(F[f_], F.spread)
    print(f"{f_:>14}{r1:>+24.3f}{p1:>9.3f}{r2:>+14.3f}{p2:>9.3f}")

# binary: bottom half of coherence = "dispersed"
y = (F.coherence < F.coherence.median()).astype(int).to_numpy()
print(f"\nsingle-feature AUC for predicting DISPERSED (bottom half of coherence, n={len(F)}):")
best = None
for f_ in FEATS:
    a = roc_auc_score(y, -F[f_])
    a = max(a, 1 - a)                      # direction is free; a rule can use either sign
    print(f"  {f_:>14}  AUC {a:.3f}")
    if best is None or a > best[1]:
        best = (f_, a)

Xf = StandardScaler().fit_transform(F[FEATS])
loo = LeaveOneOut()
pp = np.zeros(len(F))
for tr, te in loo.split(Xf):
    if len(np.unique(y[tr])) < 2:
        continue
    pp[te] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        Xf[tr], y[tr]).predict_proba(Xf[te])[:, 1]
loo_auc = roc_auc_score(y, pp)
print(f"\nbest single feature : {best[0]}, AUC {best[1]:.3f}")
print(f"all features, LOO-CV: AUC {loo_auc:.3f}")
# how surprising is the best single-feature AUC given 6 features were screened on n clones?
rng = np.random.default_rng(0)
null = [max(max(roc_auc_score(rng.permutation(y), -F[f_]),
                1 - roc_auc_score(rng.permutation(y), -F[f_])) for f_ in FEATS)
        for _ in range(2000)]
print(f"permutation null for the BEST-of-{len(FEATS)} AUC: mean {np.mean(null):.3f}, "
      f"95th pct {np.percentile(null, 95):.3f}  ->  p = {np.mean(np.array(null) >= best[1]):.3f}")
print(f"\n--> " + ("LINK PRESENT: a transferable rule is worth fitting"
                   if loo_auc > 0.75 or best[1] > 0.80 else
                   "NO USABLE LINK: transfer would carry nothing"))
F.to_csv(HERE / "bio_link_check.csv", index=False)
