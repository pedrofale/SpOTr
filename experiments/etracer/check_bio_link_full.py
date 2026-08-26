"""THE CHECK, at full scale: does any scRNA-seq clone feature predict spatial dispersion?

The first attempt used the 2,000-bin / 1,515-cell subsample prepared for the OT runs, which
left only 7 usable clones -- at that n the best-of-6-features AUC of 0.833 sat BELOW its own
permutation null (mean 0.874, p = 0.76), and the multivariate LOO-CV AUC was 0.167, i.e. pure
overfitting. Uninformative, not negative.

Here: all 7,568 LE bins for the spatial labels, and up to 200 cells per clone drawn from the
full 91,411-cell matrix, giving 23 usable clones (>=20 bins, >=15 cells).

Labels (from barcodes, spatial): per-clone kNN coherence and spatial spread.
Features (scRNA-seq only, no spatial, no mapping): size, expression heterogeneity, distance
from clone centroid, cross-validated separability, library size, genes detected.

n = 23 is still small, so the headline is the single-feature Spearman correlation against
CONTINUOUS coherence (no information thrown away by binarising), plus a permutation null that
accounts for having screened several features.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
S = pd.read_csv(HERE / "link_cells.csv")
cols = np.loadtxt(HERE / "link_cols.txt", dtype=int)
remap = {c: i for i, c in enumerate(cols)}

E = pd.read_csv(HERE / "link_entries.txt", sep=" ", header=None,
                names=["g", "c", "v"], dtype={"g": np.int32, "c": np.int32, "v": np.float32})
M = coo_matrix((E.v, (E.g.values - 1, E.c.map(remap).values)),
               shape=(32285, len(cols))).tocsc()
print(f"matrix {M.shape}, {M.nnz:,} nonzeros")

X = np.asarray(M.todense(), dtype=np.float32).T          # cells x genes
lib = X.sum(1)
ngene = (X > 0).sum(1)
X = np.log1p(X / np.maximum(lib[:, None], 1) * 1e4)
v = X.var(0)
hv = np.argsort(-v)[:2000]
Xh = X[:, hv]
clone_of = S.set_index("col").clone.to_dict()
cl = np.array([clone_of[c] for c in cols])

# ---- spatial labels from ALL LE bins ----
sp = pd.read_csv(HERE / "Tumor_Immune_Evasion_eTRACER/LineageTracing/spatial/spatial_lineage_LE.csv")
xy = sp[["x", "y"]].to_numpy(float)
bc = sp.lineageGrp.astype(str).to_numpy()
diam = np.linalg.norm(xy.max(0) - xy.min(0))
KNN = 12
nbr = np.argsort(cdist(xy, xy), axis=1)[:, 1:KNN + 1]

sc_full = pd.read_csv(HERE / "Tumor_Immune_Evasion_eTRACER/LineageTracing/single_cell/sc_cell_lineage.csv",
                      low_memory=False).lineageGrp.value_counts()
rows = []
for c in sorted(set(cl)):
    bsel, csel = bc == c, cl == c
    if bsel.sum() < 20:
        continue
    P = xy[bsel]
    rows.append(dict(
        clone=c, n_bins=int(bsel.sum()), n_cells_used=int(csel.sum()),
        coherence=float((bc[nbr[bsel]] == c).mean() - bsel.mean()),
        spread=float(np.linalg.norm(P - P.mean(0), axis=1).mean() / diam),
        size=int(sc_full[c]),
        het=float(Xh[csel].std(0).mean()),
        het_pair=float(np.linalg.norm(Xh[csel] - Xh[csel].mean(0), axis=1).mean()),
        expr_sep=float(cross_val_score(LogisticRegression(max_iter=800, class_weight="balanced"),
                                       Xh, csel.astype(int), cv=3,
                                       scoring="balanced_accuracy").mean()),
        libsize=float(np.median(lib[csel])),
        n_detected=float(np.median(ngene[csel])),
    ))
F = pd.DataFrame(rows)
print(f"\n{len(F)} clones\n")
print(F.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

FEATS = ["size", "het", "het_pair", "expr_sep", "libsize", "n_detected"]
print(f"\n{'feature':>12}{'rho vs coherence':>19}{'p':>8}{'rho vs spread':>16}{'p':>8}")
print("-" * 63)
best_r = None
for f_ in FEATS:
    r1, p1 = stats.spearmanr(F[f_], F.coherence)
    r2, p2 = stats.spearmanr(F[f_], F.spread)
    print(f"{f_:>12}{r1:>+19.3f}{p1:>8.3f}{r2:>+16.3f}{p2:>8.3f}")
    if best_r is None or abs(r1) > abs(best_r[1]):
        best_r = (f_, r1, p1)

# permutation null for the BEST-of-k |rho|, which is what we would actually act on
rng = np.random.default_rng(0)
null = [max(abs(stats.spearmanr(F[f_], rng.permutation(F.coherence.values))[0]) for f_ in FEATS)
        for _ in range(5000)]
pv = float(np.mean(np.array(null) >= abs(best_r[1])))
print(f"\nbest |rho| vs coherence: {best_r[0]} = {best_r[1]:+.3f}")
print(f"permutation null for best-of-{len(FEATS)} |rho|: mean {np.mean(null):.3f}, "
      f"95th pct {np.percentile(null, 95):.3f}  ->  p = {pv:.4f}")

y = (F.coherence < F.coherence.median()).astype(int).to_numpy()
Xf = StandardScaler().fit_transform(F[FEATS])
pp = np.zeros(len(F))
for i in range(len(F)):
    tr = np.arange(len(F)) != i
    pp[i] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        Xf[tr], y[tr]).predict_proba(Xf[[i]])[:, 1]
print(f"multivariate LOO-CV AUC (predicting dispersed): {roc_auc_score(y, pp):.3f}")
print(f"\n--> " + ("LINK PRESENT: worth fitting a transferable rule" if pv < 0.05 else
                   "NO USABLE LINK at this sample size: transfer would carry nothing"))
F.to_csv(HERE / "bio_link_check_full.csv", index=False)
