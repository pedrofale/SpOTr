"""THE CHECK, with the right label: does any scRNA-seq clone feature predict whether the TREE
predicts spatial position within that clone?

Coherence and spread were the wrong targets. What alpha_k needs to know is not "is this clone
clustered" but "does its lineage tree predict where its cells sit" -- and etracer_premise.py
already measured exactly that per clone, non-circularly, as a Mantel r between tree distance
and spatial distance with a within-clone permutation null.

So: label = per-clone tree-space concordance r (LE stage). Features = scRNA-seq only, no
spatial information and no mapping. If a feature predicts r, a rule fitted on this dataset
could set alpha_k on a new tumour. If not, the transfer idea carries nothing.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
F = pd.read_csv(HERE / "bio_link_check_full.csv")
P = pd.read_csv(HERE / "etracer_premise.csv")
P = P[P.stage == "LE"].copy()
P["cid"] = P.clone.str.split("-").str[0]
P = P.groupby("cid").agg(r=("r", "mean"), pval=("p", "min"), n_leaves=("n_leaves", "max")).reset_index()

M = F.merge(P, left_on="clone", right_on="cid", how="inner")
print(f"{len(M)} clones with BOTH scRNA features and a measured tree-space concordance\n")
print(M[["clone", "n_leaves", "r", "pval", "size", "het", "expr_sep", "libsize", "coherence"]]
      .sort_values("r", ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

FEATS = ["size", "het", "het_pair", "expr_sep", "libsize", "n_detected"]
print(f"\n{'feature':>12}{'rho vs tree-space r':>22}{'p':>8}")
print("-" * 43)
best = None
for f_ in FEATS:
    rho, pv = stats.spearmanr(M[f_], M.r)
    print(f"{f_:>12}{rho:>+22.3f}{pv:>8.3f}")
    if best is None or abs(rho) > abs(best[1]):
        best = (f_, rho, pv)

rng = np.random.default_rng(0)
null = [max(abs(stats.spearmanr(M[f_], rng.permutation(M.r.values))[0]) for f_ in FEATS)
        for _ in range(5000)]
pv_adj = float(np.mean(np.array(null) >= abs(best[1])))
print(f"\nbest: {best[0]} rho = {best[1]:+.3f} (nominal p = {best[2]:.3f})")
print(f"permutation null for best-of-{len(FEATS)} |rho|: mean {np.mean(null):.3f}, "
      f"95th pct {np.percentile(null, 95):.3f}  ->  adjusted p = {pv_adj:.4f}")

y = (M.r > M.r.median()).astype(int).to_numpy()
X = StandardScaler().fit_transform(M[FEATS])
pp = np.zeros(len(M))
for i in range(len(M)):
    tr = np.arange(len(M)) != i
    pp[i] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        X[tr], y[tr]).predict_proba(X[[i]].reshape(1, -1))[0, 1]
auc = roc_auc_score(y, pp)
print(f"multivariate LOO-CV AUC (predicting high tree-space concordance): {auc:.3f}")

# what power do we have? smallest |rho| detectable at n with 80% power, alpha 0.05
n = len(M)
z = 0.5 * np.log((1 + 0.5) / (1 - 0.5))
det = np.tanh((1.96 + 0.84) / np.sqrt(max(n - 3, 1)))
print(f"\npower: with n = {n} clones, the smallest correlation detectable at 80% power "
      f"is |rho| ~ {det:.2f}")
print(f"observed best |rho| = {abs(best[1]):.3f}")
print(f"\n--> " + ("LINK PRESENT: worth fitting a transferable rule" if pv_adj < 0.05 else
                   f"NO DETECTABLE LINK. With n = {n} this rules out |rho| above ~{det:.2f}; "
                   f"a weaker link could exist but would need far more clones to establish "
                   f"and would carry little anyway."))
M.to_csv(HERE / "bio_link_premise.csv", index=False)
