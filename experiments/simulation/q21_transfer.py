"""Q21: can alpha_k be PREDICTED from clade features, trained on ground-truth data?

alpha is unidentifiable from expression + tree alone (Q20), so it cannot be selected on the
target data. But it does not have to be. On a dataset where the mapping is known -- spatially
resolved lineage tracing, where barcodes label both cells and spots -- alpha_k can be fitted
supervised. Those fitted values do not transfer (alpha_k is indexed by clade, and clades are
dataset-specific), but a RULE mapping clade features to alpha_k can, provided the features are
computable without ground truth.

Features available on target data (no labels needed):
  size            cells in the clade
  expr_sep        cross-validated separability of the clade from the rest, scRNA-seq only
  tree_ratio      mean within-clade tree distance / mean cross-clade tree distance
  disp            spatial dispersion of the clade's cells under an alpha = 0 mapping
  bimod           2-means within/total spread of those positions
  entropy         mean row entropy of the alpha = 0 coupling for the clade's cells
  eff_spots       effective number of spots the clade's mass occupies

Target: is the clade spatially dispersed (alpha_k should be low) or compact (high).

Validation is LEAVE-ONE-SEED-OUT, so every evaluation is on a dataset the predictor never saw.
Note this is same-distribution transfer -- all seeds share a generative process -- which is the
EASIEST case. Real transfer across tumour types would be harder. If it fails here it is dead;
if it works here it still needs real-data validation.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
FEATS = ["size", "expr_sep", "tree_ratio", "disp", "bimod", "entropy", "eff_spots"]
feat_rows, ctx = [], {}
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    cidx, spot_ci, is_disp = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    xy_st, X_sc = d["xy_st"], d["X_sc"]
    K = len(d["clades"]); n = len(cidx)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

    def solve(ak):
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        return np.asarray(g)

    def acc(g):
        return (spot_ci[g.argmax(1)] == cidx).mean()

    g_null = solve(np.zeros(K))
    w = g_null / np.maximum(g_null.sum(1, keepdims=True), 1e-30)
    pred = w @ xy_st
    diam = np.linalg.norm(xy_st.max(0) - xy_st.min(0))
    Ct = np.asarray(d["C_tree"], float)
    ent = -(w * np.log(np.maximum(w, 1e-30))).sum(1)
    for k in range(K):
        sel = cidx == k
        P = pred[sel]
        km = KMeans(n_clusters=2, n_init=6, random_state=0).fit(P)
        within = np.mean([np.linalg.norm(P[km.labels_ == c] - km.cluster_centers_[c], axis=1).mean()
                          for c in (0, 1) if (km.labels_ == c).sum() > 1])
        tot = max(np.linalg.norm(P - P.mean(0), axis=1).mean(), 1e-12)
        mass = w[sel].sum(0); mass = mass / max(mass.sum(), 1e-30)
        feat_rows.append(dict(
            seed=seed, k=k, dispersed=int(is_disp[k]),
            size=sel.sum(),
            expr_sep=cross_val_score(LogisticRegression(max_iter=500, class_weight="balanced"),
                                     X_sc, sel.astype(int), cv=3,
                                     scoring="balanced_accuracy").mean(),
            tree_ratio=Ct[np.ix_(sel, sel)].mean() / max(Ct[np.ix_(sel, ~sel)].mean(), 1e-12),
            disp=tot / diam, bimod=within / tot, entropy=ent[sel].mean(),
            eff_spots=float(np.exp(-(mass * np.log(np.maximum(mass, 1e-30))).sum()))))
    print(f"  seed {seed}: features built for {K} clades")

F = pd.DataFrame(feat_rows)
seeds = sorted(F.seed.unique())
print(f"\n{len(F)} clades across {len(seeds)} seeds\n")

# --- leave-one-seed-out prediction of dispersion ---
aucs, preds = [], {}
for s in seeds:
    tr, te = F[F.seed != s], F[F.seed == s]
    sc = StandardScaler().fit(tr[FEATS])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(tr[FEATS]),
                                                                        tr.dispersed)
    pr = clf.predict_proba(sc.transform(te[FEATS]))[:, 1]
    preds[s] = pr
    aucs.append(roc_auc_score(te.dispersed, pr) if te.dispersed.nunique() > 1 else np.nan)
print(f"leave-one-seed-out AUC (predicted dispersion): mean {np.nanmean(aucs):.3f}  "
      f"per seed {', '.join(f'{v:.2f}' for v in aucs)}")

sc_all = StandardScaler().fit(F[FEATS])
clf_all = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc_all.transform(F[FEATS]),
                                                                        F.dispersed)
co = pd.Series(clf_all.coef_[0], index=FEATS).sort_values(key=abs, ascending=False)
print(f"\nstandardised coefficients (positive = predicts DISPERSED):")
print("  " + "  ".join(f"{k} {v:+.2f}" for k, v in co.items()))

# --- does the predicted alpha_k actually map better? ---
print(f"\n{'seed':>5}{'alpha=0':>10}{'best global':>13}{'PREDICTED':>11}{'oracle':>9}")
print("-" * 48)
res = []
for s in seeds:
    dd = np.load(HERE / f"sim_ctrl_s{s}.npz", allow_pickle=True)
    ci, sci, isd = dd["cell_clade"], dd["spot_clade"], dd["is_dispersed"]
    Kk = len(dd["clades"]); nn = len(ci)
    Am = np.zeros((nn, Kk), np.float32); Am[np.arange(nn), ci] = 1.0
    Cf2, CT2, CS2, a2, b2 = S.prepare_ot_inputs(ad.AnnData(dd["X_sc"]), ad.AnnData(dd["X_st"]),
                                                dd["C_tree"], dd["C_space"])
    om2 = jnp.asarray(Am); gz = a2[:, None] * b2[None, :]
    sF2, sL2 = S.make_cladexfgw_scales(Cf2, CT2, CS2, a2, b2, gz, eps=EPS)

    def run(ak, _Cf=Cf2, _CT=CT2, _CS=CS2, _a=a2, _b=b2, _om=om2, _sF=sF2, _sL=sL2, _gz=gz,
            _sci=sci, _ci=ci):
        g = _gz
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)),
                                             _Cf, _CT, _CS, _a, _b, g, _om, _sF, _sL,
                                             equalize="measured")
            g, _uv = S.sinkhorn_unrolled_safe(C, _a, _b, EPS, T)
        g = np.asarray(g)
        return (_sci[g.argmax(1)] == _ci).mean() if np.isfinite(g).all() else np.nan

    sw = {v: run(np.full(Kk, v)) for v in GRID}
    bv = max(sw, key=lambda x: sw[x])
    need_true = (~isd).astype(float)
    need_hat = 1.0 - preds[s]                       # p(dispersed) -> low alpha
    bp = bo = -np.inf
    for lo, hi in [(0.0, 0.35), (0.0, 0.50), (0.0, 0.70), (0.05, 0.50)]:
        bp = max(bp, run(lo + (hi - lo) * need_hat))
        bo = max(bo, run(lo + (hi - lo) * need_true))
    res.append(dict(seed=s, a0=sw[0.0], glob=sw[bv], pred=bp, oracle=bo))
    print(f"{s:>5}{sw[0.0]:>10.4f}{sw[bv]:>13.4f}{bp:>11.4f}{bo:>9.4f}")
R = pd.DataFrame(res); N = len(R)
print("-" * 48)
print(f"{'mean':>5}{R.a0.mean():>10.4f}{R.glob.mean():>13.4f}{R.pred.mean():>11.4f}{R.oracle.mean():>9.4f}")
print(f"\npredicted alpha beats best global in {int(((R.pred - R.glob) > 0.005).sum())}/{N} seeds, "
      f"mean {(R.pred - R.glob).mean():+.4f}")
print(f"predicted alpha beats alpha=0 in      {int(((R.pred - R.a0) > 0.005).sum())}/{N} seeds, "
      f"mean {(R.pred - R.a0).mean():+.4f}")
den = (R.oracle - R.glob).mean()
print(f"predicted recovers "
      + (f"{100 * (R.pred - R.glob).mean() / den:.0f}% of the oracle's gain over global"
         if den > 1e-4 else "n/a (oracle gain over global is not positive)"))
print(f"sanity: oracle beats global in {int(((R.oracle - R.glob) > 0.005).sum())}/{N} seeds "
      f"(should be ~all; if not, something is wrong)")
F.to_csv(HERE / "q21_features.csv", index=False)
R.to_csv(HERE / "q21_transfer.csv", index=False)
