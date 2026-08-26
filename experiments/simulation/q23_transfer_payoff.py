"""Q23: what does a working transfer rule actually buy?

Q22 showed the predictor recovers the biological link almost exactly (link AUC 0.84 ->
transfer AUC 0.83). Q21 showed that at AUC 0.56 the mapping gain is zero. The missing number
is what AUC ~0.83 is worth, evaluated the same way: leave-one-seed-out predicted alpha_k
against the best global alpha and against the oracle.
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
FEATS = ["size", "expr_sep", "tree_ratio", "disp", "bimod", "entropy", "eff_spots", "het"]
files = sorted(HERE.glob("sim_bio0p6_s*.npz"))
rows = []
for f in files:
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    seed = int(f.stem.split("_s")[1])
    cidx, is_disp, X_sc = d["cell_clade"], d["is_dispersed"], d["X_sc"]
    K = len(d["clades"]); n = len(cidx)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    g = g0
    for _ in range(J):
        C = build_perclade_alpha_cost_v2(jnp.zeros((K,), jnp.float32), Cf, C_T, C_S, a, b, g,
                                         jnp.asarray(A), sF, sL, equalize="measured")
        g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
    gn = np.asarray(g); w = gn / np.maximum(gn.sum(1, keepdims=True), 1e-30)
    pred = w @ d["xy_st"]; diam = np.linalg.norm(d["xy_st"].max(0) - d["xy_st"].min(0))
    Ct = np.asarray(d["C_tree"], float); ent = -(w * np.log(np.maximum(w, 1e-30))).sum(1)
    for k in range(K):
        sel = cidx == k; P = pred[sel]
        km = KMeans(n_clusters=2, n_init=6, random_state=0).fit(P)
        within = np.mean([np.linalg.norm(P[km.labels_ == c] - km.cluster_centers_[c], axis=1).mean()
                          for c in (0, 1) if (km.labels_ == c).sum() > 1])
        tot = max(np.linalg.norm(P - P.mean(0), axis=1).mean(), 1e-12)
        mass = w[sel].sum(0); mass = mass / max(mass.sum(), 1e-30)
        rows.append(dict(seed=seed, k=k, dispersed=int(is_disp[k]), size=sel.sum(),
                         expr_sep=cross_val_score(LogisticRegression(max_iter=500, class_weight="balanced"),
                                                  X_sc, sel.astype(int), cv=3,
                                                  scoring="balanced_accuracy").mean(),
                         tree_ratio=Ct[np.ix_(sel, sel)].mean() / max(Ct[np.ix_(sel, ~sel)].mean(), 1e-12),
                         disp=tot / diam, bimod=within / tot, entropy=ent[sel].mean(),
                         eff_spots=float(np.exp(-(mass * np.log(np.maximum(mass, 1e-30))).sum())),
                         het=float(np.asarray(X_sc)[sel].std(0).mean())))
F = pd.DataFrame(rows)
seeds = sorted(F.seed.unique())
preds, aucs = {}, []
for s in seeds:
    tr, te = F[F.seed != s], F[F.seed == s]
    sc = StandardScaler().fit(tr[FEATS])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(tr[FEATS]), tr.dispersed)
    preds[s] = clf.predict_proba(sc.transform(te[FEATS]))[:, 1]
    aucs.append(roc_auc_score(te.dispersed, preds[s]))
print(f"leave-one-seed-out AUC: mean {np.mean(aucs):.3f}\n")

print(f"{'seed':>5}{'alpha=0':>10}{'best global':>13}{'PREDICTED':>11}{'oracle':>9}")
print("-" * 48)
res = []
for s in seeds:
    d = np.load(HERE / f"sim_bio0p6_s{s}.npz", allow_pickle=True)
    ci, sci, isd = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    K = len(d["clades"]); n = len(ci)
    A = np.zeros((n, K), np.float32); A[np.arange(n), ci] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    om = jnp.asarray(A); gz = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gz, eps=EPS)
    def run(ak, _Cf=Cf, _CT=C_T, _CS=C_S, _a=a, _b=b, _om=om, _sF=sF, _sL=sL, _gz=gz,
            _sci=sci, _ci=ci):
        g = _gz
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)),
                                             _Cf, _CT, _CS, _a, _b, g, _om, _sF, _sL,
                                             equalize="measured")
            g, _u = S.sinkhorn_unrolled_safe(C, _a, _b, EPS, T)
        g = np.asarray(g)
        return (_sci[g.argmax(1)] == _ci).mean() if np.isfinite(g).all() else np.nan
    sw = {v: run(np.full(K, v)) for v in GRID}
    bv = max(sw, key=lambda x: sw[x])
    nt, nh = (~isd).astype(float), 1.0 - preds[s]
    bp = bo = -np.inf
    for lo, hi in [(0.0, 0.35), (0.0, 0.50), (0.0, 0.70), (0.05, 0.50)]:
        bp = max(bp, run(lo + (hi - lo) * nh)); bo = max(bo, run(lo + (hi - lo) * nt))
    res.append(dict(seed=s, a0=sw[0.0], glob=sw[bv], pred=bp, oracle=bo))
    print(f"{s:>5}{sw[0.0]:>10.4f}{sw[bv]:>13.4f}{bp:>11.4f}{bo:>9.4f}")
R = pd.DataFrame(res); N = len(R)
print("-" * 48)
print(f"{'mean':>5}{R.a0.mean():>10.4f}{R.glob.mean():>13.4f}{R.pred.mean():>11.4f}{R.oracle.mean():>9.4f}")
den = (R.oracle - R.glob).mean()
print(f"\npredicted beats best global in {int(((R.pred-R.glob)>0.005).sum())}/{N}, "
      f"mean {(R.pred-R.glob).mean():+.4f}")
print(f"predicted beats alpha=0 in      {int(((R.pred-R.a0)>0.005).sum())}/{N}, "
      f"mean {(R.pred-R.a0).mean():+.4f}")
print(f"recovers {100*(R.pred-R.glob).mean()/den:.0f}% of the oracle's gain over global"
      if den > 1e-4 else "oracle gain over global not positive")
R.to_csv(HERE / "q23_transfer_payoff.csv", index=False)
