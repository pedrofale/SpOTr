"""Q14: a two-stage plug-in estimator for alpha_k, since it cannot be learned from the loss.

Q10 and Q13 established that no expression-reconstruction objective can select alpha > 0:
the linear OT term IS the operator that maximises expression agreement, so any loss of that
form is monotone in alpha and pinned at 0. Held-out genes do not help, because the failure is
not overfitting.

But alpha_k does not have to be learned end-to-end. What it needs to know is which clades are
spatially coherent, and that can be ESTIMATED from a first-pass mapping:

  stage 1   map with alpha = 0 (expression only, no tree)
  stage 2   for each clade, measure how spatially concentrated its mapped cells are
  stage 3   set alpha_k high for concentrated clades, low for dispersed ones
  stage 4   re-map

Stage 1 uses no tree at all, so the estimate is not circular -- it is not measuring the thing
the GW term optimises. The obvious risk is a confound: a clade that is truly compact but
CONFUSED by expression will also look dispersed in stage 1. Whether that confound sinks the
estimator is exactly what this measures.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import roc_auc_score
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
rows, aucs_extra = [], []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    cidx, spot_ci, is_disp = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    xy_st = d["xy_st"]
    K = len(d["clades"])
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    diam = np.linalg.norm(xy_st.max(0) - xy_st.min(0))

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

    # ---- stage 1: expression-only map, then per-clade spatial dispersion ----
    g1 = solve(np.zeros(K))
    w1 = g1 / np.maximum(g1.sum(1, keepdims=True), 1e-30)
    pred = w1 @ xy_st                                  # soft position of each cell
    # Plain spread cannot separate the two failure modes: a COMPACT clade that expression
    # confuses with its partner lands as TWO TIGHT BLOBS (its territory and the partner's),
    # which has the same spread as a genuinely uniform clade. Bimodality is the discriminator
    # -- 2-means leaves tight residual clusters for the confused case and clusters almost as
    # wide as the whole for the uniform case.
    from sklearn.cluster import KMeans
    disp_hat, bimod = np.zeros(K), np.zeros(K)
    for k in range(K):
        P = pred[cidx == k]
        disp_hat[k] = np.linalg.norm(P - P.mean(0), axis=1).mean() / diam
        km = KMeans(n_clusters=2, n_init=6, random_state=0).fit(P)
        within = np.mean([np.linalg.norm(P[km.labels_ == c] - km.cluster_centers_[c], axis=1).mean()
                          for c in (0, 1) if (km.labels_ == c).sum() > 1])
        bimod[k] = within / max(np.linalg.norm(P - P.mean(0), axis=1).mean(), 1e-12)
    auc = roc_auc_score(is_disp.astype(int), disp_hat)
    auc_b = roc_auc_score(is_disp.astype(int), bimod)
    aucs_extra.append(dict(seed=seed, auc_spread=auc, auc_bimodal=auc_b))

    # ---- stages 3-4: alpha_k from the estimate, vs global and vs oracle ----
    sw = {v: acc(solve(np.full(K, v))) for v in GRID}
    bv = max(sw, key=lambda k: sw[k])
    rank = stats.rankdata(bimod) / K                    # 1 = most uniform (truly dispersed)
    need_hat = 1.0 - rank                               # concentrated -> high alpha
    need_true = (~is_disp).astype(float)
    best_p = best_o = -np.inf
    for lo, hi in [(0.0, 0.35), (0.0, 0.50), (0.0, 0.70), (0.05, 0.50), (0.05, 0.70)]:
        best_p = max(best_p, acc(solve(lo + (hi - lo) * need_hat)))
        best_o = max(best_o, acc(solve(lo + (hi - lo) * need_true)))
    rows.append(dict(seed=seed, auc=auc, a0=sw[0.0], glob=sw[bv], plugin=best_p, oracle=best_o))
    print(f"seed {seed:>2}: AUC spread {auc:.3f} bimodal {auc_b:.3f} | a=0 {sw[0.0]:.4f}  global {sw[bv]:.4f}  "
          f"plug-in {best_p:.4f}  oracle {best_o:.4f}")

r = pd.DataFrame(rows)
r["p_vs_g"] = r.plugin - r.glob
r["o_vs_g"] = r.oracle - r.glob
n = len(r)
print("\n" + "=" * 84)
print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
e = pd.DataFrame(aucs_extra)
print(f"\nstage-1 estimator AUC:  plain spread {e.auc_spread.mean():.3f}   "
      f"bimodality {e.auc_bimodal.mean():.3f}   "
      f"(bimodality above 0.5 in {int((e.auc_bimodal > 0.5).sum())}/{len(e)})")
tb, pb = stats.ttest_1samp(e.auc_bimodal, 0.5)
print(f"  bimodality vs chance: t={tb:+.2f}, p={pb:.4f}")
e.to_csv(HERE / "q14_estimators.csv", index=False)
print(f"\nstage-1 dispersion estimate: mean AUC {r.auc.mean():.3f}  "
      f"(above 0.5 in {int((r.auc > 0.5).sum())}/{n}, range {r.auc.min():.3f}-{r.auc.max():.3f})")
t, pv = stats.ttest_1samp(r.auc, 0.5)
print(f"  vs chance: t={t:+.2f}, p={pv:.4f}")
print(f"\nplug-in vs best global : mean {r.p_vs_g.mean():+.4f}  positive in {int((r.p_vs_g > 0.005).sum())}/{n}")
print(f"oracle  vs best global : mean {r.o_vs_g.mean():+.4f}  positive in {int((r.o_vs_g > 0.005).sum())}/{n}")
print(f"plug-in recovers {100 * r.p_vs_g.mean() / max(r.o_vs_g.mean(), 1e-9):.0f}% of the oracle gain")
r.to_csv(HERE / "q14_plugin.csv", index=False)
