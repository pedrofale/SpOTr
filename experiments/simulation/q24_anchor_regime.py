"""Q24: two questions raised by treating alpha as an INTERPRETIVE readout.

The repositioned claim: map from expression + tree (robust), use the anchor only to learn
alpha, and read alpha_k as the per-clade structure-vs-expression contribution. That rests on
two things worth testing rather than assuming.

  A. Can anchor quality be assessed WITHOUT ground truth? If it can, "we do not know how good
     the anchor is" is false, and one would simply measure it and pick a strategy. Two
     label-free diagnostics, neither using spatial truth:
       cell_pred  cross-validated accuracy of predicting clone identity from the CELL-side
                  anchor (clone labels come from the tree, which is given)
       spot_fit   mean best correlation between a spot's anchor profile and any clone
                  signature -- how well spot anchors look like clone anchors at all
  B. Does the INTERPRETATION survive a degraded anchor? Q15 learned alpha from a clean anchor
     (AUC 0.96-0.98). If the readout collapses once the anchor is poor, the interpretive claim
     inherits the same fragility the mapping claim had.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")

EPS, T, J = 0.02, 50, 3
QUAL = [("clean", 0.0, 150), ("good", 1.0, 50), ("poor", 3.0, 20), ("very poor", 6.0, 8)]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    rg = np.random.default_rng(11000 + seed)
    cidx, spot_ci, is_disp = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    K = len(d["clades"]); n = len(cidx)
    comp = (~is_disp).astype(int)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    X_sc, X_st = np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)
    cnv_c, cnv_s = np.asarray(d["cnv_clone"], float), np.asarray(d["cnv_spot"], float)
    nb_all = cnv_s.shape[1]
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc.astype(np.float32)),
                                             ad.AnnData(X_st.astype(np.float32)),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)

    def acc_of(g):
        g = np.asarray(g)
        return (spot_ci[g.argmax(1)] == cidx).mean() if np.isfinite(g).all() else np.nan

    def feature_route(cell_anchor, spot_anchor):
        z = lambda M: (M - M.mean(0)) / np.maximum(M.std(0), 1e-9)
        Cf2, CT2, CS2, a2, b2 = S.prepare_ot_inputs(
            ad.AnnData(np.hstack([z(X_sc), z(cell_anchor)]).astype(np.float32)),
            ad.AnnData(np.hstack([z(X_st), z(spot_anchor)]).astype(np.float32)),
            d["C_tree"], d["C_space"])
        g = a2[:, None] * b2[None, :]
        sF2, sL2 = S.make_cladexfgw_scales(Cf2, CT2, CS2, a2, b2, g, eps=EPS)
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.zeros((K,), jnp.float32), Cf2, CT2, CS2,
                                             a2, b2, g, omega, sF2, sL2, equalize="measured")
            g, _u = S.sinkhorn_unrolled_safe(C, a2, b2, EPS, T)
        return acc_of(g)

    for name, nz, nb in QUAL:
        keep = rg.choice(nb_all, size=nb, replace=False)
        cell_anchor = cnv_c[cidx][:, keep] + rg.normal(0, 0.25 + 0.15 * nz, size=(n, nb))
        spot_anchor = cnv_s[:, keep] + rg.normal(0, 0.15 * nz, size=(cnv_s.shape[0], nb))
        sig = jnp.asarray((cnv_c[:, keep] * 2.0).astype(np.float32))

        # ---- A: label-free diagnostics ----
        cell_pred = cross_val_score(LogisticRegression(max_iter=600, class_weight="balanced"),
                                    cell_anchor, cidx, cv=3, scoring="accuracy").mean()
        Sg = cnv_c[:, keep]
        Sz = (Sg - Sg.mean(1, keepdims=True)) / np.maximum(Sg.std(1, keepdims=True), 1e-9)
        Yz = (spot_anchor - spot_anchor.mean(1, keepdims=True)) / \
            np.maximum(spot_anchor.std(1, keepdims=True), 1e-9)
        spot_fit = float((Yz @ Sz.T / nb).max(1).mean())

        # ---- B: interpretation from a degraded anchor ----
        res = S.learn_alpha_gamma_perclade_alpha_data(
            Cf, jnp.asarray(spot_anchor.astype(np.float32)), C_T, C_S, a, b,
            jnp.asarray(A), sig, omega, eps=EPS, T_sinkhorn=50, J_alt=3, K_outer=300, lr=0.05,
            alpha_init=jnp.full((K,), 0.10), gamma0=gi, use_loss_ls=True)
        al = np.asarray(res["alpha"])
        ok = np.isfinite(al).all()
        rows.append(dict(seed=seed, quality=name, cell_pred=cell_pred, spot_fit=spot_fit,
                         auc=roc_auc_score(comp, al) if ok else np.nan,
                         a_c=al[comp == 1].mean() if ok else np.nan,
                         a_d=al[comp == 0].mean() if ok else np.nan,
                         spotr=acc_of(res["coupling"]) if ok else np.nan,
                         feat=feature_route(cell_anchor, spot_anchor)))
        print(f"  seed {seed:>2} {name:>10}: cell_pred {cell_pred:.3f} spot_fit {spot_fit:.3f} "
              f"| alpha AUC {rows[-1]['auc']:.3f}  SpOTr {rows[-1]['spotr']:.4f}  "
              f"feature {rows[-1]['feat']:.4f}")

r = pd.DataFrame(rows)
print("\n" + "=" * 88)
print(f"{'quality':>11}{'cell_pred':>11}{'spot_fit':>10}{'alpha AUC':>11}{'a compact':>11}"
      f"{'a disp':>9}{'SpOTr':>9}{'feature':>9}{'better':>9}")
print("-" * 88)
for name, _, _ in QUAL:
    s = r[r.quality == name]
    better = "feature" if s.feat.mean() > s.spotr.mean() else "SpOTr"
    print(f"{name:>11}{s.cell_pred.mean():>11.3f}{s.spot_fit.mean():>10.3f}{s.auc.mean():>11.3f}"
          f"{s.a_c.mean():>11.3f}{s.a_d.mean():>9.3f}{s.spotr.mean():>9.4f}{s.feat.mean():>9.4f}"
          f"{better:>9}")
print("-" * 88)
q = r.groupby("quality")[["cell_pred", "spot_fit", "feat", "spotr"]].mean()
order = [n for n, _, _ in QUAL]
q = q.loc[order]
print("\nA. do the label-free diagnostics rank anchor quality correctly?")
for dg in ["cell_pred", "spot_fit"]:
    rho, pv = stats.spearmanr(q[dg], q.feat)
    print(f"   {dg:>10} vs feature-route accuracy: Spearman {rho:+.2f} "
          f"(monotone across the 4 regimes: {'YES' if abs(rho) > 0.99 else 'no'})")
print(f"\n   per-seed: does cell_pred pick the better strategy?")
r["truth_feat_better"] = r.feat > r.spotr
for thr in [0.5, 0.6, 0.7, 0.8]:
    pick = r.cell_pred > thr
    print(f"     threshold cell_pred > {thr}: correct call in "
          f"{int((pick == r.truth_feat_better).sum())}/{len(r)} seed-quality combinations")
print(f"\nB. does the alpha readout survive degradation?  "
      f"AUC {' -> '.join(f'{r[r.quality==n].auc.mean():.3f}' for n, _, _ in QUAL)}")
r.to_csv(HERE / "q24_anchor_regime.csv", index=False)
