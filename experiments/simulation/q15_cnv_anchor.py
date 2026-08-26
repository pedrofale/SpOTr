"""Q15: does a CNV anchor fix the objective?

Q10/Q13 established that no expression-reconstruction loss can select alpha > 0: the linear
OT term IS the operator that maximises expression agreement, so such a loss is monotone in
alpha with its argmin pinned at 0. Held-out genes did not help, because the failure is not
overfitting -- it is that the cost and the loss are the same modality.

deconvolution_loss_ls was in fact written for CNV profiles ("divide by 2.0 because the CNVs
are the copy number of the alleles"). Scoring it on CNVs while the feature cost is built from
EXPRESSION breaks the circularity: the loss now measures something the linear term does not
directly optimise, and CNVs track the tree, so they separate exactly the clades expression
confuses (measured: pair separation ratio ~1.0 for CNV vs 0.57 discrimination for expression).

  Q15a  is the CNV loss minimised at alpha > 0, and does its argmin match the accuracy optimum?
  Q15b  does per-clade alpha LEARNED against the CNV loss recover the compact/dispersed split?
  Q15c  does that learned alpha give a better map than the best global alpha?
"""
import functools, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
sweep, learn = [], []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        p = subprocess.run([sys.executable, str(HERE / "simulate_controlled.py"),
                            "--seed", str(seed), "--out", f.name], cwd=HERE, capture_output=True)
        if p.returncode != 0 or not f.exists():
            continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        print(f"seed {seed}: out of band -- skipped"); continue
    cidx, spot_ci, is_disp = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    K = len(d["clades"])
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
    X_sc, X_st = d["X_sc"], d["X_st"]
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

    Y_expr = jnp.asarray(np.asarray(X_st, np.float32))
    sig_expr = jnp.asarray(np.stack([X_sc[cidx == k].mean(0) for k in range(K)]).astype(np.float32) * 2.0)
    Y_cnv = jnp.asarray(np.asarray(d["cnv_spot"], np.float32))
    sig_cnv = jnp.asarray(np.asarray(d["cnv_clone"], np.float32) * 2.0)

    def solve(ak):
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        return g

    for av in GRID:
        g = solve(np.full(K, av)); gn = np.asarray(g)
        sweep.append(dict(seed=seed, alpha=av,
                          loss_expr=float(S.deconvolution_loss_ls(Y_expr, g, jnp.asarray(A), sig_expr)),
                          loss_cnv=float(S.deconvolution_loss_ls(Y_cnv, g, jnp.asarray(A), sig_cnv)),
                          acc=(spot_ci[gn.argmax(1)] == cidx).mean()))

    # ---- Q15b/c: learn per-clade alpha against the CNV loss ----
    gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sw = {v: [x for x in sweep if x["seed"] == seed and x["alpha"] == v][0]["acc"] for v in GRID}
    bv = max(sw, key=lambda k: sw[k])
    for init in [0.2, 0.5, 0.8]:
        res = S.learn_alpha_gamma_perclade_alpha_data(
            Cf, Y_cnv, C_T, C_S, a, b, jnp.asarray(A), sig_cnv, omega,
            eps=EPS, T_sinkhorn=50, J_alt=3, K_outer=300, lr=0.05,
            alpha_init=jnp.full((K,), init), gamma0=gi, use_loss_ls=True)
        al = np.asarray(res["alpha"])
        if not np.isfinite(al).all():
            continue
        gl = np.asarray(res["coupling"])
        acc = (spot_ci[gl.argmax(1)] == cidx).mean() if np.isfinite(gl).all() else np.nan
        comp = (~is_disp).astype(int)
        learn.append(dict(seed=seed, init=init, auc=roc_auc_score(comp, al),
                          r=stats.pointbiserialr(comp, al)[0],
                          a_c=al[comp == 1].mean(), a_d=al[comp == 0].mean(),
                          acc=acc, acc_global=sw[bv], acc_a0=sw[0.0]))
        print(f"  seed {seed:>2} init {init}: alpha compact {al[comp==1].mean():.3f} "
              f"dispersed {al[comp==0].mean():.3f}  AUC {roc_auc_score(comp, al):.3f}  "
              f"acc {acc:.4f} vs global {sw[bv]:.4f}")

sw = pd.DataFrame(sweep)
print("\n" + "=" * 70 + "\nQ15a: which alpha does each objective select?")
per = []
for s, sub in sw.groupby("seed"):
    q = sub.set_index("alpha")
    per.append(dict(seed=s, a_expr=float(q.loss_expr.idxmin()), a_cnv=float(q.loss_cnv.idxmin()),
                    a_acc=float(q.acc.idxmax())))
p = pd.DataFrame(per); n = len(p)
print(f"  argmin EXPRESSION loss : {', '.join(f'{v:.2f}' for v in p.a_expr)}")
print(f"  argmin CNV loss        : {', '.join(f'{v:.2f}' for v in p.a_cnv)}")
print(f"  argmax ACCURACY        : {', '.join(f'{v:.2f}' for v in p.a_acc)}")
print(f"\n  expression loss picks alpha > 0 in {int((p.a_expr > 0).sum())}/{n}; "
      f"matches accuracy optimum in {int((p.a_expr == p.a_acc).sum())}/{n}")
print(f"  CNV loss        picks alpha > 0 in {int((p.a_cnv > 0).sum())}/{n}; "
      f"matches accuracy optimum in {int((p.a_cnv == p.a_acc).sum())}/{n}")
print(f"  mean |selected - optimal|: expression {np.abs(p.a_expr - p.a_acc).mean():.3f}, "
      f"CNV {np.abs(p.a_cnv - p.a_acc).mean():.3f}")

if learn:
    L = pd.DataFrame(learn)
    print("\n" + "=" * 70 + "\nQ15b/c: per-clade alpha LEARNED against the CNV loss")
    print(f"{'init':>6}{'n':>4}{'mean AUC':>10}{'AUC>0.5':>9}{'a compact':>11}{'a disp':>9}"
          f"{'acc':>9}{'vs global':>11}{'vs a=0':>9}")
    for init, s2 in L.groupby("init"):
        print(f"{init:>6.1f}{len(s2):>4}{s2.auc.mean():>10.3f}{int((s2.auc>0.5).sum()):>9}"
              f"{s2.a_c.mean():>11.3f}{s2.a_d.mean():>9.3f}{s2.acc.mean():>9.4f}"
              f"{(s2.acc-s2.acc_global).mean():>+11.4f}{(s2.acc-s2.acc_a0).mean():>+9.4f}")
    t, pv = stats.ttest_1samp(L.auc, 0.5)
    print(f"\n  overall AUC {L.auc.mean():.3f} (above 0.5 in {int((L.auc>0.5).sum())}/{len(L)}), "
          f"t={t:+.2f}, p={pv:.4f}")
    print(f"  init spread of mean AUC: {L.groupby('init').auc.mean().max() - L.groupby('init').auc.mean().min():.3f}")
    print(f"  learned-alpha map vs best global alpha: {(L.acc - L.acc_global).mean():+.4f} "
          f"(better in {int(((L.acc - L.acc_global) > 0.005).sum())}/{len(L)})")
    L.to_csv(HERE / "q15_learn_cnv.csv", index=False)
sw.to_csv(HERE / "q15_cnv_sweep.csv", index=False)
