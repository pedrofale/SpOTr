"""Q11 (= Q8 on the variance-controlled testbed): is the per-clade alpha LEARNABLE, or only available as an oracle?

Oracle headroom is necessary but not sufficient. The claim is that per-clade alpha learning
identifies parts of the tree that behave differently in space, which requires the parameter
to be recoverable from the data without the answer key.

There is a specific reason to expect this to work here where the phi experiment failed. In
the coarse-to-fine testbed, phi_k controlled WITHIN-clade placement while the deconvolution
loss scored BETWEEN-clade spot composition -- the parameter and the objective lived at
different levels, so phi's gradient was noise. Here alpha_k controls whether clade k is
pooled onto one territory, and the loss scores exactly whether clade k's cells land on the
right spots. Parameter and objective are at the same level.

Ground truth: alpha_k should be LOW for dispersed clades (pooling is wrong for them) and
HIGH for compact ones. Nothing supervised enters the fit; is_dispersed is used only to score
the result afterwards. Run from several inits, because the standard is that the claim must
not depend on initialisation.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")

SEEDS = [s for s in range(16)]
rows = []
for seed in SEEDS:
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    if not bool(np.load(f, allow_pickle=True)["in_band"][0]):
        continue
    d = np.load(f, allow_pickle=True)
    clades, is_disp = d["clades"], d["is_dispersed"]
    K = len(clades)
    cc = d["cell_clade"]
    cidx = np.asarray(cc)
    A = np.zeros((len(cc), K), dtype=np.float32); A[np.arange(len(cc)), cidx] = 1.0
    X_sc = d["X_sc"]
    sig = np.stack([X_sc[cidx == k].mean(0) for k in range(K)]).astype(np.float32) * 2.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    Y = jnp.asarray(np.asarray(d["X_st"], np.float32))
    g0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, 0.02, T_sinkhorn=50, J_alt=1, alpha=0.0)
    compact = (~is_disp).astype(int)
    for init in [0.2, 0.5, 0.8]:
        res = S.learn_alpha_gamma_perclade_alpha_data(
            Cf, Y, C_T, C_S, a, b, jnp.asarray(A), jnp.asarray(sig), jnp.asarray(A),
            eps=0.02, T_sinkhorn=50, J_alt=3, K_outer=300, lr=0.05,
            alpha_init=jnp.full((K,), init), gamma0=g0, use_loss_ls=True)
        al = np.asarray(res["alpha"]); loss = np.asarray(res["loss_hist"])
        if not np.isfinite(al).all():
            rows.append(dict(seed=seed, init=init, auc=np.nan, r=np.nan, p=np.nan,
                             a_c=np.nan, a_d=np.nan)); continue
        auc = roc_auc_score(compact, al)
        r_, p_ = stats.pointbiserialr(compact, al)
        rows.append(dict(seed=seed, init=init, auc=auc, r=r_, p=p_,
                         a_c=al[compact == 1].mean(), a_d=al[compact == 0].mean(),
                         loss0=float(loss[0]), loss1=float(loss[-1])))
        print(f"  seed {seed} init {init}: loss {loss[0]:.5f}->{loss[-1]:.5f}  "
              f"alpha compact {al[compact==1].mean():.3f} dispersed {al[compact==0].mean():.3f}  "
              f"AUC {auc:.3f}  r {r_:+.3f}")

r = pd.DataFrame(rows)
print("\n" + "=" * 72)
print(f"{'init':>6}{'n seeds':>9}{'mean AUC':>10}{'AUC>0.5':>9}{'mean r':>9}{'a compact':>11}{'a disp':>9}")
print("-" * 72)
for init, sub in r.groupby("init"):
    ok = sub.dropna(subset=["auc"])
    print(f"{init:>6.1f}{len(ok):>9}{ok.auc.mean():>10.3f}{int((ok.auc>0.5).sum()):>9}"
          f"{ok.r.mean():>+9.3f}{ok.a_c.mean():>11.3f}{ok.a_d.mean():>9.3f}")
ok = r.dropna(subset=["auc"])
print("-" * 72)
print(f"overall: mean AUC {ok.auc.mean():.3f}, above 0.5 in {int((ok.auc>0.5).sum())}/{len(ok)} fits, "
      f"mean r {ok.r.mean():+.3f}")
sp = ok.groupby("init").auc.mean()
print(f"init sensitivity: mean AUC by init = {', '.join(f'{i}:{v:.3f}' for i, v in sp.items())}  "
      f"(spread {sp.max()-sp.min():.3f})")
t, pv = stats.ttest_1samp(ok.auc, 0.5)
print(f"AUC vs 0.5: t={t:+.2f}, p={pv:.4f}  -> "
      + ("alpha_k IS learnable and identifies dispersed clades" if pv < 0.05 and ok.auc.mean() > 0.5
         else "not distinguishable from chance"))
r.to_csv(HERE / "q11_learn_ctrl.csv", index=False)
