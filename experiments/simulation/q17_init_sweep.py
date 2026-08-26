"""Q17: how init-dependent is the CNV-anchored learning, and is it just slow convergence?

Q15 gave AUC 0.964 / 0.828 / 0.475 at inits 0.2 / 0.5 / 0.8, and at init 0.8 alpha barely
moved (compact 0.553 vs dispersed 0.568), which looks like a flat gradient rather than a wrong
optimum. Two things to separate:

  * a finer init sweep, to see whether "low init works" is a clean rule or a cliff
  * a long run from the worst init, to see whether high-init failure is non-convergence
    (fixable by iterating longer) or a genuine bad basin (not fixable)

If low init works reliably it is a defensible default -- alpha = 0 is the no-tree null, and
starting from the null and letting the anchor push alpha up is the conservative choice. But
that has to be stated as a rule, not discovered per dataset.
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

EPS = 0.02
INITS = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    cidx, spot_ci, is_disp = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    K = len(d["clades"]); comp = (~is_disp).astype(int)
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A)
    Y = jnp.asarray(np.asarray(d["cnv_spot"], np.float32))
    sig = jnp.asarray(np.asarray(d["cnv_clone"], np.float32) * 2.0)
    gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    for init in INITS:
        for iters in (300, 1200) if init >= 0.65 else (300,):
            res = S.learn_alpha_gamma_perclade_alpha_data(
                Cf, Y, C_T, C_S, a, b, jnp.asarray(A), sig, omega,
                eps=EPS, T_sinkhorn=50, J_alt=3, K_outer=iters, lr=0.05,
                alpha_init=jnp.full((K,), init), gamma0=gi, use_loss_ls=True)
            al = np.asarray(res["alpha"])
            if not np.isfinite(al).all():
                continue
            g = np.asarray(res["coupling"])
            rows.append(dict(seed=seed, init=init, iters=iters,
                             auc=roc_auc_score(comp, al),
                             a_c=al[comp == 1].mean(), a_d=al[comp == 0].mean(),
                             acc=(spot_ci[g.argmax(1)] == cidx).mean() if np.isfinite(g).all() else np.nan))

r = pd.DataFrame(rows)
base = r[r.iters == 300]
print(f"{'init':>6}{'n':>4}{'mean AUC':>10}{'sd':>7}{'>0.5':>7}{'a compact':>11}{'a disp':>9}{'ratio':>8}{'acc':>9}")
print("-" * 71)
for init, s in base.groupby("init"):
    print(f"{init:>6.2f}{len(s):>4}{s.auc.mean():>10.3f}{s.auc.std():>7.3f}"
          f"{int((s.auc > 0.5).sum()):>7}{s.a_c.mean():>11.3f}{s.a_d.mean():>9.3f}"
          f"{s.a_c.mean() / max(s.a_d.mean(), 1e-9):>8.1f}{s.acc.mean():>9.4f}")
lowc = base[base.init <= 0.20]
print(f"\ninit <= 0.20 : mean AUC {lowc.auc.mean():.3f}, above 0.5 in "
      f"{int((lowc.auc > 0.5).sum())}/{len(lowc)}")
hic = base[base.init >= 0.50]
print(f"init >= 0.50 : mean AUC {hic.auc.mean():.3f}, above 0.5 in "
      f"{int((hic.auc > 0.5).sum())}/{len(hic)}")
lng = r[(r.init >= 0.65) & (r.iters == 1200)]
sht = r[(r.init >= 0.65) & (r.iters == 300)]
if len(lng):
    print(f"\nhigh init, 300 iters : mean AUC {sht.auc.mean():.3f}")
    print(f"high init, 1200 iters: mean AUC {lng.auc.mean():.3f}  -> "
          + ("non-convergence, fixable by iterating longer" if lng.auc.mean() - sht.auc.mean() > 0.1
             else "a genuine bad basin, NOT fixable by iterating longer"))
r.to_csv(HERE / "q17_init_sweep.csv", index=False)
