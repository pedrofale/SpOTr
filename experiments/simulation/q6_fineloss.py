"""Q6: phi has no gradient because the loss cannot see what phi does.

Q5 fitted phi_k against deconvolution_loss_ls with CLADE-level signatures. That loss scores
how well each spot's expression is explained as a mixture of per-clade mean profiles -- a
purely COARSE quantity, "which clade sits in this spot". phi_k controls the WITHIN-clade
structural term, i.e. where a cell sits INSIDE its clade's region. The loss is blind to that
by construction, so phi's gradient is noise: AUC 0.278 / 0.639 / 0.500 across inits.

`learn_alpha_gamma_perclade_alpha_data` takes `clone_assignments` (what the loss
deconvolves into) and `omega` (what the per-clade parameters are indexed by) as SEPARATE
arguments. So the loss can be made fine-grained -- deconvolve into SUBCLADE proportions --
while phi stays per-clade. Then the objective can actually see within-clade placement.

Runs both granularities from the same three inits. Nothing supervised: subclade identity
comes from the tree, not from the answer key.
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

d = np.load(HERE / "sim_cf.npz", allow_pickle=True)
X_sc, X_st = d["X_sc"], d["X_st"]
cell_clade, cell_sub = d["cell_clade"], d["cell_sub"]
clades, is_tf = d["clades"], d["is_tree_fine"]
K = len(clades)


def onehot(lab):
    u = np.array(sorted(set(map(str, lab))))
    M = np.zeros((len(lab), len(u)), dtype=np.float32)
    for i, c in enumerate(map(str, lab)):
        M[i, list(u).index(c)] = 1.0
    return M, u


omega_np, _ = onehot(cell_clade)
A_coarse, u_coarse = omega_np, clades
A_fine, u_fine = onehot(cell_sub)
sig_coarse = np.stack([X_sc[np.array(list(map(str, cell_clade))) == str(c)].mean(0)
                       for c in u_coarse]).astype(np.float32) * 2.0
sig_fine = np.stack([X_sc[np.array(list(map(str, cell_sub))) == str(c)].mean(0)
                     for c in u_fine]).astype(np.float32) * 2.0
print(f"loss granularity: coarse = {len(u_coarse)} clades, fine = {len(u_fine)} subclades; "
      f"phi is per-clade ({K}) in both")

asc, ast = ad.AnnData(np.asarray(X_sc, np.float32)), ad.AnnData(np.asarray(X_st, np.float32))
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
Y = jnp.asarray(np.asarray(X_st, np.float32))
omega = jnp.asarray(omega_np)
Omega = jnp.asarray(omega_np @ omega_np.T)
g0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, 0.02, T_sinkhorn=50, J_alt=1, alpha=0.0)
truth = np.asarray(is_tf, dtype=bool)

rows = []
for gran, Aa, sg in [("coarse (clade)", A_coarse, sig_coarse), ("fine (subclade)", A_fine, sig_fine)]:
    for init in [0.2, 0.5, 0.8]:
        res = S.learn_alpha_gamma_perclade_alpha_data(
            Cf, Y, C_T, C_S, a, b, jnp.asarray(Aa), jnp.asarray(sg), omega,
            use_decomp=True, Omega=Omega, eps=0.02, T_sinkhorn=50, J_alt=3,
            K_outer=400, lr=0.05,
            alpha_init=jnp.full((K,), 0.7), phi_init=jnp.full((K,), init),
            gamma0=g0, use_loss_ls=True)
        phi = np.asarray(res["phi"]); loss = np.asarray(res["loss_hist"])
        if not np.isfinite(phi).all():
            # the fit diverged; record it rather than crashing -- a readout that only exists
            # when the optimiser happens not to blow up is not a readout either
            print(f"  {gran:<16} init {init}  DIVERGED (phi has NaN)")
            rows.append(dict(gran=gran, init=init, auc=np.nan, r=np.nan, p=np.nan,
                             phi_tf=np.nan, phi_ef=np.nan, spread=np.nan,
                             loss0=float(loss[0]), loss1=np.nan))
            continue
        auc = roc_auc_score(truth, phi)
        r, p = stats.pointbiserialr(truth.astype(int), phi)
        rows.append(dict(gran=gran, init=init, auc=auc, r=r, p=p,
                         phi_tf=phi[truth].mean(), phi_ef=phi[~truth].mean(),
                         spread=float(phi.max() - phi.min()),
                         loss0=float(loss[0]), loss1=float(loss[-1])))
        print(f"  {gran:<16} init {init}  loss {loss[0]:.5f}->{loss[-1]:.5f}  "
              f"phi tf {phi[truth].mean():.3f} ef {phi[~truth].mean():.3f}  "
              f"AUC {auc:.3f}  r {r:+.3f} (p {p:.3f})")

r = pd.DataFrame(rows)
print("\n" + "=" * 78)
print(f"{'loss granularity':<18}{'init':>6}{'AUC':>8}{'r':>9}{'p':>8}{'phi tf':>9}{'phi ef':>9}")
print("-" * 78)
for _, x in r.iterrows():
    f = lambda v, w, d=3: (f"{v:>{w}.{d}f}" if np.isfinite(v) else f"{'--':>{w}}")
    print(f"{x.gran:<18}{x.init:>6.1f}{f(x.auc,8)}{f(x.r,9)}{f(x.p,8)}{f(x.phi_tf,9)}{f(x.phi_ef,9)}")
print("-" * 78)
for gran, sub in r.groupby("gran", sort=False):
    ok = sub.auc.dropna()
    nd = int(sub.auc.isna().sum())
    if len(ok) == 0:
        print(f"{gran:<18} all fits diverged"); continue
    print(f"{gran:<18} AUC range {ok.min():.3f} - {ok.max():.3f}   "
          f"mean r {sub.r.mean():+.3f}   {nd} diverged   "
          + ("RECOVERS the split, init-independent" if ok.min() > 0.65 and nd == 0
             else "no reliable signal"))
r.to_csv(HERE / "q6_fineloss.csv", index=False)
