"""Q13: does a HELD-OUT-GENE objective select the right alpha?

Q10 showed deconvolution_loss_ls is monotone increasing in alpha in 11/11 seeds, so its
argmin is pinned at 0 and alpha cannot be learned from it. The reason is structural: the loss
scores reconstruction of spot expression from clade profiles, which is precisely what the
expression term of the OT cost optimises. Structure can only pull the coupling off that
optimum, so the loss is really measuring "how much did you use expression".

The standard fix for an objective that rewards fitting the thing you fitted on is to score it
on data you did not fit. Genes are split in half: the OT feature cost is built from FIT genes
only, and the deconvolution loss is evaluated on HELD-OUT genes. An expression-only coupling
should overfit the fit genes; if the tree genuinely improves the mapping, the coupling it
produces should transfer better, giving the held-out loss an interior minimum near the
accuracy optimum.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    rg = np.random.default_rng(1000 + seed)
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    K = len(d["clades"])
    X_sc, X_st = d["X_sc"], d["X_st"]
    G = X_sc.shape[1]
    perm = rg.permutation(G)
    fit_g, ev_g = perm[: G // 2], perm[G // 2:]
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0

    # feature cost from FIT genes only
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc[:, fit_g]), ad.AnnData(X_st[:, fit_g]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    Y_fit = jnp.asarray(np.asarray(X_st[:, fit_g], np.float32))
    Y_ev = jnp.asarray(np.asarray(X_st[:, ev_g], np.float32))
    sig_fit = jnp.asarray(np.stack([X_sc[cidx == k][:, fit_g].mean(0) for k in range(K)]).astype(np.float32) * 2.0)
    sig_ev = jnp.asarray(np.stack([X_sc[cidx == k][:, ev_g].mean(0) for k in range(K)]).astype(np.float32) * 2.0)

    for av in GRID:
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        gn = np.asarray(g)
        rows.append(dict(seed=seed, alpha=av,
                         loss_fit=float(S.deconvolution_loss_ls(Y_fit, g, jnp.asarray(A), sig_fit)),
                         loss_ev=float(S.deconvolution_loss_ls(Y_ev, g, jnp.asarray(A), sig_ev)),
                         acc=(spot_ci[gn.argmax(1)] == cidx).mean()))

r = pd.DataFrame(rows)
print(f"{'seed':>5} " + "".join(f"{'a=' + str(v):>22}" for v in GRID))
print(f"{'':>5} " + "".join(f"{'fit / heldout / acc':>22}" for _ in GRID))
print("-" * (5 + 22 * len(GRID)))
per = []
for s, sub in r.groupby("seed"):
    q = sub.set_index("alpha")
    print(f"{s:>5} " + "".join(f"{q.loss_fit[v]:>7.4f}/{q.loss_ev[v]:.4f}/{q.acc[v]:.3f}" for v in GRID))
    per.append(dict(seed=s, a_fit=float(q.loss_fit.idxmin()), a_ev=float(q.loss_ev.idxmin()),
                    a_acc=float(q.acc.idxmax())))
p = pd.DataFrame(per)
print("-" * (5 + 22 * len(GRID)))
print(f"\nalpha minimising FIT loss      : {', '.join(f'{v:.2f}' for v in p.a_fit)}")
print(f"alpha minimising HELD-OUT loss : {', '.join(f'{v:.2f}' for v in p.a_ev)}")
print(f"alpha maximising ACCURACY      : {', '.join(f'{v:.2f}' for v in p.a_acc)}")
n = len(p)
print(f"\nfit-loss picks the accuracy-optimal alpha in      {int((p.a_fit == p.a_acc).sum())}/{n}")
print(f"held-out loss picks the accuracy-optimal alpha in {int((p.a_ev == p.a_acc).sum())}/{n}")
print(f"held-out loss picks alpha > 0 in                  {int((p.a_ev > 0).sum())}/{n}")
print(f"mean |chosen - optimal| : fit {np.abs(p.a_fit - p.a_acc).mean():.3f}, "
      f"held-out {np.abs(p.a_ev - p.a_acc).mean():.3f}")
r.to_csv(HERE / "q13_heldout.csv", index=False)
