"""Q10: is the training objective even pointing at the right thing?

Learned alpha fell from every init (0.2->0.04, 0.5->0.16, 0.8->0.39) with compact and
dispersed clades ending up indistinguishable. So the gradient exists but points uniformly
"use less tree". A plausible reason: deconvolution_loss_ls scores how well each spot's
expression is reconstructed from clade proportions. The expression term optimises exactly
that; the structural term can only interfere with it. If so, the loss is a proxy for
expression fit rather than for mapping accuracy, and no amount of tuning will make alpha
learnable from it.

Direct test: sweep alpha, and plot the training loss against the true mapping accuracy. If
the loss is minimised where accuracy is minimised, the objective is misaligned and that --
not the parameterisation -- is what blocks claim (b).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats

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
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    K = len(d["clades"])
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
    X_sc = d["X_sc"]
    sig = jnp.asarray(np.stack([X_sc[cidx == k].mean(0) for k in range(K)]).astype(np.float32) * 2.0)
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    Y = jnp.asarray(np.asarray(d["X_st"], np.float32))
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    for av in GRID:
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        gn = np.asarray(g)
        loss = float(S.deconvolution_loss_ls(Y, g, jnp.asarray(A), sig))
        rows.append(dict(seed=seed, alpha=av, loss=loss,
                         acc=(spot_ci[gn.argmax(1)] == cidx).mean()))

r = pd.DataFrame(rows)
print(f"{'seed':>5} " + "".join(f"{'a=' + str(v):>16}" for v in GRID))
print(f"{'':>5} " + "".join(f"{'loss / acc':>16}" for _ in GRID))
print("-" * (5 + 16 * len(GRID)))
per = []
for s, sub in r.groupby("seed"):
    sub = sub.set_index("alpha")
    print(f"{s:>5} " + "".join(f"{sub.loss[v]:>8.4f}/{sub.acc[v]:.3f}" for v in GRID))
    per.append(dict(seed=s, argmin_loss=float(sub.loss.idxmin()), argmax_acc=float(sub.acc.idxmax()),
                    r=float(np.corrcoef(sub.loss, sub.acc)[0, 1])))
p = pd.DataFrame(per)
print("-" * (5 + 16 * len(GRID)))
print(f"\nalpha minimising the LOSS   : {', '.join(f'{v:.2f}' for v in p.argmin_loss)}")
print(f"alpha maximising ACCURACY   : {', '.join(f'{v:.2f}' for v in p.argmax_acc)}")
print(f"corr(loss, accuracy) per seed: {', '.join(f'{v:+.2f}' for v in p.r)}")
print(f"\nmean corr(loss, accuracy) = {p.r.mean():+.3f}")
agree = int((np.abs(p.argmin_loss - p.argmax_acc) < 1e-9).sum())
print(f"loss-optimal alpha equals accuracy-optimal alpha in {agree}/{len(p)} seeds")
# Correlation is the WRONG statistic here and reports the opposite of the truth. Past the
# accuracy peak both loss and accuracy fall together, so they anti-correlate (-0.61) while the
# objective's argmin is still wrong in every seed. What matters is whether the loss SELECTS the
# right alpha, i.e. argmin agreement -- and whether the loss is monotone, in which case it can
# only ever select the endpoint.
mono = int(sum(1 for _, sub in r.groupby("seed")
               if (sub.sort_values("alpha").loss.diff().dropna() > 0).all()))
print(f"\nloss is MONOTONE increasing in alpha in {mono}/{len(p)} seeds "
      f"-> its argmin is pinned at alpha=0 regardless of accuracy")
print(f"argmin(loss) == argmax(accuracy) in {agree}/{len(p)} seeds")
print("\n-> " + ("objective SELECTS the right alpha" if agree > len(p) / 2 else
                 "objective is MISALIGNED: it is minimised where the map is NOT best, so "
                 "alpha cannot be learned from it"))
r.to_csv(HERE / "q10_loss_alignment.csv", index=False)
