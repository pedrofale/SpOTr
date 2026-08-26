"""Q12: is the mechanism what I claim -- dispersed clades CAP the global alpha?

Q9 showed the tree helps compact clades (+0.084) while harm to dispersed clades at the CHOSEN
global alpha is ~0. Those are only consistent if the threat of harming dispersed clades is
what stops the global alpha rising further, and per-clade weighting escapes that cap.

Prediction, if true:
  1. compact accuracy keeps rising past the best GLOBAL alpha
  2. dispersed accuracy falls over that same range
  3. the oracle's compact-clade alpha exceeds the best global alpha
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    cidx, spot_ci, is_disp = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    K = len(d["clades"]); disp = is_disp[cidx]
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    for av in GRID:
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        gn = np.asarray(g); ok = spot_ci[gn.argmax(1)] == cidx
        rows.append(dict(seed=seed, alpha=av, all=ok.mean(),
                         compact=ok[~disp].mean(), dispersed=ok[disp].mean()))
r = pd.DataFrame(rows)
m = r.groupby("alpha")[["all", "compact", "dispersed"]].agg(["mean", "std"])
print(f"{'alpha':>6} {'ALL':>16} {'COMPACT':>16} {'DISPERSED':>16}   (mean +- sd over seeds)")
print("-" * 60)
for av in GRID:
    x = m.loc[av]
    print(f"{av:>6.2f} {x[('all','mean')]:>9.4f}+-{x[('all','std')]:.3f} "
          f"{x[('compact','mean')]:>9.4f}+-{x[('compact','std')]:.3f} "
          f"{x[('dispersed','mean')]:>9.4f}+-{x[('dispersed','std')]:.3f}")
bg = r.groupby("seed").apply(lambda s: s.alpha[s["all"].idxmax()])
bc = r.groupby("seed").apply(lambda s: s.alpha[s.compact.idxmax()])
bd = r.groupby("seed").apply(lambda s: s.alpha[s.dispersed.idxmax()])
print(f"\nbest alpha for OVERALL   : {', '.join(f'{v:.2f}' for v in bg)}   mean {bg.mean():.3f}")
print(f"best alpha for COMPACT   : {', '.join(f'{v:.2f}' for v in bc)}   mean {bc.mean():.3f}")
print(f"best alpha for DISPERSED : {', '.join(f'{v:.2f}' for v in bd)}   mean {bd.mean():.3f}")
print(f"\ncompact wants a HIGHER alpha than the global optimum in "
      f"{int((bc > bg).sum())}/{len(bg)} seeds")
print(f"dispersed wants a LOWER alpha than the global optimum in "
      f"{int((bd < bg).sum())}/{len(bg)} seeds")
r.to_csv(HERE / "q12_mechanism.csv", index=False)
