"""A label-free criterion suggested by the (alpha, eps) grid.

From alpha_eps_grid_M3-4.csv:
  alpha=0.1 accuracy across eps: 0.190 0.218 0.228 0.218 0.207 0.199 0.210 0.219  (range 0.038)
  alpha=0.6 accuracy across eps: 0.148 0.155 0.187 0.289 0.318 0.250 0.257 0.161  (range 0.170)
alpha=0.1 is ROBUST to eps; alpha=0.6 is a gamble that pays only if eps is right.

Stability across eps is measurable WITHOUT LABELS: solve at several eps and ask how much the
hard assignment moves. If accuracy-stability and coupling-stability agree, this is a genuine
label-free selector -- one that would pick a safe alpha rather than the oracle-best one.

Reports, per alpha: coupling stability across eps (label-free) vs accuracy mean/range (oracle).
"""
import sys, numpy as np, jax.numpy as jnp, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from petracer_common import Sec, T, J, TAU
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed
from scipy.stats import pearsonr, spearmanr

SEC = sys.argv[1] if len(sys.argv) > 1 else "M3-4"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
EPSES = [0.01, 0.015, 0.02, 0.03, 0.05]
s = Sec(SEC)
Cf, C_T, C_S, a, b, omega, sF, sL = s._p
xy = np.asarray(s.d["xy"], float)
Dsp = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
Drand = float(Dsp.mean())

def solve(alpha, eps):
    ak = jnp.full(s.K, float(alpha), jnp.float32)
    def build(g):
        return build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                            equalize="measured")
    g, _ = solve_perclade_semirelaxed(build, a, b, eps, T_sinkhorn=T, J_alt=J, tau=TAU)
    return np.asarray(g)

rows = []
for v in ALPHAS:
    Gs = [solve(v, e) for e in EPSES]
    ok = [g for g in Gs if np.isfinite(g).all()]
    if len(ok) < 2: continue
    J_ = [np.asarray(g).argmax(1) for g in ok]
    accs = [s.score(g) for g in ok]
    # label-free stability: mean pairwise spatial displacement of the hard assignment
    disp, agree = [], []
    for p in range(len(J_)):
        for q in range(p + 1, len(J_)):
            disp.append(Dsp[J_[p], J_[q]].mean() / Drand)
            agree.append(float((J_[p] == J_[q]).mean()))
    rows.append(dict(alpha=v, stab_disp=float(np.mean(disp)), stab_agree=float(np.mean(agree)),
                     acc_mean=float(np.mean(accs)), acc_max=float(np.max(accs)),
                     acc_range=float(np.max(accs) - np.min(accs))))
    r = rows[-1]
    print(f"  alpha {v:<5} | LABEL-FREE displacement {r['stab_disp']:.4f} agreement "
          f"{r['stab_agree']:.4f} || ORACLE acc mean {r['acc_mean']:.4f} max {r['acc_max']:.4f} "
          f"range {r['acc_range']:.4f}", flush=True)

t = pd.DataFrame(rows); t.to_csv(HERE / f"eps_stability_{SEC}.csv", index=False)
print("\n" + "=" * 78)
sel_d = t.loc[t.stab_disp.idxmin()]; sel_a = t.loc[t.stab_agree.idxmax()]
print(f"most stable alpha by DISPLACEMENT: {sel_d.alpha} -> acc mean {sel_d.acc_mean:.4f}")
print(f"most stable alpha by AGREEMENT:    {sel_a.alpha} -> acc mean {sel_a.acc_mean:.4f}")
print(f"oracle-best alpha by mean acc:     {t.loc[t.acc_mean.idxmax(),'alpha']} "
      f"-> {t.acc_mean.max():.4f}")
print(f"alpha=0 baseline:                  {t[t.alpha==0].acc_mean.iloc[0]:.4f}")
print(f"\ncorr(stability, accuracy) across alpha:")
for c, sign in [("stab_disp", -1), ("stab_agree", +1)]:
    r_, p_ = pearsonr(t[c], t.acc_mean); rho, pr = spearmanr(t[c], t.acc_mean)
    print(f"  {c:<11} pearson r={r_:+.3f} (p={p_:.3f})  spearman rho={rho:+.3f} (p={pr:.3f})"
          f"   [want sign {'-' if sign<0 else '+'}]")
print(f"\nexcluding alpha=0 (which is trivially stable and useless):")
u = t[t.alpha > 0]
for c in ["stab_disp", "stab_agree"]:
    rho, pr = spearmanr(u[c], u.acc_mean)
    print(f"  {c:<11} spearman rho={rho:+.3f} (p={pr:.3f})")
