"""Is alpha confounded with the entropic temperature?

Realised row entropy of the coupling varies ~5x across the alpha grid (0.62 -> 3.17) because
the GW term is far less row-discriminative than the feature term. So sweeping alpha at fixed
eps sweeps TWO things at once: the tree weight and the transport sharpness.

If that confound is what pins every selection criterion to a grid endpoint, then adapting
eps(alpha) to hold realised sharpness constant should change the picture. Specifically:

  H1: the accuracy-vs-alpha curve changes shape (part of "the tree helps" was temperature)
  H2: the expression-reconstruction loss stops being monotone in alpha
      (the 11/11 monotone result is the single strongest obstacle on record)

Both are falsifiable here. eps(alpha) is bisected so mean row entropy of the SOLVED coupling
matches its value at alpha=0.
"""
import sys, numpy as np, jax.numpy as jnp, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from petracer_common import Sec, GRID_G, T, J, TAU
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

SEC = sys.argv[1] if len(sys.argv) > 1 else "M3-4"
s = Sec(SEC)
Cf, C_T, C_S, a, b, omega, sF, sL = s._p


def solve_eps(alpha, eps):
    ak = jnp.full(s.K, float(alpha), jnp.float32)
    def build(g):
        return build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                            equalize="measured")
    g, _ = solve_perclade_semirelaxed(build, a, b, eps, T_sinkhorn=T, J_alt=J, tau=TAU)
    return np.asarray(g)


def row_entropy(g):
    P = np.asarray(g)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-30)
    return float(np.mean(-(P * np.log(P + 1e-30)).sum(1)))


def expr_loss(g):
    """The canonical criterion: reconstruct spot expression from the coupling."""
    X_sc = np.asarray(s.d["X_sc"], np.float64); X_st = np.asarray(s.d["X_st"], np.float64)
    P = np.asarray(g); w = P.sum(0)
    pred = (P.T @ X_sc) / np.maximum(w[:, None], 1e-12)
    return float(np.mean((pred - X_st) ** 2))


TARGET = row_entropy(solve_eps(0.0, 0.02))
print(f"[{SEC}] target row entropy (alpha=0, eps=0.02): {TARGET:.4f}\n")

rows = []
for v in GRID_G:
    g_fix = solve_eps(v, 0.02)
    # bisect eps so realised entropy matches TARGET
    lo, hi = 1e-4, 0.5
    best = None
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        gm = solve_eps(v, mid)
        e = row_entropy(gm)
        if not np.isfinite(e): break
        best = (mid, gm, e)
        if e > TARGET: hi = mid
        else: lo = mid
        if abs(e - TARGET) < 0.01: break
    eps_m, g_m, e_m = best
    rows.append(dict(alpha=v,
                     eps_fixed=0.02, ent_fixed=row_entropy(g_fix),
                     acc_fixed=s.score(g_fix), loss_fixed=expr_loss(g_fix),
                     eps_matched=eps_m, ent_matched=e_m,
                     acc_matched=s.score(g_m), loss_matched=expr_loss(g_m)))
    r = rows[-1]
    print(f"  alpha {v:<5} | FIXED eps=0.02 ent {r['ent_fixed']:.3f} acc {r['acc_fixed']:.4f} "
          f"loss {r['loss_fixed']:.5f} || MATCHED eps={eps_m:.5f} ent {e_m:.3f} "
          f"acc {r['acc_matched']:.4f} loss {r['loss_matched']:.5f}", flush=True)

t = pd.DataFrame(rows); t.to_csv(HERE / f"entropy_matched_{SEC}.csv", index=False)
print("\n" + "=" * 78)
def mono(x): 
    d = np.diff(np.asarray(x)); return bool((d > 0).all() or (d < 0).all())
print(f"H1  accuracy peak: fixed-eps alpha={t.loc[t.acc_fixed.idxmax(),'alpha']} "
      f"({t.acc_fixed.max():.4f})  |  entropy-matched alpha="
      f"{t.loc[t.acc_matched.idxmax(),'alpha']} ({t.acc_matched.max():.4f})")
print(f"    accuracy at alpha=0: fixed {t.acc_fixed.iloc[0]:.4f}  matched {t.acc_matched.iloc[0]:.4f}")
print(f"    tree gain: fixed {t.acc_fixed.max()-t.acc_fixed.iloc[0]:+.4f}  "
      f"matched {t.acc_matched.max()-t.acc_matched.iloc[0]:+.4f}")
print(f"\nH2  expression loss monotone in alpha?  fixed-eps: {mono(t.loss_fixed)}  "
      f"entropy-matched: {mono(t.loss_matched)}")
print(f"    argmin(loss): fixed alpha={t.loc[t.loss_fixed.idxmin(),'alpha']}  "
      f"matched alpha={t.loc[t.loss_matched.idxmin(),'alpha']}  "
      f"| argmax(acc) matched alpha={t.loc[t.acc_matched.idxmax(),'alpha']}")
print(f"\n    eps(alpha) needed: {[f'{e:.4f}' for e in t.eps_matched]}")
