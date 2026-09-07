"""The (alpha, eps) landscape: is alpha identifiable only jointly with eps?"""
import sys, numpy as np, jax.numpy as jnp, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from petracer_common import Sec, T, J, TAU
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

SEC = sys.argv[1] if len(sys.argv) > 1 else "M3-4"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
EPSES = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]

s = Sec(SEC)
Cf, C_T, C_S, a, b, omega, sF, sL = s._p
X_sc = np.asarray(s.d["X_sc"], np.float64); X_st = np.asarray(s.d["X_st"], np.float64)
A_np = np.asarray(s.d["A"], np.float64); b_np = np.asarray(b)

def solve(alpha, eps):
    ak = jnp.full(s.K, float(alpha), jnp.float32)
    def build(g):
        return build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                            equalize="measured")
    g, _ = solve_perclade_semirelaxed(build, a, b, eps, T_sinkhorn=T, J_alt=J, tau=TAU)
    return np.asarray(g)

rows = []
for e in EPSES:
    for v in ALPHAS:
        G = solve(v, e)
        if not np.isfinite(G).all():
            rows.append(dict(eps=e, alpha=v, acc=np.nan, expr_loss=np.nan, comp_loss=np.nan,
                             ent=np.nan, viol=np.nan)); continue
        w = G.sum(0)
        pred = (G.T @ X_sc) / np.maximum(w[:, None], 1e-12)
        expr = float(np.mean((pred - X_st) ** 2))
        p = G.T @ A_np; p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        comp = float(np.mean((s.COMP - p) ** 2))
        mass = G.sum(1); mass = mass / mass.sum()
        P = G / np.maximum(G.sum(1, keepdims=True), 1e-30)
        ent = float((mass * -(P * np.log(P + 1e-30)).sum(1)).sum())
        rows.append(dict(eps=e, alpha=v, acc=s.score(G), expr_loss=expr, comp_loss=comp,
                         ent=ent, viol=float(np.abs(G.sum(0) - b_np).sum())))
    print(f"  eps={e:<7} acc: " + " ".join(f"{r['acc']:.3f}" for r in rows[-len(ALPHAS):]), flush=True)

t = pd.DataFrame(rows); t.to_csv(HERE / f"alpha_eps_grid_{SEC}.csv", index=False)
piv = t.pivot(index="eps", columns="alpha", values="acc")
print(f"\n=== {SEC}: soft accuracy over (eps, alpha); random floor {s.floor:.4f} ===")
print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
best = t.loc[t.acc.idxmax()]
print(f"\nglobal optimum: alpha={best.alpha}, eps={best.eps}, acc={best.acc:.4f}")
print("best alpha AT EACH eps:  " + "  ".join(
    f"eps{e}:a={t[t.eps==e].loc[t[t.eps==e].acc.idxmax(),'alpha']}" for e in EPSES))
print("best eps AT EACH alpha:  " + "  ".join(
    f"a{v}:eps={t[t.alpha==v].loc[t[t.alpha==v].acc.idxmax(),'eps']}" for v in ALPHAS))
print(f"\nalpha=0 accuracy across eps: {[f'{x:.4f}' for x in piv[0.0].values]} "
      f"(range {piv[0.0].max()-piv[0.0].min():.4f})")
print("tree gain (max over alpha, minus alpha=0) at each eps: " +
      " ".join(f"{piv.loc[e].max()-piv.loc[e,0.0]:+.3f}" for e in EPSES))
print("\ndo the label-free criteria track the ridge?")
a0 = float(piv.loc[0.02, 0.0])
for crit in ["expr_loss", "comp_loss"]:
    sel = t.loc[t[crit].idxmin()]
    print(f"  {crit:<10} selects alpha={sel.alpha}, eps={sel.eps} -> acc {sel.acc:.4f} "
          f"(optimum {best.acc:.4f}, alpha=0 at eps=0.02 {a0:.4f})")
for crit in ["expr_loss", "comp_loss"]:
    hits = sum(1 for e in EPSES
               if t[t.eps==e].loc[t[t.eps==e][crit].idxmin(),"alpha"]
                  == t[t.eps==e].loc[t[t.eps==e].acc.idxmax(),"alpha"])
    print(f"  {crit:<10} picks the accuracy-optimal alpha in {hits}/{len(EPSES)} eps slices")
