"""Where is the Sinkhorn solve actually converged? Entropy must be MONOTONE in eps there."""
import sys, numpy as np, jax.numpy as jnp
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from petracer_common import Sec, TAU
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

s = Sec("M3-4")
Cf, C_T, C_S, a, b, omega, sF, sL = s._p
b_np = np.asarray(b)

def run(alpha, eps, T, J=3):
    ak = jnp.full(s.K, float(alpha), jnp.float32)
    def build(g):
        return build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                            equalize="measured")
    g, _ = solve_perclade_semirelaxed(build, a, b, eps, T_sinkhorn=T, J_alt=J, tau=TAU)
    G = np.asarray(g)
    if not np.isfinite(G).all(): return None
    P = G / np.maximum(G.sum(1, keepdims=True), 1e-30)
    ent = float(np.mean(-(P * np.log(P + 1e-30)).sum(1)))
    viol = float(np.abs(G.sum(0) - b_np).sum())      # spot marginal should be exact
    return ent, viol, s.score(G)

EPS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
for T in [50, 400]:
    print(f"\n=== T_sinkhorn = {T} ===")
    print(f"{'eps':>7} | {'a=0.0 ent':>10} {'viol':>9} {'acc':>7} | {'a=0.6 ent':>10} {'viol':>9} {'acc':>7}")
    for e in EPS:
        r0, r6 = run(0.0, e, T), run(0.6, e, T)
        f = lambda r: (f"{r[0]:>10.3f} {r[1]:>9.2e} {r[2]:>7.4f}" if r else f"{'NaN':>10} {'-':>9} {'-':>7}")
        print(f"{e:>7} | {f(r0)} | {f(r6)}")
