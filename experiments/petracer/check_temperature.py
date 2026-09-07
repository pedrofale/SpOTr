"""Does the per-clade cost path leave the entropic temperature free to vary with alpha?

build_fgw_cost (global path) ends with s = Z/den, holding median|C| ~ constant across alpha.
build_perclade_alpha_cost_v2 equalizes ACROSS CLADES only: target = median_k(m_k), so for a
UNIFORM alpha vector every m_k is equal, s_k == 1, and the global magnitude of C0 is free.

If median|C0| varies with alpha, then eps=0.02 means something different at each grid point
and the alpha sweep is partly measuring transport sharpness, not the tree weight.
"""
import numpy as np, jax, jax.numpy as jnp
from petracer_common import Sec, GRID_G, EPS
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

for name in ["M3-1", "M3-4"]:
    s = Sec(name)
    Cf, C_T, C_S, a, b, omega, sF, sL = s._p
    g0 = a[:, None] * b[None, :]
    print(f"\n=== {name} ===")
    print(f"{'alpha':>6} {'median|C0| perclade':>21} {'s_k (min/max)':>16} "
          f"{'median|C| global path':>23} {'C/eps perclade':>16} {'row entropy':>12}")
    for v in GRID_G:
        ak = jnp.full(s.K, float(v), jnp.float32)
        C_pc = build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g0, omega, sF, sL,
                                            equalize="measured")
        C0 = build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g0, omega, sF, sL,
                                          equalize="none")
        sk = np.asarray(jnp.median(jnp.abs(C_pc)) / jnp.maximum(jnp.median(jnp.abs(C0)), 1e-9))
        C_gl = S.build_fgw_cost(float(v), Cf, C_T, C_S, a, b, g0, sF, sL)
        m_pc = float(jnp.median(jnp.abs(C_pc)))
        m_gl = float(jnp.median(jnp.abs(C_gl)))
        # transport sharpness actually realised
        P = np.asarray(jax.nn.softmax(-np.asarray(C_pc) / EPS, axis=1)) if False else None
        K_ = np.exp(-(np.asarray(C_pc) - np.asarray(C_pc).min(1, keepdims=True)) / EPS)
        K_ = K_ / K_.sum(1, keepdims=True)
        ent = float(np.mean(-(K_ * np.log(K_ + 1e-30)).sum(1)))
        print(f"{v:>6} {m_pc:>21.4f} {float(sk):>16.3f} {m_gl:>23.4f} "
              f"{m_pc/EPS:>16.1f} {ent:>12.3f}")

    ms = [float(jnp.median(jnp.abs(build_perclade_alpha_cost_v2(
        jnp.full(s.K, float(v), jnp.float32), Cf, C_T, C_S, a, b, g0, omega, sF, sL,
        equalize="measured")))) for v in GRID_G]
    gs = [float(jnp.median(jnp.abs(S.build_fgw_cost(float(v), Cf, C_T, C_S, a, b, g0, sF, sL))))
          for v in GRID_G]
    print(f"  per-clade path: median|C| ranges {min(ms):.4f}-{max(ms):.4f}, "
          f"max/min = {max(ms)/min(ms):.2f}x")
    print(f"  global    path: median|C| ranges {min(gs):.4f}-{max(gs):.4f}, "
          f"max/min = {max(gs)/min(gs):.2f}x")
