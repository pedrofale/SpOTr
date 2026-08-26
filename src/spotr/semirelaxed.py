"""Semi-relaxed optimal transport for SpOTr.

WHY
---
Two separate failures in this project trace to the same place: the hard marginal
constraints of balanced OT.

1. **Composition is pinned to the reference.** With a uniform source marginal, each clone's
   total transported mass is fixed at its share of the reference cells. Measured on
   eTRACER: corr(output composition, reference composition) = 1.000, identical to ~1e-7 at
   every eps from 0.5 down to 0.005. The two modalities differ by TVD 0.538 at natural
   composition (clone C4 is 7.5% of dissociated cells but 40.8% of tissue), so ~54% of the
   transported mass lands on the wrong clone before any cost is computed.

2. **Per-clade parameters cannot be read out.** Because every column of gamma must sum to
   b_j, mass given to one clade at a spot is taken from another, so perturbing one clade's
   parameter necessarily moves all of them and the unperturbed configuration wins by
   default. On simulated data with a known graded per-clade lambda, recovered phi
   correlated with truth at -0.098 (p = 0.59); neutralising the per-clade equalizer
   (`perclade_v2`) only moved it to +0.168 (p = 0.40), and the readout was proved
   irrelevant (column sums are constant, so normalising cannot change a correlation).

The two constraints are not equally justified:

- **Target** `gamma.T @ 1 = b`: a spot really is fully composed of cells. Physically true --
  keep it hard.
- **Source** `gamma @ 1 = a`: a cell's share of a dissociated suspension reflects
  dissociation efficiency, viability and capture, not its abundance in tissue. Requiring it
  to ship exactly 1/n is an artefact of the formulation -- relax it.

That is *semi-relaxed* OT: relax one marginal, keep the other.

HOW
---
Replace the exact source update with a damped one (Chizat et al. 2018 scaling iterations).
With a KL penalty of weight `tau` on the source marginal:

    rho = tau / (tau + eps)
    u  <-  (a / (K v)) ** rho          # damped; rho = 1 recovers balanced
    v  <-   b / (K.T u)                # exact, target stays hard

`tau -> inf` gives rho -> 1 and reproduces balanced Sinkhorn exactly. `tau -> 0` gives
rho -> 0, leaving the source completely free. Intermediate values trade off how far the
fitted clone abundances may drift from the reference.

Nothing in `spotr.spotr` is modified; import from here to opt in.
"""

import jax
import jax.numpy as jnp

from .spotr import (
    exp_smooth_clipped, build_fgw_cost, compute_Lgw, dc, mad_abs,
)


def sinkhorn_semirelaxed(C, a, b, eps, T, tau=1.0, uv0=None, tiny=1e-300, M=40.0):
    """Sinkhorn with a KL-relaxed SOURCE marginal and an exact target marginal.

    tau: weight of the KL penalty pulling the source marginal towards `a`.
         Large tau -> balanced; small tau -> source abundances free to be fitted.
    """
    K = exp_smooth_clipped(-C / eps, M=M) + tiny
    rho = tau / (tau + eps)                      # damping exponent on the source update
    u = jnp.ones_like(a) if uv0 is None else jnp.maximum(uv0[0], tiny)
    v = jnp.ones_like(b) if uv0 is None else jnp.maximum(uv0[1], tiny)
    supp_a, supp_b = (a > 0), (b > 0)

    def body(_, carry):
        u, v = carry
        ratio = a / jnp.maximum(K @ v, tiny)
        u = jnp.where(supp_a, jnp.power(jnp.maximum(ratio, tiny), rho), 0.0)
        v = jnp.where(supp_b, b / jnp.maximum(K.T @ u, tiny), 0.0)
        return (u, v)

    u, v = jax.lax.fori_loop(0, T, body, (u, v))
    gamma = u[:, None] * K * v[None, :]
    return gamma, (u, v)


def sinkhorn_fgw_semirelaxed(C_feature, C_tree, C_space, a, b, eps,
                             T_sinkhorn=50, J_alt=3, alpha=0.5,
                             tau=1.0, gamma0=None, uv0=None):
    """`sinkhorn_fgw` with the semi-relaxed solver. Same signature plus `tau`."""
    if gamma0 is None:
        gamma0 = a[:, None] * b[None, :]
    if uv0 is None:
        uv0 = (jnp.ones_like(a), jnp.ones_like(b))

    F_c = dc(C_feature, a, b)
    L_c = dc(compute_Lgw(C_tree, C_space, a, b, gamma0), a, b)
    sF_ref, sL_ref = mad_abs(F_c), mad_abs(L_c)

    def body(carry, _):
        gamma, uv = carry
        C = build_fgw_cost(alpha, C_feature, C_tree, C_space, a, b, gamma, sF_ref, sL_ref)
        gamma, uv = sinkhorn_semirelaxed(C, a, b, eps, T_sinkhorn, tau=tau, uv0=uv)
        return (gamma, uv), None

    (gamma, uv), _ = jax.lax.scan(body, (gamma0, uv0), None, length=J_alt)
    return gamma, uv


def solve_perclade_semirelaxed(build_cost_fn, a, b, eps, T_sinkhorn=50, J_alt=3,
                               tau=1.0, gamma0=None):
    """Alternating solve for an arbitrary per-clade cost builder, semi-relaxed.

    `build_cost_fn(gamma) -> C` lets this be used with `perclade_v2`.
    """
    gamma = (a[:, None] * b[None, :]) if gamma0 is None else gamma0
    uv = None
    for _ in range(J_alt):
        C = build_cost_fn(gamma)
        gamma, uv = sinkhorn_semirelaxed(C, a, b, eps, T_sinkhorn, tau=tau, uv0=uv)
    return gamma, uv
