"""Per-clade cost construction, v2 -- fixes the equalizer that made per-clade parameters
unreadable.

WHY THIS EXISTS
---------------
`spotr.build_perclade_alpha_cost` ends with a per-clade rescaling of the cost:

    alpha_ref_k = stop_gradient(alpha_k)
    den_k       = (1 - alpha_ref_k) * mF_k + alpha_ref_k * mL_k
    s_k         = (mF_k + mL_k) / den_k
    C           = C0 * (omega @ s_k)[:, None]

The intent is sound: as alpha varies, the blended cost `C0` changes magnitude, and under a
shared entropic `eps` that silently changes how sharp the transport is. The equalizer
renormalises so `eps` means the same thing throughout.

The problem is that `s_k` is a FUNCTION OF alpha_k (and, on the decomposed path, of phi_k
through `mL_k`). Multiplying one clade's rows of the cost by a different factor is, in
entropic OT, equivalent to giving that clade its own temperature. So a clade whose alpha or
phi differs from the rest is transported at a different sharpness for a purely arithmetic
reason -- nothing to do with whether its tree is spatially informative.

Consequences observed:

* Perturbing a single clade away from a uniform vector is penalised in itself. In a sweep
  over one clade's phi, 15 of 23 clades scored best at exactly the baseline value, with a
  clean spike there (e.g. -0.012, -0.012, +0.058, -0.011, -0.011).
* Sweeping globally avoids that but then every clade moves together, so a clade's score
  curve reflects the whole system's response rather than its own preference.

Between the two there was no valid way to read a per-clade preference, which is why the
per-clade interpretability test could neither pass nor fail honestly: on simulated data with
a KNOWN graded per-clade ground truth (corr(lambda, induced concordance) = +0.807) the
recovered phi correlated with truth at -0.098 (p = 0.59).

THE FIX
-------
Equalise on the MEASURED magnitude of the blended cost rather than on a magnitude PREDICTED
from alpha:

    m_k    = per-clade median |C0|          (measured from the actual cost)
    target = median over clades of m_k
    s_k    = target / m_k

This achieves the original goal directly -- every clade ends up with comparable |C|, so a
shared `eps` is meaningful everywhere -- while `s_k` no longer depends on alpha_k or phi_k.
A clade that deviates is no longer rescaled differently *because* it deviates, so its score
curve reflects its own structure and single-clade perturbation becomes interpretable.

`equalize` selects the behaviour:
    "measured" (default) -- the fix above
    "alpha"              -- reproduces the original, for A/B comparison
    "none"               -- no per-clade rescaling at all

Nothing in `spotr.spotr` is modified; import from here to opt in.
"""

import jax
import jax.numpy as jnp

from .spotr import compute_Lgw, compute_Lcladegw, _dc_rect


def build_perclade_alpha_cost_v2(
    alpha_k,
    C_feature, C_tree, C_space,
    a, b, gamma,
    omega,
    sF_ref, sL_ref,
    *,
    use_decomp=False,
    Omega=None,
    phi_k=None,
    equalize="measured",
    s_min=1e-6, s_max=1e6,
    rel_floor=5e-2,
    s_lo=5e-1, s_hi=2.0,
):
    """Same cost as `build_perclade_alpha_cost`, with an alpha-independent equalizer.

    Returns the [n, m] cost matrix.
    """
    # 1) feature part
    F_c = _dc_rect(C_feature, a, b)

    sF = jax.lax.stop_gradient(jnp.clip(sF_ref, s_min, s_max))
    sL = jax.lax.stop_gradient(jnp.clip(sL_ref, s_min, s_max))
    F_hat = F_c / sF

    # 2) structure part
    if not use_decomp:
        L = compute_Lgw(C_tree, C_space, a, b, gamma)
        L_hat = _dc_rect(L, a, b) / sL
    else:
        assert Omega is not None and phi_k is not None, \
            "Omega and phi_k are required when use_decomp=True"
        Lw_hat = _dc_rect(compute_Lcladegw(C_tree, C_space, a, b, gamma, Omega), a, b) / sL
        Lx_hat = _dc_rect(compute_Lcladegw(C_tree, C_space, a, b, gamma, 1.0 - Omega), a, b) / sL
        phi_row = jnp.clip(omega @ phi_k, 0.0, 1.0)
        L_hat = (1.0 - phi_row)[:, None] * Lx_hat + phi_row[:, None] * Lw_hat

    # 3) row-wise alpha blend
    alpha_row = jnp.clip(omega @ alpha_k, 0.0, 1.0)
    C0 = (1.0 - alpha_row)[:, None] * F_hat + alpha_row[:, None] * L_hat

    if equalize == "none":
        return C0

    cnt_k = jnp.maximum(omega.sum(axis=0), 1e-8)

    if equalize == "measured":
        # Scale from the ACTUAL cost, so s_k does not depend on alpha_k or phi_k.
        m_i = jnp.median(jnp.abs(C0), axis=1)
        m_k = (omega.T @ m_i) / cnt_k
        m_k = jax.lax.stop_gradient(jnp.clip(m_k, s_min, s_max))
        target = jax.lax.stop_gradient(jnp.median(m_k))
        s_k = jnp.clip(target / m_k, s_lo, s_hi)

    elif equalize == "alpha":
        # Original behaviour, kept for A/B comparison.
        alpha_ref_k = jax.lax.stop_gradient(jnp.clip(alpha_k, 0.0, 1.0))
        mF_i = jnp.median(jnp.abs(F_hat), axis=1)
        mL_i = jnp.median(jnp.abs(L_hat), axis=1)
        mF_k = jax.lax.stop_gradient(jnp.clip((omega.T @ mF_i) / cnt_k, s_min, s_max))
        mL_k = jax.lax.stop_gradient(jnp.clip((omega.T @ mL_i) / cnt_k, s_min, s_max))
        mLk_safe = jnp.maximum(mL_k, rel_floor * mF_k)
        den_k = jnp.clip((1.0 - alpha_ref_k) * mF_k + alpha_ref_k * mLk_safe, s_min, s_max)
        s_k = jnp.clip((mF_k + mL_k) / den_k, s_lo, s_hi)

    else:
        raise ValueError(f"unknown equalize={equalize!r}")

    s_k = jax.lax.stop_gradient(s_k)
    return C0 * (omega @ s_k)[:, None]
