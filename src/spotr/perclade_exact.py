"""Clade-aware FGW (main.tex eq:cladefgw_cost) as an actual gradient, solved exactly.

THE PAPER'S COST
----------------
    C_nm = (1 - a_t(n)) C^F_nm  +  a_t(n) [ (1 - th_t(n)) L^x_nm  +  th_t(n) L^w_nm ]

with the within-clade mask Om^w_nn' = 1[t(n)=t(n')], Om^x = 11' - Om^w, and L = L^x + L^w.
Write the per-cell structural coefficients

    u_n = a_n * th_n          (weight on WITHIN-clade pairs)
    v_n = a_n * (1 - th_n)    (weight on CROSS-clade pairs)

WHEN IS THAT A GRADIENT?
------------------------
For F(T) = sum_nn'mm' w_nn' |C^T_nn' - C^S_mm'|^2 T_nm T_n'm' we need, for every pair,
w_nn' + w_n'n = (the coefficient the cost puts on row n). Splitting by block:

  WITHIN-clade pairs: both cells share a clade, so u_n = u_n'. w = u/2 works for ANY a_t, th_t.
    -> the mask removes the obstruction completely on this block.
  CROSS-clade pairs:  swapping n <-> n' demands  a_n(1-th_n) = a_n'(1-th_n')  for all cross pairs,
    i.e.  a_t (1 - th_t) = const  across clades.

So the paper's cost IS a valid gradient iff a_t(1-th_t) is clade-independent. Two useful cases:
  th_t = 1 for all t   -> cross term absent, a_t entirely free. Fully separable per clade.
  th_t = 1 - c/a_t     -> cross weight pinned to a shared c, within weight a_t - c free.
This is what the unmasked module could not do: there, a_t had to be uniform, full stop.

THE FIX FOR THE GENERAL CASE
----------------------------
Weight each pair by the mean of its two cells' coefficients, block by block:

    dF/dT_ab = (1-a_a) M_ab
             + u_a * Tw(T)[a,b] + Tw(T; weights u)[a,b]
             + v_a * Tx(T)[a,b] + Tx(T; weights v)[a,b]

where Tw / Tx are the within- / cross-clade GW tensors. When a_t(1-th_t) is constant this
coincides with the paper's cost; otherwise it is the nearest thing to it that minimises something.

Only square_loss, matching the project.
"""
import numpy as np


# ---------------------------------------------------------------- GW tensors

def tens(C1, C2, Y, w=None):
    """sum_kl w_k (C1_ak - C2_bl)^2 Y_kl, exact for arbitrary Y (no marginal assumption)."""
    Yw = Y if w is None else w[:, None] * Y
    r = Yw.sum(1)
    c = Yw.sum(0)
    return ((C1 ** 2) @ r)[:, None] + ((C2 ** 2) @ c)[None, :] - 2.0 * (C1 @ Yw @ C2.T)


def tens_within(C1, C2, Y, blocks, w=None):
    """Same sum, restricted to k in the same block as a. `blocks` is a list of index arrays.

    Block-diagonal Om^w, so this costs sum_k n_k^2 m rather than n^2 m.
    """
    Yw = Y if w is None else w[:, None] * Y
    r = Yw.sum(1)
    out = np.zeros(Y.shape, dtype=float)
    C2sq = C2 ** 2
    for idx in blocks:
        Cb = C1[np.ix_(idx, idx)]
        Yb = Yw[idx]
        out[idx] = (((Cb ** 2) @ r[idx])[:, None]
                    + (C2sq @ Yb.sum(0))[None, :]
                    - 2.0 * (Cb @ Yb @ C2.T))
    return out


def _blocks(labels):
    labels = np.asarray(labels)
    return [np.where(labels == k)[0] for k in np.unique(labels)]


# ---------------------------------------------------------------- cost & objective

def _uv(alpha, theta):
    alpha = np.asarray(alpha, float)
    theta = np.full_like(alpha, 1.0) if theta is None else np.asarray(theta, float)
    return alpha * theta, alpha * (1.0 - theta)


def grad_gw(C1, C2, T, alpha, theta=None, blocks=None):
    """d/dT of the pair-mean-weighted structural term. blocks=None -> no mask (theta ignored)."""
    alpha = np.asarray(alpha, float)
    if blocks is None:
        return alpha[:, None] * tens(C1, C2, T) + tens(C1, C2, T, w=alpha)
    u, v = _uv(alpha, theta)
    has_u, has_v = bool(np.any(u != 0.0)), bool(np.any(v != 0.0))
    Tw = tens_within(C1, C2, T, blocks) if (has_u or has_v) else None
    G = np.zeros(T.shape, dtype=float)
    if has_u:
        G += u[:, None] * Tw + tens_within(C1, C2, T, blocks, w=u)
    if has_v:                                  # skipped entirely at theta = 1
        c = float(v[0])
        if np.all(v == c):
            # v constant (the well-posed family): the cross terms collapse to
            #   v*Tx + Tx(v) = 2c*(tens - Tw), so only ONE full tensor is needed.
            G += 2.0 * c * (tens(C1, C2, T) - Tw)
        else:
            Tx = tens(C1, C2, T) - Tw
            G += v[:, None] * Tx + tens(C1, C2, T, w=v) - tens_within(C1, C2, T, blocks, w=v)
    return G


def cost(M, C1, C2, T, alpha, theta=None, blocks=None):
    """Linearised clade-aware FGW cost at T."""
    return (1.0 - np.asarray(alpha, float))[:, None] * M + grad_gw(C1, C2, T, alpha, theta, blocks)


def cost_paper(M, C1, C2, T, alpha, theta, blocks):
    """The cost EXACTLY as written in main.tex -- row n weighted by its own u_n, v_n.

    Equals `cost` iff alpha_t (1 - theta_t) is constant across clades; otherwise it is not the
    gradient of anything. Kept so the two can be compared directly.
    """
    alpha = np.asarray(alpha, float)
    u, v = _uv(alpha, theta)
    Tw = tens_within(C1, C2, T, blocks)
    Tx = tens(C1, C2, T) - Tw
    return (1.0 - alpha)[:, None] * M + 2.0 * (u[:, None] * Tw + v[:, None] * Tx)


def objective(M, C1, C2, T, alpha, theta=None, blocks=None):
    """F(T). The structural part is quadratic, so it is half its inner product with its gradient."""
    lin = float(((1.0 - np.asarray(alpha, float))[:, None] * M * T).sum())
    return lin + 0.5 * float((grad_gw(C1, C2, T, alpha, theta, blocks) * T).sum())


# ---------------------------------------------------------------- solver

def perclade_fgw(M, C1, C2, p, q, alpha, theta=None, labels=None,
                 max_iter=200, tol=1e-9, verbose=False):
    """Conditional gradient, exact EMD step, closed-form quadratic line search.

    labels=None reproduces the unmasked behaviour (and POT's FGW at uniform alpha).
    Pass labels (per-cell clade id) and theta to use the paper's within/cross decomposition.
    """
    import ot
    alpha = np.asarray(alpha, float)
    blocks = None if labels is None else _blocks(labels)
    T = np.outer(p, q)
    f_prev = objective(M, C1, C2, T, alpha, theta, blocks)
    hist = [f_prev]
    for it in range(int(max_iter)):
        G = cost(M, C1, C2, T, alpha, theta, blocks)
        Tn = ot.emd(p, q, G, numItermax=1_000_000)
        D = Tn - T
        a2 = 0.5 * float((grad_gw(C1, C2, D, alpha, theta, blocks) * D).sum())
        b1 = float((G * D).sum())
        if a2 > 1e-15:
            g = min(1.0, max(0.0, -b1 / (2.0 * a2)))
        else:
            g = 1.0 if b1 < 0 else 0.0
        if g <= 0.0:
            break
        T = T + g * D
        f = objective(M, C1, C2, T, alpha, theta, blocks)
        hist.append(f)
        if verbose:
            print(f"    it {it:>3}  step {g:.4f}  F {f:.8f}")
        if abs(f_prev - f) <= tol * max(abs(f), 1.0):
            f_prev = f
            break
        f_prev = f
    return T, {"objective": hist, "n_iter": len(hist) - 1}
