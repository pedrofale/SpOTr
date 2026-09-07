"""Per-cell trust with a PRODUCT weight, so a cell can actually be removed from the tree term.

The pair-mean form w_ik = (a_i + a_k)/2 cannot exclude a cell: setting a_i = 0 still leaves the
pair weighted by a_k/2, so a "distrusted" cell keeps imposing tree constraints on its relatives
even as it stops obeying them. That is why down-weighting known stray cells came out 48.8 um
WORSE than uniform, and worse than shuffling the same values.

The product form does what was intended:

    F(T) = <M, T> + lam * sum_ijkl a_i a_k L(C1_ik, C2_jl) T_ij T_kl

a_i = 0 zeroes every pair involving cell i, on both sides. The weight is symmetric, so this is
always the gradient of that objective -- no i<->k obstruction. Uniform a = 1 recovers ordinary
FGW with tree/expression ratio lam. Per-cell a is COMPUTED from a statistic; only lam is fitted.
"""
import numpy as np


def tens(C1, C2, Y, w=None):
    Yw = Y if w is None else w[:, None] * Y
    return (((C1 ** 2) @ Yw.sum(1))[:, None] + ((C2 ** 2) @ Yw.sum(0))[None, :]
            - 2.0 * (C1 @ Yw @ C2.T))


def grad(M, C1, C2, T, a, lam):
    return M + 2.0 * lam * a[:, None] * tens(C1, C2, T, w=a)


def objective(M, C1, C2, T, a, lam):
    return float((M * T).sum()) + lam * float((a[:, None] * tens(C1, C2, T, w=a) * T).sum())


def solve(M, C1, C2, p, q, a, lam, max_iter=200, tol=1e-9):
    import ot
    a = np.asarray(a, float)
    T = np.outer(p, q)
    f_prev = objective(M, C1, C2, T, a, lam); hist = [f_prev]
    for _ in range(int(max_iter)):
        G = grad(M, C1, C2, T, a, lam)
        D = ot.emd(p, q, G, numItermax=1_000_000) - T
        a2 = lam * float((a[:, None] * tens(C1, C2, D, w=a) * D).sum())
        b1 = float((G * D).sum())
        if a2 > 1e-15:
            g = min(1.0, max(0.0, -b1 / (2.0 * a2)))
        else:
            g = 1.0 if (b1 + a2) < 0 else 0.0
        if g <= 0.0:
            break
        T = T + g * D
        f = objective(M, C1, C2, T, a, lam); hist.append(f)
        if abs(f_prev - f) <= tol * max(abs(f), 1.0):
            f_prev = f; break
        f_prev = f
    return T, {"objective": hist, "n_iter": len(hist) - 1}
