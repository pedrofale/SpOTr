r"""Neighbourhood-regularised OT: the tree constrains WHO IS NEAR WHOM, not how far apart.

MOTIVATION
----------
The tree metric is ultrametric (verified: 0% of triples violate) and equals 2 x days since the
LCA. An ultrametric on n points needs ~n-1 dimensions; on PEtracer M3-4 the top two dimensions
carry 18.1% of its variance and 90% needs 298. So no 2-D configuration can reproduce it, and
Gromov-Wasserstein -- which penalises (d_tree(i,k) - d_space(j,l))^2 -- is asking for something
unattainable. Empirically the best 2-D embedding of the tree correlates only 0.607 with the tree
metric, yet preserves 37-54% of tree NEIGHBOURHOODS against a 0.8-8.3% chance rate. The local
content survives the dimension collapse; the metric content does not.

THE MODEL
---------
Cells i = 1..n with marginal p, spots j = 1..m at positions y_j in R^2 with marginal q, coupling
T in Pi(p, q), expression cost M in R^{n x m}.

The position the map assigns to cell i is the conditional expectation

    yhat_i(T) = (1 / p_i) * sum_j T_ij y_j                                              (1)

From the tree take a kernel with a PER-CELL BANDWIDTH sigma_i, symmetrised so the penalty is a
genuine quadratic form:

    w_ik = exp(-D^T_ik / sigma_i),   W_ik = (w_ik + w_ki) / 2,   W_ii = 0                (2)

and penalise relatives landing apart:

    R(T) = sum_{i,k} W_ik || yhat_i(T) - yhat_k(T) ||^2  =  2 tr( Yhat^T L Yhat )        (3)

with L = diag(W 1) - W the graph Laplacian of W. The problem is

    min_{T in Pi(p,q)}   <M, T>  +  lambda R(T)                                          (4)

WHAT IS ABSENT, AND WHY IT MATTERS
----------------------------------
No target distance appears anywhere. (3) says relatives should be near each other and is silent
about how near, and silent about non-relatives (W_ik ~ 0). Three consequences:

  * the tree's metric SCALE becomes irrelevant -- calibration, the 70.0 ceiling and the 49% of
    pairs piled there simply drop out, since those pairs carry no weight;
  * the penalty is one-sided (attractive). GW is two-sided: it also insists distant pairs be far
    apart, which is where half its budget went and which 2-D cannot honour anyway;
  * sigma_i is a RESOLUTION, not a trust weight. A motile cell that left its siblings but not
    its clade keeps a constraint, at the scale on which it is still true. Lowering alpha instead
    REMOVES the constraint and hands the cell to expression, which is far weaker -- measured
    549.7 um for expression alone against 356.8 um with the tree.

COST AND CONDITIONING
---------------------
Writing P = diag(p) and Y in R^{m x 2},

    grad_T R = 4 P^{-1} L P^{-1} T Y Y^T                                                 (5)

L is sparse (a kNN graph on the tree), and Y has two columns, so a gradient costs
O(nnz(L) + n m) rather than GW's O(n^2 m + n m^2). The quadratic is rank 2 in the spatial
dimension. Because W is symmetric with scalar entries, (4) is always the gradient of its own
objective -- none of the i<->k obstruction that made per-cell alpha ill-posed under GW.

SEPARABILITY
------------
sigma_i enters only the pairs involving cell i. Changing it does not alter the constraints
among other cells, which is what makes per-cell (or per-clade) resolution well-posed here and
was not true of alpha_i in FGW.

ESTIMATING sigma_i FROM GROUND TRUTH
------------------------------------
With lineage-traced spatial data the true position x_i is known, so sigma_i can be read off
directly by leave-one-out prediction. For a candidate sigma, using only the OTHER cells:

    mu_i(sigma)  = sum_{k != i} w_ik(sigma) x_k / sum_{k != i} w_ik(sigma)
    s2_i(sigma)  = sum_{k != i} w_ik ||x_k - mu_i||^2 / (2 sum_{k != i} w_ik)
    loglik_i     = -||x_i - mu_i||^2 / (2 s2_i) - log s2_i                                (6)
    sigmahat_i   = argmax_sigma loglik_i

(6) rewards accuracy and penalises breadth, so a wide neighbourhood does not win by default --
a mistake worth avoiding: scoring by ||x_i - mu_i|| / spread instead makes coarser always better
and every cell picks the same scale.
"""
import numpy as np


def tree_kernel(DT, sigma, sym=True):
    """W from (2). sigma is scalar or per-cell."""
    s = np.asarray(sigma, float)
    W = np.exp(-DT / (s[:, None] if s.ndim else s))
    if sym:
        W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def positions(T, p, Y):
    """yhat from (1)."""
    return (T @ Y) / np.maximum(p[:, None], 1e-300)


def penalty(T, p, Y, W):
    """R(T) from (3)."""
    Yh = positions(T, p, Y)
    deg = W.sum(1)
    return float((deg[:, None] * Yh * Yh).sum() - (Yh * (W @ Yh)).sum()) * 2.0


def grad_penalty(T, p, Y, W):
    """grad_T R from (5)."""
    Yh = positions(T, p, Y)
    LY = W.sum(1)[:, None] * Yh - W @ Yh
    return 4.0 * (LY / np.maximum(p[:, None], 1e-300)) @ Y.T


def solve(M, p, q, Y, W, lam, max_iter=200, tol=1e-9):
    """Conditional gradient with exact EMD steps and a closed-form quadratic line search."""
    import ot
    T = np.outer(p, q)
    f = lambda X: float((M * X).sum()) + lam * penalty(X, p, Y, W)
    f_prev = f(T); hist = [f_prev]
    for _ in range(int(max_iter)):
        G = M + lam * grad_penalty(T, p, Y, W)
        D = ot.emd(p, q, G, numItermax=1_000_000) - T
        a2 = lam * penalty(D, p, Y, W)          # R is a pure quadratic, so R(D) is its curvature
        b1 = float((G * D).sum())
        if a2 > 1e-15:
            g = min(1.0, max(0.0, -b1 / (2.0 * a2)))
        else:
            g = 1.0 if (b1 + a2) < 0 else 0.0
        if g <= 0.0:
            break
        T = T + g * D
        fn = f(T); hist.append(fn)
        if abs(f_prev - fn) <= tol * max(abs(fn), 1.0):
            f_prev = fn; break
        f_prev = fn
    return T, {"objective": hist, "n_iter": len(hist) - 1}


def estimate_sigma(DT, X, sigmas):
    """sigmahat_i from (6), leave-one-out. Returns (sigmahat, loglik[n, len(sigmas)])."""
    n = len(DT)
    L = np.empty((n, len(sigmas)))
    for c, s in enumerate(sigmas):
        W = np.exp(-DT / s); np.fill_diagonal(W, 0.0)
        Z = np.maximum(W.sum(1), 1e-300)
        mu = (W @ X) / Z[:, None]
        d2 = ((X[None, :, :] - mu[:, None, :]) ** 2).sum(2) if n <= 3000 else None
        if d2 is None:
            s2 = np.array([float((W[i] * ((X - mu[i]) ** 2).sum(1)).sum() / (2 * Z[i]))
                           for i in range(n)])
        else:
            s2 = (W * d2).sum(1) / (2 * Z)
        s2 = np.maximum(s2, 1e-12)
        L[:, c] = -((X - mu) ** 2).sum(1) / (2 * s2) - np.log(s2)
    return np.asarray(sigmas)[L.argmax(1)], L


# ---------------------------------------------------------------------------------------------
# CORRECTED PENALTY.
# R(T) above penalises || E[y_i] - E[y_k] ||^2, the distance between the cells' MEAN positions.
# That is minimised by the maximally blurred coupling T = p q^T, where every cell's mean is the
# same spot centroid: R = 0 and grad R = 0 there (measured: 4e-44 and 8e-25). So the penalty is
# degenerate at the Frank-Wolfe start and, worse, rewards blurring at every iterate -- it fights
# the sharpening the mapping needs. Empirically it barely improved on expression alone
# (549.7 -> 541.8 um) and larger sigma made it monotonically worse.
#
# The right object is the MEAN distance, not the distance between means:
#
#     R2(T) = sum_{i,k} W_ik  sum_{j,l} T_ij T_kl || y_j - y_l ||^2
#
# i.e. the expected squared separation between where i and k land. Jensen's gap between the two
# is exactly the within-row variance, which is what penalises blur. Expanding, with
# a_i = sum_j T_ij ||y_j||^2 and B = T Y,
#
#     R2(T) = 2 a^T W p  -  2 tr(B^T W B)
#     grad R2[i,j] = 2 (Wp)_i ||y_j||^2  -  4 < (W T Y)_i , y_j >
#
# It is non-zero at the product coupling and still one-sided: no target distance, non-relatives
# carry no weight.

def penalty2(T, p, Y, W):
    a = T @ (Y ** 2).sum(1)
    B = T @ Y
    return 2.0 * float(a @ (W @ p)) - 2.0 * float((B * (W @ B)).sum())


def grad_penalty2(T, p, Y, W):
    return (2.0 * (W @ p)[:, None] * (Y ** 2).sum(1)[None, :]
            - 4.0 * (W @ (T @ Y)) @ Y.T)


def solve2(M, p, q, Y, W, lam, max_iter=300, tol=1e-10):
    import ot
    T = np.outer(p, q)
    f = lambda X: float((M * X).sum()) + lam * penalty2(X, p, Y, W)
    f_prev = f(T); hist = [f_prev]
    for _ in range(int(max_iter)):
        G = M + lam * grad_penalty2(T, p, Y, W)
        D = ot.emd(p, q, G, numItermax=1_000_000) - T
        # R2 is quadratic in T but has a linear part, so the curvature is its purely
        # quadratic piece: -2 tr(B_D^T W B_D) with B_D = D Y.
        BD = D @ Y
        a2 = lam * (-2.0 * float((BD * (W @ BD)).sum()))
        b1 = float((G * D).sum())
        if a2 > 1e-15:
            g = min(1.0, max(0.0, -b1 / (2.0 * a2)))
        else:
            g = 1.0 if (b1 + a2) < 0 else 0.0
        if g <= 0.0:
            break
        T = T + g * D
        fn = f(T); hist.append(fn)
        if abs(f_prev - fn) <= tol * max(abs(fn), 1.0):
            f_prev = fn; break
        f_prev = fn
    return T, {"objective": hist, "n_iter": len(hist) - 1}


def estimate_sigma_corrected(DT, X, sigmas, ess_min=3.0):
    r"""sigmahat_i with the small-neighbourhood degeneracy removed.

    THE PROBLEM. estimate_sigma scores a candidate sigma by the Gaussian log-density of x_i under
    (mu_i, s2_i) with s2_i the WEIGHTED VARIANCE OF THE SAME NEIGHBOURHOOD. As sigma shrinks the
    neighbourhood collapses onto a single nearest relative, s2_i -> 0, and -log s2_i -> +inf: the
    likelihood is unbounded and the argmax runs to the smallest sigma. Measured on M3-4: 36% of
    cells selected a bandwidth whose effective neighbourhood was 1-3 cells, i.e. "predict me from
    my single nearest relative", and the median effective size was 5.

    THE FIX. s2_i is estimated from an effective sample of
        ESS_i = (sum_k w_ik)^2 / sum_k w_ik^2                                   (Kish)
    points, so it is both biased low and not the right spread for predicting a NEW point. Use the
    unbiased estimate and the predictive variance for one further draw:

        s2_unb  = s2 * ESS / (ESS - 1)
        s2_pred = s2_unb * (1 + 1 / ESS)

    As ESS -> 1 the correction diverges, so a one-relative neighbourhood is now penalised rather
    than rewarded, and the degeneracy disappears without an arbitrary cutoff. ess_min only guards
    the ESS <= 1 division.
    """
    n = len(DT)
    DTd = DT.copy(); np.fill_diagonal(DTd, np.inf)
    L = np.full((n, len(sigmas)), -np.inf)
    E = np.zeros((n, len(sigmas)))
    for c, s in enumerate(sigmas):
        W = np.exp(-DTd / s)
        Z = np.maximum(W.sum(1), 1e-300)
        ess = Z ** 2 / np.maximum((W ** 2).sum(1), 1e-300)
        mu = (W @ X) / Z[:, None]
        d2 = ((X[None, :, :] - mu[:, None, :]) ** 2).sum(2)
        s2 = np.maximum((W * d2).sum(1) / (2 * Z), 1e-12)
        ok = ess > max(ess_min, 1.0 + 1e-9)
        s2p = s2 * (ess / np.maximum(ess - 1.0, 1e-9)) * (1.0 + 1.0 / ess)
        ll = -((X - mu) ** 2).sum(1) / (2 * s2p) - np.log(s2p)
        L[ok, c] = ll[ok]
        E[:, c] = ess
    best = L.argmax(1)
    return np.asarray(sigmas)[best], L, E[np.arange(n), best]
