"""FGW with the structural term split by TREE RESOLUTION rather than by clade membership.

WHY. Binning PEtracer's cell pairs by patristic distance and measuring their spatial separation
against a random pair (1.00 = no information) gives three distinct classes:

                                   fraction of pairs        spatial dist / random
                                   M3-4  M3-1  M3-2         M3-4   M3-1   M3-2
    same clade, resolved           10.1%  7.2%  6.8%        0.591  0.664  0.641   attract
    different clade, still resolved 40.7% 65.1% 63.0%       0.753  0.947  0.910   attract weakly
    at the saturation maximum      49.2% 27.7% 30.2%        1.288  1.213  1.270   REPEL

The paper's Omega^w / Omega^x split is same-clade vs different-clade, which puts the middle
class (attractive) in the same bucket as the saturated class (repulsive) and asks one weight to
serve both. Splitting on the RESOLUTION boundary instead keeps them apart:

    F(T) = <M, T> + sum_c w_c * sum_ijkl Omega^c_ik L(C1_ik, C2_jl) T_ij T_kl

with c in {near, mid, far}. The weights are scalars SHARED ACROSS CLADES, so there is no
per-clade overfitting, and because every mask is symmetric with a scalar weight the cost is
always the gradient of that objective -- none of the i<->k obstruction that per-cell alpha runs
into. The linear term is pinned at weight 1: scaling the whole cost does not move the argmin, so
three weights is exactly the free parameterisation.

Uniform w_near = w_mid = w_far recovers ordinary FGW (up to the alpha/(1-alpha) reparameterisation).
"""
import numpy as np


def tens(C1, C2, Y, Om=None):
    """sum_kl Om_ak (C1_ak - C2_bl)^2 Y_kl.  Om=None means all ones (ordinary GW)."""
    r = Y.sum(1)
    if Om is None:
        return ((C1 ** 2) @ r)[:, None] + ((C2 ** 2) @ Y.sum(0))[None, :] - 2.0 * (C1 @ Y @ C2.T)
    return (((Om * C1 ** 2) @ r)[:, None]
            + (Om @ Y) @ (C2 ** 2).T
            - 2.0 * ((Om * C1) @ Y @ C2.T))


def tens_blocks(C1, C2, Y, blocks, Om=None):
    """Same, for a block-diagonal mask given as index arrays. Cheaper than a dense mask.

    Om, if given, further restricts within each block (used to intersect `near` with `resolved`,
    without which a same-clade pair above the cap would leave mid = resolved - near negative)."""
    r = Y.sum(1)
    out = np.zeros(Y.shape, dtype=float)
    C2sq = C2 ** 2
    for idx in blocks:
        Cb = C1[np.ix_(idx, idx)]; Yb = Y[idx]
        if Om is not None:
            Ob = Om[np.ix_(idx, idx)]
            out[idx] = (((Ob * Cb ** 2) @ r[idx])[:, None] + (Ob @ Yb) @ C2sq.T
                        - 2.0 * ((Ob * Cb) @ Yb @ C2.T))
        else:
            out[idx] = (((Cb ** 2) @ r[idx])[:, None] + (C2sq @ Yb.sum(0))[None, :]
                        - 2.0 * (Cb @ Yb @ C2.T))
    return out


def make_masks(CT, clade, cap=None):
    """near = same clade (all same-clade pairs are below the cap); resolved = below the cap."""
    if cap is None:
        cap = float(CT.max())
    resolved = (CT < cap - 1e-9).astype(float)
    # A cell is at tree distance 0 from itself, so the k = i term belongs to `near`. Keeping it
    # there (rather than dropping it) is what makes uniform weights reproduce ordinary FGW
    # exactly, so the baseline sits inside the grid.
    np.fill_diagonal(resolved, 1.0)
    blocks = [np.where(clade == k)[0] for k in np.unique(clade)]
    return resolved, blocks


def grad(M, C1, C2, T, w, resolved, blocks):
    """d/dT of the objective. w = (w_near, w_mid, w_far)."""
    t_all = tens(C1, C2, T)
    t_near = tens_blocks(C1, C2, T, blocks, Om=resolved)
    t_res = tens(C1, C2, T, Om=resolved)
    t_mid = t_res - t_near
    t_far = t_all - t_res
    return M + 2.0 * (w[0] * t_near + w[1] * t_mid + w[2] * t_far)


def objective(M, C1, C2, T, w, resolved, blocks):
    lin = float((M * T).sum())
    quad = float(((grad(M, C1, C2, T, w, resolved, blocks) - M) * T).sum())
    return lin + 0.5 * quad


def solve(M, C1, C2, p, q, w, resolved, blocks, max_iter=200, tol=1e-9):
    """Conditional gradient, exact EMD step, closed-form quadratic line search."""
    import ot
    w = np.asarray(w, float)
    T = np.outer(p, q)
    f_prev = objective(M, C1, C2, T, w, resolved, blocks)
    hist = [f_prev]
    for _ in range(int(max_iter)):
        G = grad(M, C1, C2, T, w, resolved, blocks)
        D = ot.emd(p, q, G, numItermax=1_000_000) - T
        a2 = 0.5 * float(((grad(M, C1, C2, D, w, resolved, blocks) - M) * D).sum())
        b1 = float((G * D).sum())
        # F(T + gD) = F(T) + g*b1 + g^2*a2. With unequal mask weights the quadratic form is not
        # guaranteed positive, so a2 <= 0 must fall back to the better endpoint rather than to g=1.
        if a2 > 1e-15:
            g = min(1.0, max(0.0, -b1 / (2.0 * a2)))
        else:
            g = 1.0 if (b1 + a2) < 0 else 0.0
        if g <= 0.0:
            break
        T = T + g * D
        f = objective(M, C1, C2, T, w, resolved, blocks)
        hist.append(f)
        if abs(f_prev - f) <= tol * max(abs(f), 1.0):
            f_prev = f
            break
        f_prev = f
    return T, {"objective": hist, "n_iter": len(hist) - 1}


# ---------------------------------------------------------------------------------------------
# Fast path.  On PEtracer the `resolved` mask is EXACTLY constant within every clade-pair block
# (324/324 blocks pure, 0.0000% of cells misassigned, all three sections): the recorder either
# resolves the relationship between two clades or it does not, with no within-block variation.
# So all three masks are clade-block matrices, and the whole structural term is one K x K weight
# matrix W. That collapses the cost:
#
#   Q(T)[i,j] = sum_k W[c_i,c_k] C1_ik^2 r_k        (A: n^2, and independent of T except via r)
#             - 2 sum_k W[c_i,c_k] C1_ik U[k,j]     (B: n^2 m  -- the only heavy term)
#             + sum_k W[c_i,c_k] V[k,j]             (C: collapses to a K x m aggregation)
#
# with U = T C2^T and V = T (C2^2)^T (both n m^2, cheap). B's matrix (W .* C1) does not depend on
# T, so it is built once per weight setting. Net effect on M3-4: 277 s -> ~40 s per solve.
class BlockFGW:
    def __init__(self, M, C1, C2, clade, resolved_kk, dtype=np.float32):
        self.M = np.asarray(M, dtype)
        self.C2T = np.asarray(C2, dtype).T.copy()
        self.C2sqT = np.asarray(C2 ** 2, dtype).T.copy()
        self.cl = np.asarray(clade)
        self.K = int(self.cl.max()) + 1
        self.Rkk = np.asarray(resolved_kk, dtype)          # K x K, 1 where the pair is resolved
        self.C1 = np.asarray(C1, dtype)
        self.dtype = dtype
        self.onehot = np.zeros((self.K, len(self.cl)), dtype)
        self.onehot[self.cl, np.arange(len(self.cl))] = 1.0

    def set_weight_matrix(self, Wkk):
        """Arbitrary clade-level pair weights: W[a,b] is the weight on every pair of cells in
        clades a and b. Any decomposition whose classes are unions of clade-pairs -- tree-distance
        bins, the near/mid/far split, per-clade alpha -- is a special case of this."""
        self.W = np.asarray(Wkk, self.dtype)
        Wexp = self.W[np.ix_(self.cl, self.cl)]
        self.A = (Wexp * self.C1 ** 2)
        self.B = (Wexp * self.C1)
        return self

    def set_weights(self, w_near, w_mid, w_far):
        """W[a,b] = near if a==b and resolved, mid if a!=b and resolved, far otherwise."""
        K = self.K
        eye = np.eye(K, dtype=self.dtype)
        W = (w_near * eye * self.Rkk + w_mid * (1 - eye) * self.Rkk + w_far * (1 - self.Rkk))
        self.W = W.astype(self.dtype)
        Wexp = W[np.ix_(self.cl, self.cl)]
        self.A = (Wexp * self.C1 ** 2)                      # for term A
        self.B = (Wexp * self.C1)                           # for term B
        return self

    def _quad_grad(self, T):
        T = np.asarray(T, self.dtype)
        r = T.sum(1)
        U = T @ self.C2T
        V = T @ self.C2sqT
        termA = (self.A @ r)[:, None]
        termB = -2.0 * (self.B @ U)
        termC = (self.W @ (self.onehot @ V))[self.cl]       # K x m aggregation, then scatter
        return 2.0 * (termA + termB + termC)

    def grad(self, T):
        return self.M + self._quad_grad(T)

    def objective(self, T):
        T = np.asarray(T, self.dtype)
        return float((self.M * T).sum()) + 0.5 * float((self._quad_grad(T) * T).sum())

    def solve(self, p, q, max_iter=200, tol=1e-9):
        import ot
        T = np.outer(p, q).astype(self.dtype)
        f_prev = self.objective(T); hist = [f_prev]
        for _ in range(int(max_iter)):
            G = self.grad(T)
            D = ot.emd(p, q, np.asarray(G, np.float64), numItermax=1_000_000) - T
            a2 = 0.5 * float((self._quad_grad(D) * D).sum())
            b1 = float((G * D).sum())
            if a2 > 1e-15:
                g = min(1.0, max(0.0, -b1 / (2.0 * a2)))
            else:
                g = 1.0 if (b1 + a2) < 0 else 0.0
            if g <= 0.0:
                break
            T = T + g * D
            f = self.objective(T); hist.append(f)
            if abs(f_prev - f) <= tol * max(abs(f), 1.0):
                f_prev = f; break
            f_prev = f
        return T, {"objective": hist, "n_iter": len(hist) - 1}


def clade_resolved_matrix(CT, clade, cap=None):
    """K x K matrix of whether each clade pair is resolved. Errors if the mask is not actually
    block-constant, since the fast path depends on it."""
    if cap is None:
        cap = float(CT.max())
    res = (CT < cap - 1e-9)
    K = int(clade.max()) + 1
    R = np.zeros((K, K))
    for a in range(K):
        ia = clade == a
        for b in range(K):
            v = res[np.ix_(ia, clade == b)]
            if v.size == 0:
                continue
            f = v.mean()
            if 1e-9 < f < 1 - 1e-9:
                raise ValueError(f"resolved mask is not block-constant for clades {a},{b} "
                                 f"({f:.3f} resolved) -- the fast path does not apply")
            R[a, b] = float(f > 0.5)
    np.fill_diagonal(R, 1.0)
    return R
