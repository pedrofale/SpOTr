"""Fused GW with a LEARNED structural loss.

FACT 0 (verified in experiments/night/00_affine_invariance.py).  Under fixed marginals

    sum_ijkl (C1_ik - C2_jl)^2 T_ij T_kl  =  const - 2 <C1, T C2 T'>,

so square-loss GW maximises a single scalar -- the covariance between patristic distance and the
spatial distance between the images -- and any affine rescaling of either matrix is invisible
except through alpha.  The only two ways to change what the tree MEANS are a NONLINEAR transform
of the distances, or a reweighting of pairs.

This module implements the first, in its general form.  Let (t, s) be the joint law of (tree
distance, spatial distance) measured on a matched reference section.  Its correspondence-analysis
(Hirschfeld-Gebelein-Renyi) expansion is

    p(t, s) / (p(t) p(s))  =  1 + sum_r rho_r phi_r(t) psi_r(s),

with rho_1 >= rho_2 >= ... the canonical correlations and phi_r, psi_r standardised to mean 0 and
variance 1 under the marginals.  We use the leading R components as the structural loss:

    F(T) = (1 - alpha) <M, T>
         + alpha * sum_r rho_r * sum_ijkl (phi_r(C1)_ik - psi_r(C2)_jl)^2 T_ij T_kl.

ORDINARY FGW IS THE RANK-1 IDENTITY CASE: R = 1, phi = (t - mean)/sd, psi = (s - mean)/sd,
rho_1 = 1.  By Fact 0 the standardisation is free, so this is byte-for-byte POT's FGW at a
reparameterised alpha.  Verified in tests/test_genfgw.py.

Cost per gradient: R products of size n^2 m and R of size n m^2 -- one ordinary GW gradient
times R.
"""
import numpy as np


class GenFGW:
    """M (n x m); Phi list of R matrices (n x n); Psi list of R matrices (m x m); rho (R,)."""

    def __init__(self, M, Phi, Psi, rho, alpha, dtype=np.float32):
        self.M = np.asarray(M, dtype)
        self.Phi = [np.asarray(P, dtype) for P in Phi]
        self.Psi = [np.asarray(P, dtype) for P in Psi]
        self.Phi2 = [P ** 2 for P in self.Phi]
        self.Psi2 = [P ** 2 for P in self.Psi]
        self.rho = np.asarray(rho, float)
        self.alpha = float(alpha)
        self.dtype = dtype

    def _tens(self, T):
        """sum_r rho_r * d/dT of GW_r, i.e. 2 * (Phi^2 r 1' + 1 c' Psi^2 - 2 Phi T Psi)."""
        T = np.asarray(T, self.dtype)
        r = T.sum(1); c = T.sum(0)
        G = np.zeros(T.shape, self.dtype)
        for k in range(len(self.Phi)):
            w = self.dtype(self.rho[k])
            G += w * ((self.Phi2[k] @ r)[:, None] + (self.Psi2[k] @ c)[None, :]
                      - 2.0 * (self.Phi[k] @ (T @ self.Psi[k])))
        return 2.0 * G

    def grad(self, T):
        return (1.0 - self.alpha) * self.M + self.alpha * self._tens(T)

    def objective(self, T):
        T = np.asarray(T, self.dtype)
        return float(((1.0 - self.alpha) * self.M * T).sum()) \
            + 0.5 * self.alpha * float((self._tens(T) * T).sum())

    def solve(self, p, q, max_iter=200, tol=1e-9, G0=None):
        import ot
        T = (np.outer(p, q) if G0 is None else np.asarray(G0)).astype(self.dtype)
        f_prev = self.objective(T)
        hist = [f_prev]
        for _ in range(int(max_iter)):
            G = self.grad(T)
            D = ot.emd(p, q, np.asarray(G, np.float64), numItermax=1_000_000) - T
            # F is quadratic along T + gD:  F(T) + g<G,D> + g^2 * a2
            a2 = 0.5 * self.alpha * float((self._tens(D) * D).sum())
            b1 = float((G * D).sum())
            if a2 > 1e-15:
                g = min(1.0, max(0.0, -b1 / (2.0 * a2)))
            else:
                g = 1.0 if (b1 + a2) < 0 else 0.0
            if g <= 0.0:
                break
            T = T + g * D
            f = self.objective(T)
            hist.append(f)
            if abs(f_prev - f) <= tol * max(abs(f), 1.0):
                f_prev = f
                break
            f_prev = f
        return T, {"objective": hist, "n_iter": len(hist) - 1}


# ---------------------------------------------------------------------------------------------
# Estimating the transforms from a matched reference section
# ---------------------------------------------------------------------------------------------
def tree_bins(t, nbin=17, min_frac=0.002):
    """Bins of tree distance, EQUAL WIDTH IN DIVERGENCE TIME, with the saturation ceiling given a
    bin of its own.

    Two properties of the PEtracer metric force this.  (i) The ceiling -- pairs whose last common
    ancestor is the founding cell -- is a point mass holding 28-49% of all pairs, so it must not
    be pooled with anything else.  (ii) Below the ceiling the distribution is itself concentrated
    near the top, so quantile edges collapse onto repeated values and leave the informative
    short-divergence range in a single enormous bin.  Equal width in days keeps that range
    resolved; bins holding less than min_frac of the pairs are merged rightwards."""
    t = np.asarray(t, float)
    cap = t.max()
    below = t[(t < cap - 1e-9)]
    e = list(np.linspace(below.min(), below.max(), nbin))
    # merge sparse bins rightwards so every phi value is estimated from enough pairs
    keep = [e[0]]
    for k in range(1, len(e)):
        if ((t >= keep[-1]) & (t < e[k])).mean() >= min_frac or k == len(e) - 1:
            keep.append(e[k])
    return np.r_[keep[:-1], cap - 1e-9, cap + 1e-9]


def space_bins(s, nbin=24):
    s = np.asarray(s, float)
    return np.unique(np.r_[np.quantile(s, np.linspace(0, 1, nbin)), s.max() * 1.001 + 1e-9])


def fit_kernel(t, s, ncomp=2, nbin=24, te=None, se=None):
    """Correspondence analysis of the joint histogram of (tree distance, spatial distance)."""
    t = np.asarray(t, float); s = np.asarray(s, float)
    te = tree_bins(t, nbin) if te is None else te
    se = space_bins(s, nbin) if se is None else se
    P, _, _ = np.histogram2d(t, s, bins=[te, se])
    P = P / P.sum()
    a = P.sum(1); b = P.sum(0)
    kr = a > 0; kc = b > 0
    Q = P[np.ix_(kr, kc)] / np.sqrt(np.outer(a[kr], b[kc]))
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)
    tc = 0.5 * (te[:-1] + te[1:]); sc = 0.5 * (se[:-1] + se[1:])
    rho, phi, psi = [], [], []
    for r in range(1, min(ncomp + 1, len(S))):
        f = np.zeros(len(a)); g = np.zeros(len(b))
        f[kr] = U[:, r] / np.sqrt(a[kr])
        g[kc] = Vt[r] / np.sqrt(b[kc])
        if np.polyfit(tc[kr], f[kr], 1)[0] < 0:      # sign: phi increasing in tree distance
            f, g = -f, -g
        rho.append(float(S[r])); phi.append(f); psi.append(g)
    return dict(rho=np.array(rho), phi=np.array(phi), psi=np.array(psi),
                tc=tc, sc=sc, te=te, se=se, a=a, b=b, P=P)


def apply_lookup(C, grid, values):
    """Elementwise piecewise-linear lookup, clamped outside the fitted range."""
    return np.interp(np.asarray(C, float), grid, values)


def standardise(A):
    """Zero mean, unit variance over the strict upper triangle.  Free by Fact 0; it only makes
    alpha mean the same thing across different transforms."""
    A = np.asarray(A, float)
    iu = np.triu_indices(A.shape[0], 1)
    v = A[iu]
    return (A - v.mean()) / max(v.std(), 1e-12)


def build(C_tree, C_space, K, ncomp=None, standardise_out=True):
    """Turn a fitted kernel K into the (Phi, Psi, rho) triple for GenFGW."""
    R = len(K["rho"]) if ncomp is None else min(ncomp, len(K["rho"]))
    Phi, Psi = [], []
    for r in range(R):
        A = apply_lookup(C_tree, K["tc"], K["phi"][r])
        B = apply_lookup(C_space, K["sc"], K["psi"][r])
        if standardise_out:
            A = standardise(A); B = standardise(B)
        # the diagonal keeps whatever the lookup clamps to at distance 0, which is the
        # transform's value for "as close as it gets" -- the same convention as C_ii = 0 in GW
        Phi.append(A); Psi.append(B)
    return Phi, Psi, K["rho"][:R]


# ---------------------------------------------------------------------------------------------
# The whole method, end to end
# ---------------------------------------------------------------------------------------------
def learn_relationship(C_tree_ref, xy_ref, ncomp=2, nbin=17):
    """STEP 1.  Estimate the tree->space relationship from a MATCHED reference: cells for which
    both the lineage tree and the true positions are known.

    C_tree_ref : (r x r) patristic distances among the reference cells
    xy_ref     : (r x 2) their positions
    Returns the kernel (canonical correlations and transforms) to hand to `map_cells`."""
    from scipy.spatial.distance import pdist
    C_tree_ref = np.asarray(C_tree_ref, float)
    iu = np.triu_indices(len(C_tree_ref), 1)
    return fit_kernel(C_tree_ref[iu], pdist(np.asarray(xy_ref, float)),
                      ncomp=ncomp, nbin=nbin)


def map_cells(M, C_tree, C_space, kernel=None, w=None, q=None, p=None,
              max_iter=200, tol=1e-9, dtype=np.float32):
    """STEP 2.  Place cells onto spots.

    M       : (n x m) expression cost between the cells to place and the spots
    C_tree  : (n x n) patristic distances among those cells
    C_space : (m x m) distances between spot centres
    kernel  : output of learn_relationship; None gives ordinary square-loss FGW
    w       : weight on the structural term, w = alpha / (1 - alpha) on standardised matrices.
              None calls select_weight, which needs `kernel`.
    q       : spot marginal.  Default uniform, but the observed cell density per spot (nuclei
              counts) is both available in practice and worth ~20 um -- see
              experiments/night/20_marginals.py.
    p       : cell marginal, default uniform.

    Returns (T, w) with T the (n x m) coupling."""
    M = np.asarray(M, float)
    n, m = M.shape
    p = np.ones(n) / n if p is None else np.asarray(p, float)
    q = np.ones(m) / m if q is None else np.asarray(q, float)
    if kernel is None:
        Phi, Psi, rho = [standardise(C_tree)], [standardise(C_space)], np.array([1.0])
    else:
        Phi, Psi, rho = build(C_tree, C_space, kernel)
    if w is None:
        if kernel is None:
            raise ValueError("automatic weight selection needs a kernel")
        w = select_weight(M, C_tree, C_space, kernel, q=q, p=p)
    T, _ = GenFGW(M, Phi, Psi, rho, w / (1.0 + w), dtype=dtype).solve(
        p, q, max_iter=max_iter, tol=tol)
    return np.asarray(T, float), w


def select_weight(M, C_tree, C_space, kernel, grid=None, use_kernel=False, q=None, p=None,
                  **kw):
    """Choose w with NO ground truth on the cells being placed.

    A mapping induces its own joint distribution of (tree distance, distance between the assigned
    spots).  Pick the weight whose induced joint is closest in KL to the reference joint.  Every
    ingredient is observable: the cells' own tree, the spot coordinates, and the reference.

    Measured accuracy (experiments/night/14_select_w.py): with use_kernel=False -- i.e. selecting
    on the square-loss arm, where the criterion is not the objective being optimised -- the choice
    lands 0.0 / 3.7 / 4.7 um from the oracle on the three PEtracer sections.  With use_kernel=True
    the criterion is partly self-fulfilling and the gap grows to 14-45 um."""
    import ot
    M = np.asarray(M, float); C_tree = np.asarray(C_tree, float)
    C_space = np.asarray(C_space, float)
    n, m = M.shape
    p = np.ones(n) / n if p is None else np.asarray(p, float)
    q = np.ones(m) / m if q is None else np.asarray(q, float)
    grid = np.geomspace(0.002, 1.5, 16) if grid is None else np.asarray(grid, float)
    if use_kernel:
        Phi, Psi, rho = build(C_tree, C_space, kernel)
    else:
        Phi, Psi, rho = [standardise(C_tree)], [standardise(C_space)], np.array([1.0])
    iu = np.triu_indices(n, 1)
    t = C_tree[iu]
    P = kernel["P"]
    best, best_kl = grid[0], np.inf
    for w in grid:
        T, _ = GenFGW(M, Phi, Psi, rho, w / (1.0 + w), **kw).solve(p, q, max_iter=200, tol=1e-9)
        am = np.asarray(T).argmax(1)
        H, _, _ = np.histogram2d(t, C_space[am[iu[0]], am[iu[1]]],
                                 bins=[kernel["te"], kernel["se"]])
        H = H / H.sum()
        kl = float((H * np.log((H + 1e-10) / (P + 1e-10))).sum())
        if kl < best_kl:
            best_kl, best = kl, float(w)
    return best


def solve_semirelaxed(model, marginal, side="cells", max_iter=200, tol=1e-9, G0=None):
    """Conditional gradient for a SEMI-RELAXED problem: one marginal is fixed, the other is free.

    side="cells"  fixes T 1 = marginal   -- every cell is placed, spot occupancy free
    side="spots"  fixes T' 1 = marginal  -- every spot is filled, the MIX OF CELL TYPES IS FREE

    side="spots" is the right formulation for deconvolution.  A balanced problem forces the
    predicted composition over spots to equal the composition of the reference cells, so if the
    dissociated data is not a representative sample of the tissue -- on BCSA3 the two differ by a
    total variation distance of 0.55 -- a balanced map is wrong by construction before the cost is
    even consulted.

    Motivation: the uniform spot marginal asserts that every spot holds the same number of the
    cells being mapped, which is an assumption about the query set, not about the tissue.  Dropping
    it costs nothing computationally -- the linear step becomes a row-wise argmin instead of an
    EMD -- and it lets clonal territories keep their real, uneven density."""
    marginal = np.asarray(marginal, float)
    n, m = model.M.shape
    if side == "cells":
        T = (np.outer(marginal, np.ones(m) / m) if G0 is None else np.asarray(G0))
    else:
        T = (np.outer(np.ones(n) / n, marginal) if G0 is None else np.asarray(G0))
    T = T.astype(model.dtype)
    f_prev = model.objective(T)
    hist = [f_prev]
    for _ in range(int(max_iter)):
        G = model.grad(T)
        S = np.zeros_like(T)
        Gd = np.asarray(G, np.float64)
        if side == "cells":
            S[np.arange(n), Gd.argmin(1)] = marginal.astype(model.dtype)
        else:
            S[Gd.argmin(0), np.arange(m)] = marginal.astype(model.dtype)
        D = S - T
        a2 = 0.5 * model.alpha * float((model._tens(D) * D).sum())
        b1 = float((G * D).sum())
        if a2 > 1e-15:
            g = min(1.0, max(0.0, -b1 / (2.0 * a2)))
        else:
            g = 1.0 if (b1 + a2) < 0 else 0.0
        if g <= 0.0:
            break
        T = T + g * D
        f = model.objective(T)
        hist.append(f)
        if abs(f_prev - f) <= tol * max(abs(f), 1.0):
            f_prev = f
            break
        f_prev = f
    return T, {"objective": hist, "n_iter": len(hist) - 1}
