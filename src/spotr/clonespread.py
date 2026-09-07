"""THE CLONE-SPREAD COST: an interpretable replacement for the Gromov-Wasserstein term.

WHAT GW GETS WRONG.  Square-loss GW asks patristic distance to equal spatial distance.  Those have
different units, and worse, the conditional law of separation given divergence age changes SHAPE
with age: a tight spike near zero for recent divergence, and, at the saturation ceiling where
28-49% of pairs sit, simply the distance distribution between two random points in the tissue.
There the tree carries no positional information at all, and GW still hands those pairs a numeric
distance target.

WHAT THE TREE ACTUALLY SAYS.  Two cells whose last common ancestor lived `a` days ago started in
the SAME PLACE and have been moving and dividing apart ever since.  So the tree predicts a SCALE,
not a distance:

    r(a)  =  the spread of a clone whose founding cell lived a days ago      [microns]

and the separation of such a pair is Rayleigh with that scale.  The negative log-likelihood of an
assignment is then

    -log P(d | a)  =  d^2 / (2 r(a)^2)  -  log d  +  2 log r(a)

THE SIMPLIFICATION THAT MAKES IT CHEAP.  Summed against T (x) T under FIXED MARGINALS, the last
two terms are constants: `2 log r(a_ik)` does not depend on the spots and contributes
sum_ik 2 log r(a_ik) p_i p_k, and `-log d_jl` does not depend on the cells and contributes
sum_jl -log(d_jl) q_j q_l.  Both drop out of the argmin.  What is left is

    S(T)  =  sum_ijkl  w_ik * d_jl^2 * T_ij T_kl  / 2 ,        w_ik = 1 / r(a_ik)^2

an inverse-variance-weighted attraction: recent relatives are pulled together hard, ancient ones
are barely constrained, and nothing is ever pushed apart on the strength of a tree that has
stopped saying anything.

COST.  Expanding d_jl^2 = ||y_j||^2 + ||y_l||^2 - 2 y_j.y_l leaves only n x n and n x m work --
NO n^2 m tensor.  One gradient is ~1000x cheaper than a GW gradient on our sizes.

INTERPRETABILITY.  One monotone function in microns, estimable from a single slice.  Per-clade
deviations from it are a motility readout in physical units: a clade whose cells sit at 1.4 x r(a)
is 1.4 x more dispersed than its divergence times predict.
"""
import numpy as np


# ---------------------------------------------------------------------------------------------
# the one function that has to be estimated
# ---------------------------------------------------------------------------------------------
def fit_spread(t, s, nbin=22, min_pairs=200):
    """r(a) from reference pairs.  t = patristic distance, s = spatial distance (um).

    Age a = t/2 (patristic distance is the sum of both root-to-leaf paths).  r(a) is estimated as
    sqrt(E[s^2 | a] / 2), the per-axis RMS separation, on equal-width age bins with the saturation
    ceiling given a bin of its own."""
    t = np.asarray(t, float); s = np.asarray(s, float)
    a = t / 2.0
    cap = a.max()
    below = a[a < cap - 1e-9]
    edges = np.r_[np.linspace(below.min(), below.max(), nbin), cap - 1e-9, cap + 1e-9]
    ac, r = [], []
    for i in range(len(edges) - 1):
        m = (a >= edges[i]) & (a < edges[i + 1])
        if m.sum() < min_pairs:
            continue
        ac.append(a[m].mean())
        r.append(np.sqrt((s[m] ** 2).mean() / 2.0))
    ac, r = np.array(ac), np.array(r)
    r = np.maximum.accumulate(r)                    # monotone: a clone cannot un-spread with age
    return dict(age=ac, r=r)


def spread_weights(C_tree, spread, floor=1e-3):
    """W[i,k] = 1 / r(a_ik)^2, normalised to mean 1 so alpha means the same thing across fits."""
    a = np.asarray(C_tree, float) / 2.0
    r = np.interp(a, spread["age"], spread["r"])
    W = 1.0 / np.maximum(r, floor) ** 2
    iu = np.triu_indices(W.shape[0], 1)
    return W / W[iu].mean()


# ---------------------------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------------------------
class CloneSpreadOT:
    """F(T) = (1-alpha) <M,T> + alpha * 0.5 * sum_ijkl W_ik d_jl^2 T_ij T_kl

    Y is the (m x 2) matrix of spot coordinates, in the same units r(a) was fitted in."""

    def __init__(self, M, W, Y, alpha, dtype=np.float64):
        self.M = np.asarray(M, dtype)
        self.W = np.asarray(W, dtype)
        self.Y = np.asarray(Y, dtype)
        self.y2 = (self.Y ** 2).sum(1)
        self.alpha = float(alpha)
        self.dtype = dtype
        # scale the structural term so alpha is comparable with an FGW alpha on normalised costs
        self.scale = 1.0 / max(self.y2.mean(), 1e-12)

    def _pieces(self, T):
        r = T.sum(1)                       # row masses
        u = T @ self.Y                     # n x 2   (mass-weighted position of each cell)
        v = T @ self.y2                    # n       (mass-weighted squared position)
        return r, u, v

    def _quad_grad(self, T):
        """d/dT of  S(T) = 0.5 * sum_ijkl W_ik d_jl^2 T_ij T_kl.

        Expanding d_jl^2 = y2_j + y2_l - 2 y_j.y_l and using W symmetric,
            S(T) = v.(W r) - sum_ik W_ik u_i.u_k
        so
            dS/dT_ab = y2_b (W r)_a + (W v)_a - 2 y_b.(W u)_a
        The middle term is constant along a row, so it does not move the EMD step, but it is kept
        because the objective and the line search must agree."""
        r, u, v = self._pieces(T)
        Wr = self.W @ r
        Wv = self.W @ v
        Wu = self.W @ u
        return self.scale * (np.outer(Wr, self.y2) + Wv[:, None] - 2.0 * (Wu @ self.Y.T))

    def grad(self, T):
        return (1.0 - self.alpha) * self.M + self.alpha * self._quad_grad(T)

    def objective(self, T):
        r, u, v = self._pieces(T)
        S = float(v @ (self.W @ r)) - float((u * (self.W @ u)).sum())
        return float(((1.0 - self.alpha) * self.M * T).sum()) + self.alpha * self.scale * S

    def solve(self, p, q, max_iter=200, tol=1e-9, G0=None):
        import ot
        T = (np.outer(p, q) if G0 is None else np.asarray(G0)).astype(self.dtype)
        f_prev = self.objective(T)
        hist = [f_prev]
        for _ in range(int(max_iter)):
            G = self.grad(T)
            D = ot.emd(p, q, np.asarray(G, np.float64), numItermax=1_000_000) - T
            # F is quadratic along T + gD.  The structural part is a homogeneous quadratic, so
            # its second-order coefficient is exactly S_raw(D) -- no factor of one half.
            a2 = self.alpha * self.scale * (
                float((D @ self.y2) @ (self.W @ D.sum(1)))
                - float(((D @ self.Y) * (self.W @ (D @ self.Y))).sum()))
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
# motility readout
# ---------------------------------------------------------------------------------------------
def clade_dispersion(C_tree, xy, clade, spread):
    """For each clade: the ratio of its cells' observed separation to what r(a) predicts.

    1.0 = exactly as dispersed as its divergence times say; > 1 = more motile."""
    a = np.asarray(C_tree, float) / 2.0
    out = {}
    for k in np.unique(clade):
        idx = np.where(clade == k)[0]
        if len(idx) < 15:
            continue
        iu = np.triu_indices(len(idx), 1)
        aa = a[np.ix_(idx, idx)][iu]
        dd = np.linalg.norm(xy[idx][iu[0]] - xy[idx][iu[1]], axis=1)
        pred = np.interp(aa, spread["age"], spread["r"]) * np.sqrt(2.0)
        out[k] = float(np.sqrt((dd ** 2).mean()) / np.sqrt((pred ** 2).mean()))
    return out
