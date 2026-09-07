"""POSITIVE CONTROL: plant per-branch alpha structure and see whether the method recovers it.

CONSTRUCTION (v2 -- v1 clustered cells around subclade centres, which made the uniform spot
marginal badly wrong and destroyed the mapping before alpha could matter; alpha=0 beat plain
expression nearest-neighbour 0.039 vs 0.120. Fixed by assigning cells to spots uniformly.)

  - Every spot receives EXACTLY n/m cells, so the uniform marginals the solver assumes are
    exactly the truth. A cell's position is its spot's position.
  - Expression is a smooth random field of position plus noise -- identical rule for both
    branches, so expression is equally (un)informative on each side.
  - The tree is built from a latent per-cell coordinate:
        BRANCH A: latent = true position + noise    -> tree distance ~ spatial distance
                  the tree PREDICTS location  -> these cells should want HIGH alpha
        BRANCH B: latent = pure noise                -> tree says nothing about location
                  -> these cells should want LOW alpha
    tree_noise controls branch A's concordance; everything else is shared.

The two branches differ in ONE respect only: whether the tree is informative about position.
That is precisely what alpha_tau is supposed to model, so this is the cleanest possible test of
whether the method can express it.
"""
import numpy as np


def simulate(seed=0, grid=12, per_spot=8, tree_noise=0.10, sigma_expr=1.30, G=40,
             branch_frac=0.5, branch_offset=0.28):
    rng = np.random.default_rng(seed)

    gx, gy = np.meshgrid(np.linspace(0, 1, grid), np.linspace(0, 1, grid))
    xy_st = np.c_[gx.ravel(), gy.ravel()]
    m = len(xy_st)
    n = m * per_spot

    # every spot gets exactly per_spot cells -> uniform marginals are exactly correct
    true_spot = np.repeat(np.arange(m), per_spot)
    order = rng.permutation(n)
    true_spot = true_spot[order]
    xy_sc = xy_st[true_spot]

    nA = int(round(n * branch_frac))
    br = np.ones(n, dtype=int); br[rng.permutation(n)[:nA]] = 0

    # latent tree coordinates
    zc = rng.uniform(0, 1, (n, 2))                                  # branch B: pure noise
    zc[br == 0] = xy_sc[br == 0] + rng.normal(0, tree_noise, (int((br == 0).sum()), 2))
    # Branch separation is CALIBRATED to PEtracer, where the cross-branch / within-branch
    # mean patristic ratio is only 1.13x. An arbitrary large offset makes C_tree.mean()
    # cross-branch-dominated, squashes the within-branch signal, and the GW term then
    # compares incompatible scales -- which is a bug in the generator, not the method.
    zc[br == 1] += np.array([branch_offset, 0.0])
    CT = np.linalg.norm(zc[:, None] - zc[None, :], axis=-1)
    np.fill_diagonal(CT, 0.0)

    CS = np.linalg.norm(xy_st[:, None] - xy_st[None, :], axis=-1)

    W = rng.normal(size=(6, G))
    def field(P):
        return np.c_[P, P ** 2, P[:, :1] * P[:, 1:], np.ones((len(P), 1))] @ W
    X_st = field(xy_st)
    X_sc = field(xy_sc) + rng.normal(0, sigma_expr, (n, G))

    return dict(X_sc=X_sc, X_st=X_st, C_tree=CT, C_space=CS, xy_sc=xy_sc, xy_st=xy_st,
                branch=br, true_spot=true_spot, n=n, m=m)


if __name__ == "__main__":
    import ot
    from scipy.stats import pearsonr
    for tn in [0.05, 0.10, 0.20]:
        d = simulate(0, tree_noise=tn)
        line = [f"tree_noise={tn:.2f}  n={d['n']} m={d['m']}"]
        for b in (0, 1):
            s = np.where(d["branch"] == b)[0]
            iu = np.triu_indices(len(s), 1)
            ct = d["C_tree"][np.ix_(s, s)][iu]
            cs = np.linalg.norm(d["xy_sc"][s][:, None] - d["xy_sc"][s][None, :], axis=-1)[iu]
            line.append(f"branch {b} corr(tree,space) {pearsonr(ct, cs)[0]:+.3f}")
        M = ot.dist(d["X_sc"], d["X_st"])
        line.append(f"expr-NN acc {(M.argmin(1) == d['true_spot']).mean():.3f} "
                    f"(chance {1/d['m']:.3f})")
        print("  |  ".join(line))
