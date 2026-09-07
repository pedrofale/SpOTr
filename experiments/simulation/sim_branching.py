"""A REALISTIC generator: cells divide, daughters displace, lineage distance saturates.

WHY THE PREVIOUS GENERATOR WAS NOT A FAIR TEST OF THE COST
   tree = position + Gaussian noise  makes GLOBAL (metric) and LOCAL (neighbourhood)
   concordance rise and fall together, so it cannot distinguish a cost that reads one from a
   cost that reads the other. Real lineage DECOUPLES them: daughters stay near parents (strong
   local) but patristic distance saturates a few divisions back (no global metric relation).

THIS GENERATOR
   A spatial branching process. One founder; each generation every cell divides and each
   daughter displaces by N(0, sigma) from its parent. Patristic distance is the sum of branch
   lengths back to the MRCA, with lognormal branch lengths so the metric is not perfectly
   discrete. After g generations from an MRCA, spatial separation grows like sigma*sqrt(2g)
   while tree distance grows like 2g -- a CONCAVE relation, which is exactly what flattens the
   global correlation while keeping neighbours close.

   Per-clade motility (sigma_k) is the biologically natural way to plant per-clade alpha:
   a low-motility clade stays where it was born, a high-motility one scatters. That is the
   mechanism the paper's own text describes.

SPOTS. Cells are partitioned into m spatially contiguous groups of EXACTLY n/m cells by
recursive median (k-d) splits, so the uniform marginals the solver assumes are exactly true
while the spatial layout stays faithful. n and m must be powers of two.
"""
import numpy as np


def _kd_split(xy, idx, depth):
    if depth == 0:
        return [idx]
    ax = depth % 2
    o = idx[np.argsort(xy[idx, ax], kind="stable")]
    h = len(o) // 2
    return _kd_split(xy, o[:h], depth - 1) + _kd_split(xy, o[h:], depth - 1)


# Defaults calibrated against PEtracer: clade coherence 0.659 (target 0.59-0.66),
# local concordance 0.471 (0.44-0.51), global 0.256 (0.19-0.21), 50% of pairs at the
# tree maximum (49% on M3-4). Between-clade CV is 0.078 against 0.043 -- still ~2x high.
def simulate(seed=0, n_gen=10, m_spots=128, sigma=0.55, clade_depth=4,
             clade_sigma=None, sigma_expr=1.30, G=40, bl_shape=8.0, sat_q=None,
             sigma_decay=0.85, local_mix=0.40):
    rng = np.random.default_rng(seed)
    n = 2 ** n_gen

    pos = np.zeros((1, 2))
    depth_of = np.zeros(1, int)
    # path[i] = the sequence of child choices from the root, used for MRCA depth and clades
    path = np.zeros((1, 0), dtype=np.int8)
    blen = np.zeros((1, 0))                       # branch length taken at each generation

    for g in range(n_gen):
        k = len(pos)
        # clade identity is fixed once we pass clade_depth; motility can depend on it
        if clade_sigma is not None and g >= clade_depth:
            cl = np.array([int("".join(map(str, p)), 2) for p in path[:, :clade_depth]])
            s_ = clade_sigma[cl % len(clade_sigma)]
        else:
            s_ = np.full(k, sigma)
        # Displacement shrinks with generation: clones separate early, then pack. Without this
        # late diffusion washes clade territories out (coherence 0.80 against PEtracer's 0.59-0.66).
        s_ = s_ * (sigma_decay ** g)
        b = rng.gamma(bl_shape, 1.0 / bl_shape, size=(k, 2))      # mean 1, variable
        step = rng.normal(0, 1.0, size=(k, 2, 2)) * (s_[:, None, None] * np.sqrt(b)[:, :, None])
        pos = np.concatenate([pos + step[:, 0], pos + step[:, 1]])
        path = np.concatenate([np.c_[path, np.zeros(k, np.int8)],
                               np.c_[path, np.ones(k, np.int8)]])
        blen = np.concatenate([np.c_[blen, b[:, 0]], np.c_[blen, b[:, 1]]])
        depth_of = np.concatenate([depth_of + 1, depth_of + 1])

    # LOCAL MIXING. After the lineage is set, cells shuffle locally within their clone's
    # territory (motility, packing, growth pressure). This decouples FINE-scale lineage from
    # fine-scale position while leaving clade territories intact -- which is what PEtracer looks
    # like: coherent clades (0.59-0.66) but weak local concordance (0.44-0.51). Without it the
    # generator cannot produce both at once.
    if local_mix > 0:
        pos = pos + rng.normal(0, local_mix * pos.std(), pos.shape)

    # patristic distance = sum of branch lengths on both paths back to the MRCA
    eq = path[:, None, :] == path[None, :, :]
    mrca = np.argmin(np.cumprod(eq, axis=2), axis=2)              # first differing generation
    allsame = eq.all(2)
    mrca = np.where(allsame, n_gen, mrca)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(blen, 1)], 1)
    tot = cum[:, -1]
    CT = (tot[:, None] - cum[np.arange(n)[:, None], mrca]) + \
         (tot[None, :] - cum[np.arange(n)[None, :], mrca])
    np.fill_diagonal(CT, 0.0)

    # RECORDER SATURATION. A finite lineage recorder cannot resolve relationships beyond a
    # certain depth: once the barcode is exhausted, every deeper pair gets the same maximum
    # distance. PEtracer shows exactly this -- 49% of M3-4 pairs sit at exactly 70.0. Capping
    # at the sat_q quantile reproduces it. sat_q=None leaves the tree unsaturated, which is the
    # control that isolates what saturation costs the method.
    if sat_q is not None:
        cap = float(np.quantile(CT[np.triu_indices(n, 1)], 1.0 - sat_q))
        CT = np.minimum(CT, cap)

    clade = np.array([int("".join(map(str, p)), 2) for p in path[:, :clade_depth]])
    branch = path[:, 0].astype(int)

    # spots: equal-size, spatially contiguous
    leaves = _kd_split(pos, np.arange(n), int(np.log2(m_spots)))
    true_spot = np.empty(n, int)
    xy_st = np.empty((m_spots, 2))
    for j, idx in enumerate(leaves):
        true_spot[idx] = j
        xy_st[j] = pos[idx].mean(0)

    W = rng.normal(size=(6, G))
    def field(P):
        Pn = (P - pos.mean(0)) / (pos.std(0) + 1e-9)
        return np.c_[Pn, Pn ** 2, Pn[:, :1] * Pn[:, 1:], np.ones((len(Pn), 1))] @ W
    X_st = field(xy_st)
    X_sc = field(xy_st[true_spot]) + rng.normal(0, sigma_expr, (n, G))

    CS = np.linalg.norm(xy_st[:, None] - xy_st[None, :], axis=-1)
    return dict(X_sc=X_sc, X_st=X_st, C_tree=CT, C_space=CS, xy_sc=xy_st[true_spot],
                xy_st=xy_st, branch=branch, clade=clade, true_spot=true_spot,
                n=n, m=m_spots, pos=pos)
