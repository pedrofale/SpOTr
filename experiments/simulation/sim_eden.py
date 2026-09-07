"""Tumour growth with volume exclusion: Eden growth on a lattice.

CALIBRATED against PEtracer (local_mix=0.45, tree_noise=2.0):
    statistic                    PEtracer            this
    clade coherence              0.59-0.66           0.526   (11% low)
    global corr(tree,space)      0.19-0.21           0.190
    local concordance            0.44-0.51           0.484
    kNN same-clade excess        +0.20 to +0.31      +0.236
    pairs at the tree maximum    28-49%              50%
    between-clade CV             0.040-0.043         0.172   (still ~4x high)
Three mechanisms are each necessary: volume-exclusion growth for territoriality, local mixing
for the weak fine-scale concordance, and tree reconstruction error for the low GLOBAL
correlation. Free diffusion cannot produce territoriality; growth alone cannot produce a low
global correlation. Only the combination matches.

WHY, over free diffusion. A branching process with Brownian displacement produces overlapping
Gaussian clouds: clades interpenetrate and no clone owns a territory. Real tumours grow under
volume exclusion -- a dividing cell must push into free space -- which produces compact clonal
SECTORS. That is what PEtracer shows (k-NN same-clade excess +0.306 on M3-4 against +0.135 for
free diffusion), and territoriality is exactly the feature that makes a lineage tree useful:
the tree encodes clade identity, and clade identity only predicts position if clades own space.

MODEL. One founder on a lattice. Repeatedly pick a random cell that has an empty neighbouring
site and divide it: one daughter stays, the other occupies the free site. Lineage distance is
the summed branch length back to the MRCA. Optional recorder saturation caps deep distances,
and local_mix adds post-hoc shuffling (motility/packing noise).
"""
import numpy as np

NBR2 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
NBR3 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)]


def _patristic(parent, blen, leaves, n_nodes):
    """Pairwise patristic distance among the GIVEN leaves (which may be a subset of all
    living cells -- a 2D section through a 3D tumour keeps only a slab). One pass over the
    internal nodes; nodes with no selected descendant contribute nothing."""
    kids = [[] for _ in range(n_nodes)]
    for v in range(1, n_nodes):
        kids[parent[v]].append(v)
    root_d = np.zeros(n_nodes)
    order = [0]
    for v in order:
        for c in kids[v]:
            root_d[c] = root_d[v] + blen[c]
            order.append(c)
    leaf_index = {l: i for i, l in enumerate(leaves)}
    n = len(leaves)
    D = np.zeros((n, n))
    sub = [None] * n_nodes
    for v in reversed(order):
        if not kids[v]:
            # a living cell outside the selected section contributes no pairs
            sub[v] = np.array([leaf_index[v]]) if v in leaf_index else np.empty(0, int)
        else:
            parts = [sub[c] for c in kids[v] if sub[c] is not None and len(sub[c])]
            if not parts:
                sub[v] = np.empty(0, int)
                for c in kids[v]:
                    sub[c] = None
                continue
            for a in range(len(parts)):
                for b in range(a + 1, len(parts)):
                    I, J = parts[a], parts[b]
                    d = (root_d[np.array(leaves)[I]][:, None]
                         + root_d[np.array(leaves)[J]][None, :] - 2 * root_d[v])
                    D[np.ix_(I, J)] = d
                    D[np.ix_(J, I)] = d.T
            sub[v] = np.concatenate(parts)
            for c in kids[v]:
                sub[c] = None
    return D


def _kd_split(xy, idx, depth):
    if depth == 0:
        return [idx]
    ax = depth % 2
    o = idx[np.argsort(xy[idx, ax], kind="stable")]
    h = len(o) // 2
    return _kd_split(xy, o[:h], depth - 1) + _kd_split(xy, o[h:], depth - 1)


def simulate(seed=0, n=1024, m_spots=128, n_clades=16, sigma_expr=1.30, G=40,
             bl_shape=8.0, sat_q=0.5, local_mix=0.45, jitter=0.30,
             dim=2, grow_mult=16, slab=3, tree_noise=2.0, group_mix=None):
    """dim=3 grows a 3D tumour and takes a thin slab, as a real section is. Clonal sectors
    survive the cut (territories preserved) but the lineage relationships among the cells that
    happen to lie in the slab are a thin subsample of the 3D tree, which is what pulls the
    GLOBAL tree-space correlation down to the observed 0.19-0.21 while 2D growth sits at 0.33+."""
    rng = np.random.default_rng(seed)
    NBR = NBR2 if dim == 2 else NBR3
    origin = (0, 0) if dim == 2 else (0, 0, 0)
    n_grow = n if dim == 2 else n * grow_mult
    occ = {origin: 0}
    pos = {0: origin}
    parent = [-1]; blen = [0.0]
    alive = {0}                                    # node ids of living cells
    frontier = [0]                                 # candidates; may hold stale entries

    def free_sites(p):
        out = []
        for d_ in NBR:
            q = tuple(p[i] + d_[i] for i in range(dim))
            if q not in occ:
                out.append(q)
        return out

    while len(alive) < n_grow:
        while True:                                # drop stale / interior candidates
            fi = int(rng.integers(len(frontier)))
            v = frontier[fi]
            fs = free_sites(pos[v]) if v in alive else None
            if fs:
                break
            frontier[fi] = frontier[-1]; frontier.pop()
        site = fs[int(rng.integers(len(fs)))]
        b = rng.gamma(bl_shape, 1.0 / bl_shape, 2)
        c1, c2 = len(parent), len(parent) + 1
        parent += [v, v]; blen += [float(b[0]), float(b[1])]
        pos[c1] = pos[v]; pos[c2] = site
        occ[site] = c2
        alive.discard(v); alive.add(c1); alive.add(c2)
        frontier += [c1, c2]

    living = sorted(alive)
    P = np.array([pos[l] for l in living], float)
    if dim == 3:                                    # cut a thin slab: this is the section
        z = P[:, 2] - np.median(P[:, 2])
        keep = np.where(np.abs(z) <= slab / 2.0)[0]
        if len(keep) < n:
            raise RuntimeError(f"slab holds {len(keep)} cells, need {n}; raise grow_mult")
        keep = rng.choice(keep, n, replace=False)
        living = [living[i] for i in keep]
        P = P[keep][:, :2]
    leaves = living
    XY = P + rng.normal(0, jitter, P.shape)
    CT = _patristic(parent, np.array(blen), leaves, len(parent))

    # Clades and branches are derived from the patristic matrix by average-linkage fcluster --
    # the SAME procedure applied to PEtracer, so the two datasets are treated identically.
    # (Cutting at a fixed node depth fails here: Eden growth gives a Yule-like tree whose leaf
    # depths span 2-42, so a depth cut produces wildly lopsided groups.)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(CT, checks=False), "average")
    clade = fcluster(Z, n_clades, "maxclust") - 1
    branch = fcluster(Z, 2, "maxclust") - 1

    # PER-GROUP MOTILITY. The paper's own mechanism: "cells from another clade with more
    # motility may have diverged more from their initial locations after cell division." A
    # low-motility group stays where the lineage put it, so its tree predicts position and it
    # should want a HIGH alpha; a high-motility group is scattered and should want a LOW one.
    #
    # The groups are BALANCED UNIONS OF CLADES, not the tree's root split. The root split of an
    # Eden tree is badly unbalanced (305 vs 719) and its geometry alone shifts the local
    # concordance statistic by -0.33 with motility held equal -- a confound that would sit
    # underneath any planted effect. Balanced clade unions are still tree-coherent (so the
    # Omega mask is well defined) but carry no such baseline asymmetry.
    sizes = np.bincount(clade)
    grp_of_clade = np.zeros(len(sizes), int)
    tot = [0, 0]
    for k in np.argsort(-sizes):                      # greedy balance, largest first
        g = int(tot[0] > tot[1]); grp_of_clade[k] = g; tot[g] += sizes[k]
    mixgrp = grp_of_clade[clade]
    if group_mix is not None:
        lm = np.where(mixgrp == 0, group_mix[0], group_mix[1])
    else:
        lm = np.full(len(leaves), local_mix)
    if np.any(lm > 0):
        XY = XY + rng.normal(0, 1.0, XY.shape) * (lm[:, None] * XY.std())

    # TREE RECONSTRUCTION ERROR, applied AFTER clades are fixed. A lineage recorder resolves
    # clade MEMBERSHIP reliably but its branch-length estimates are noisy, so the inferred
    # metric is close to categorical: it says which clade a cell is in, not how far apart two
    # cells are. This is the one mechanism that lowers the GLOBAL tree-space correlation
    # (0.42 -> 0.10) while leaving LOCAL concordance (0.53 -> 0.48) and territoriality intact,
    # which is the combination PEtracer shows and that growth alone cannot produce.
    if tree_noise > 0:
        E = rng.normal(0, tree_noise * CT[CT > 0].mean(), CT.shape)
        E = np.abs((E + E.T) / 2.0); np.fill_diagonal(E, 0.0)
        CT = CT + E

    if sat_q is not None:
        cap = float(np.quantile(CT[np.triu_indices(len(leaves), 1)], 1.0 - sat_q))
        CT = np.minimum(CT, cap)

    leaves_idx = _kd_split(XY, np.arange(len(leaves)), int(np.log2(m_spots)))
    true_spot = np.empty(len(leaves), int); xy_st = np.empty((m_spots, 2))
    for j, idx in enumerate(leaves_idx):
        true_spot[idx] = j; xy_st[j] = XY[idx].mean(0)

    W = rng.normal(size=(6, G))
    def field(P):
        Pn = (P - XY.mean(0)) / (XY.std(0) + 1e-9)
        return np.c_[Pn, Pn**2, Pn[:, :1]*Pn[:, 1:], np.ones((len(Pn), 1))] @ W
    X_st = field(xy_st)
    X_sc = field(xy_st[true_spot]) + rng.normal(0, sigma_expr, (len(leaves), G))
    CS = np.linalg.norm(xy_st[:, None] - xy_st[None, :], axis=-1)

    return dict(X_sc=X_sc, X_st=X_st, C_tree=CT, C_space=CS, xy_sc=xy_st[true_spot],
                xy_st=xy_st, pos=XY, branch=branch, clade=clade, mixgrp=mixgrp,
                true_spot=true_spot,
                n=len(leaves), m=m_spots)


