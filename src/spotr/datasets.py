"""A toy tumor for trying SpOTr (numpy only): the setting SpOTr is designed for, in which the clones are hard to distinguish by their expression but each
holds its own territory (the paper's simulations, without Cassiopeia or scCube).

    sc_adata, st_adata, tree, truth = spotr.datasets.toy_tumor(seed=0)

Lineage: a Yule tree over the tumor cells; clones: the clades at the earliest time at which n_clones clades hold at least 3% of the cells.
Tissue: cells on a jittered hexagonal grid 12 um apart; the tumor a blob in the centre, cut along the tree into nested territories (each clade a
contiguous part of its parent's, along the principal axis turned by a random angle), the rest stroma with immune aggregates, and some immune
cells infiltrating the tumor.  Expression: a baseline, a shift per cell type, and `programs` heritable programs that evolve along the tree as
Brownian motion with strength clone_signal; Poisson counts of `library` per cell.  Measurement: one random half of the cells under Visium-like
spots (55 um, 100 um apart on a hexagonal lattice); of the other half, a share `capture` is sequenced, and the lineage of a share `traced` of its
tumor cells is read.
Returns sc_adata (counts; obs cell_type and clone, NaN for cells without a lineage; obsm spatial_true), st_adata (counts; obsm spatial), the
tree over the traced dissociated cells (Newick, leaves named by obs_names) and the truth (each spot's number of cells of each clone and type).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import tree as _tree


def _yule(n_leaves, rng):
    """parent pointers (pre-order after sorting), branch lengths and the leaves of a Yule tree"""
    parent, birth = [-1], [0.0]; leaves = [0]; t = 0.0
    while len(leaves) < n_leaves:
        t += rng.exponential(1.0 / len(leaves)); i = int(rng.integers(len(leaves))); v = leaves[i]
        a, b = len(parent), len(parent) + 1; parent += [v, v]; birth += [t, t]; leaves[i] = a; leaves.append(b)
    T = t + rng.exponential(1.0 / len(leaves)); death = np.full(len(parent), T)
    for v, p in enumerate(parent):
        if p >= 0:
            death[p] = min(death[p], birth[v])
    length = np.array([death[v] - (death[parent[v]] if parent[v] >= 0 else 0.0) for v in range(len(parent))])
    return np.array(parent), length, np.array(leaves), death


def _territories(parent, leaf_node, sites, rng, jitter=1.0):
    """the site of each leaf: each clade a contiguous part of its parent's sites, cut along their principal axis turned by N(0, jitter)"""
    kids = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p >= 0:
            kids[p].append(v)
    leaf_of = {int(l): i for i, l in enumerate(leaf_node)}; size = np.zeros(len(parent), int)
    for v in range(len(parent) - 1, -1, -1):                                                    # children have larger indices than parents
        size[v] = 1 if v in leaf_of else sum(size[c] for c in kids[v])
    site = np.zeros(len(leaf_node), int); root = int(np.where(parent < 0)[0][0]); stack = [(root, np.arange(len(sites)))]
    while stack:
        v, S = stack.pop()
        if v in leaf_of:
            site[leaf_of[v]] = S[0]; continue
        cs = [c for c in kids[v] if size[c] > 0]
        X = sites[S] - sites[S].mean(0); u = np.linalg.svd(X, full_matrices=False)[2][0]; th = rng.normal(0.0, jitter)
        u = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ u; o = S[np.argsort(X @ u)]; rng.shuffle(cs); k = 0
        for c in cs:
            stack.append((c, o[k:k + size[c]])); k += size[c]
    return site


def _clones(parent, ntime, leaf_node, n_clones, min_share=0.03):
    """the clades at the earliest cut where n_clones clades hold at least min_share of the leaves (the n_clones largest), the rest -1"""
    n = len(leaf_node); root = int(np.where(parent < 0)[0][0]); internal = np.bincount(parent[parent >= 0], minlength=len(parent)) > 0
    for cut in np.sort(np.unique(ntime[internal])):
        v = leaf_node.copy()
        while True:                                                                             # each leaf's highest ancestor below the cut
            u = parent[v]; mv = (u >= 0) & (u != root) & (ntime[np.maximum(u, 0)] >= cut)
            if not mv.any():
                break
            v[mv] = u[mv]
        anc, cnt = np.unique(v, return_counts=True); big = anc[cnt >= min_share * n]
        if len(big) >= n_clones:
            big = anc[np.argsort(-cnt)][:n_clones]; lab = np.full(n, -1)
            for k, b in enumerate(big):
                lab[v == b] = k
            return lab
    raise ValueError("the tree has fewer than n_clones large clades")


def _hex(n_x, n_y, spacing, rng, jitter=0.15):
    pts = np.array([((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3) / 2) for j in range(n_y) for i in range(n_x)])
    return pts + rng.normal(0.0, jitter * spacing, pts.shape)


def toy_tumor(n_clones: int = 5, clone_signal: float = 0.1, n_genes: int = 300, programs: int = 5, library: float = 1000.0, side_um: float = 1300.0,
              cell_spacing: float = 12.0, tumor_share: float = 0.5, capture: float = 0.2, traced: float = 0.8, jitter: float = 1.0, seed: int = 0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    # the tissue
    n_side = int(side_um / cell_spacing); xy = _hex(n_side, int(n_side * 2 / np.sqrt(3)), cell_spacing, rng); xy = xy[(xy[:, 1] < side_um)]
    c = xy.mean(0); r = np.linalg.norm(xy - c, axis=1); th = np.arctan2(*(xy - c).T[::-1])
    edge = np.quantile(r, tumor_share) * (1 + 0.12 * np.sin(3 * th + 1.0) + 0.08 * np.cos(5 * th))       # a lobed blob
    tumor = r < edge
    agg = c + rng.normal(0, 1, (4, 2)) * 0.0 + np.array([[np.cos(a), np.sin(a)] for a in rng.uniform(0, 2 * np.pi, 4)]) * np.quantile(r, 0.8)
    near_agg = np.min(np.linalg.norm(xy[:, None] - agg[None], axis=2), axis=1) < 60
    ctype = np.where(tumor, "tumor", np.where(near_agg, "immune", "stroma")).astype(object)
    infil = tumor & (rng.random(len(xy)) < 0.05); ctype[infil] = "immune"; tumor = ctype == "tumor"
    # the lineage and the tumor's territories
    parent, length, leaf_node, death = _yule(int(tumor.sum()), rng)
    pos = _territories(parent, leaf_node, xy[tumor], rng, jitter)                                  # each leaf's tumor site
    tumor_idx = np.where(tumor)[0]; leaf_of_cell = np.full(len(xy), -1); leaf_of_cell[tumor_idx[pos]] = np.arange(len(leaf_node))
    clone_of_leaf = _clones(parent, death, leaf_node, n_clones)
    # expression
    depth = np.zeros(len(parent)); h = np.zeros((len(parent), programs))
    for v in range(len(parent)):
        if parent[v] >= 0:
            depth[v] = depth[parent[v]] + length[v]; h[v] = h[parent[v]] + rng.normal(0.0, np.sqrt(max(length[v], 0.0)), programs)
    H = h[leaf_node]; H = (H - H.mean(0)) / max(H.std(), 1e-12)
    types = ["tumor", "stroma", "immune"]; b = rng.normal(0, 1, n_genes); shift = {t: rng.normal(0, 1, n_genes) for t in types}
    A = rng.normal(0, 1 / np.sqrt(programs), (n_genes, programs))
    eta = b[None, :] + np.stack([shift[t] for t in ctype])
    eta[tumor_idx[pos]] += clone_signal * H @ A.T
    p = np.exp(eta - eta.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); X = rng.poisson(library * p)
    genes = [f"g{i}" for i in range(n_genes)]
    # measurement: spots over one half, a captured dissociated sample of the other
    sec = rng.random(len(xy)) < 0.5; a_, d_ = 100.0, 55.0
    grid = _hex(int(side_um / a_) + 1, int(side_um / (a_ * np.sqrt(3) / 2)) + 1, a_, rng, jitter=0.0) + a_ / 4
    from scipy.spatial import cKDTree
    dist, near = cKDTree(grid).query(xy[sec]); hit = dist <= d_ / 2; idx = np.where(sec)[0][hit]
    used, inv = np.unique(near[hit], return_inverse=True); spots = grid[used]; M = len(spots)
    Y = np.zeros((M, n_genes), int); np.add.at(Y, inv, X[idx])
    cl_cell = np.full(len(xy), -1); cl_cell[tumor_idx[pos]] = clone_of_leaf
    clone_names = [f"clone {k + 1}" for k in range(n_clones)]
    counts = pd.DataFrame(0, index=[f"spot{i}" for i in range(M)], columns=clone_names + ["smaller clones"] + types[1:])
    for s, cell in zip(inv, idx):
        if ctype[cell] == "tumor":
            counts.iloc[s, cl_cell[cell] if cl_cell[cell] >= 0 else n_clones] += 1
        else:
            counts.loc[counts.index[s], ctype[cell]] += 1
    diss = np.where(~sec)[0]; diss = diss[rng.random(len(diss)) < capture]
    has = (ctype[diss] == "tumor") & (rng.random(len(diss)) < traced)
    clone_obs = np.array([clone_names[cl_cell[c]] if (h_ and cl_cell[c] >= 0) else (("smaller clones" if h_ else np.nan)) for c, h_ in zip(diss, has)], dtype=object)
    names = [f"cell{i}" for i in range(len(diss))]
    sc_adata = ad.AnnData(X[diss].astype(np.float32), obs=pd.DataFrame({"cell_type": pd.Categorical(ctype[diss].astype(str)), "clone": clone_obs}, index=names))
    sc_adata.var_names = genes; sc_adata.obsm["spatial_true"] = xy[diss]
    st_adata = ad.AnnData(Y.astype(np.float32), obs=pd.DataFrame(index=counts.index)); st_adata.var_names = genes; st_adata.obsm["spatial"] = spots
    # the tree over the traced dissociated cells, leaves named by obs_names
    keep = {int(leaf_node[leaf_of_cell[c]]): nm for c, nm, h_ in zip(diss, names, has) if h_}
    tnames, tpar, tlen = _tree.subtree(parent, length, keep)
    truth = dict(clone_counts=counts, cell_xy=xy, cell_type=ctype.astype(str), section=sec)
    return sc_adata, st_adata, _tree.to_newick(tpar, tlen, tnames), truth
