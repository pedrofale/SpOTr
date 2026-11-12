"""The expression and the measurement of a simulated tumor (environment benchmark_env, numpy only).

Tissue. The tumor cells are at their sites from simulate.py, and the non-tumor cells at the other sites of the layout, with their types.
Expression. We simulate the counts of cell n from Poisson distributions with means library * softmax_g(eta_ng), where
    eta_ng = b_g + c_{type(n),g} + sigma_tree * (A h_n)_g + sigma_space * (B f(x_n))_g,
  b_g ~ N(0, 1)                        is a baseline shared by all cells,
  c_{t,g} ~ N(0, 1)                    is a shift per cell type (tumor and the three non-tumor types),
  h_n                                  are `programs` heritable programs of the tumor cells, which follow Brownian motion along the true tree and
                                       are scaled to unit variance at the leaves (0 for non-tumor cells),
  f(x) = sqrt(2) cos(W x + u)          are `programs` smooth random functions of position (random Fourier features, with W ~ N(0, 1/l^2), u ~
                                       U(0, 2 pi) and l = length_scale x the diameter of the section),
  A, B ~ N(0, 1/programs)              are fixed random loadings of the programs on the genes.
Measurement. We split the cells at random into two halves and measure the half of the section as by Visium, with spots of spot_diameter on a
hexagonal lattice of spot_spacing. The counts of a spot are the sum of the counts of the cells whose centers are within it, the cells between
spots are not measured, and the spots without a cell are dropped. We sequence a fraction `capture` of the other half as dissociated cells, and
read the lineage of a fraction `traced` of the sequenced tumor cells.

Writes Y (counts, spots x genes), spots (positions), side and pitch (spot diameter and spacing), ncell (cells per spot), Ncl (true tumor cells of
each clone, spots x clones) and Nme (true non-tumor cells of each type, spots x 3). For the dissociated cells, it writes Xc (counts), kind (0
tumor, 1 non-tumor), me_type, traced, clone, leaf (index of the tumor cells among the leaves of the tree) and xy_true, and K (the number of
clones including the pooled one). For every cell of the tissue, it writes tissue_xy, tissue_type (0 tumor, 1-3 the non-tumor types), tissue_clone
(-1 for non-tumor cells) and tissue_section (in the measured half).

Usage: python express.py <params.json> <seed> <sim.npz> <layout.npz> <out.npz>
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree


def visium(xy, sec, spacing, diameter):
    """Return the spots of a hexagonal lattice over the section that contain a cell of the measured half, as (positions, measured cells, the spot of
    each measured cell).
    """
    a, dm = spacing, diameter; lo, hi = xy.min(0), xy.max(0); rows = np.arange(lo[1] + a / 4, hi[1], a * np.sqrt(3) / 2)
    grid = np.array([(x, y) for i, y in enumerate(rows) for x in np.arange(lo[0] + a / 4 + (a / 2) * (i % 2), hi[0], a)])
    dist, near = cKDTree(grid).query(xy[sec]); hit = dist <= dm / 2; idx = np.where(sec)[0][hit]
    used, inv = np.unique(near[hit], return_inverse=True); return grid[used], idx, inv.ravel()


def main(pfile, seed, sim, layout, out):
    p = json.load(open(pfile)); rng = np.random.default_rng(int(seed) + 7); z = np.load(sim)
    xy_t, clone_t, K = z["xy"], z["clone"], int(z["K"]); n_t = len(xy_t); G, P = int(p["genes"]), int(p["programs"])
    parent, time = z["parent"], z["time"]; h = np.zeros((len(parent), P))                         # heritable programs along the true tree
    for v in range(len(parent)):
        if parent[v] >= 0:
            h[v] = h[parent[v]] + rng.normal(0.0, np.sqrt(max(time[v] - time[parent[v]], 0.0) / float(z["depth"])), P)
    H = h[z["leaf_node"]]; H = H / max(H.std(), 1e-12)
    L = np.load(layout); lt = L["type"]; xy_me = L["xy"][lt > 0]; me_type = lt[lt > 0] - 1; n_me = len(xy_me)
    xy = np.concatenate([xy_t, xy_me]); kind = np.r_[np.zeros(n_t, int), np.ones(n_me, int)]; ctype = np.r_[np.zeros(n_t, int), 1 + me_type]
    b = rng.normal(0, 1, G); c = rng.normal(0, 1, (4, G)); A = rng.normal(0, 1 / np.sqrt(P), (G, P)); Bm = rng.normal(0, 1 / np.sqrt(P), (G, P))
    diam = float(np.ptp(xy, 0).max()); W = rng.normal(0, 1 / (float(p["length_scale"]) * diam), (P, 2)); u = rng.uniform(0, 2 * np.pi, P)
    F = np.sqrt(2) * np.cos(xy @ W.T + u); Hall = np.r_[H, np.zeros((n_me, P))]
    eta = b[None, :] + c[ctype] + float(p["sigma_tree"]) * Hall @ A.T + float(p["sigma_space"]) * F @ Bm.T
    pr = np.exp(eta - eta.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True); X = rng.poisson(float(p["library"]) * pr)
    sec = rng.random(len(xy)) < 0.5; diss = ~sec                                                   # the half of the section and the dissociated half
    spots, idx, inv = visium(xy, sec, float(p["spot_spacing"]), float(p["spot_diameter"]))
    S = len(spots); Y = np.zeros((S, G), int); np.add.at(Y, inv, X[idx]); ncell = np.bincount(inv, minlength=S)
    tum = kind[idx] == 0; Ncl = np.zeros((S, K)); np.add.at(Ncl, inv[tum], np.eye(K)[clone_t[idx[tum]]])
    Nme = np.zeros((S, 3)); np.add.at(Nme, inv[~tum], np.eye(3)[ctype[idx[~tum]] - 1])
    cap = np.where(diss)[0]; cap = cap[rng.random(len(cap)) < float(p["capture"])]
    traced = (kind[cap] == 0) & (rng.random(len(cap)) < float(p["traced"]))
    np.savez_compressed(out, Y=Y, spots=spots, side=float(p["spot_diameter"]), pitch=float(p["spot_spacing"]), ncell=ncell, Ncl=Ncl, Nme=Nme, Xc=X[cap],
                        kind=kind[cap], me_type=np.where(kind[cap] == 1, ctype[cap] - 1, -1), traced=traced,
                        clone=np.where(kind[cap] == 0, clone_t[np.where(kind[cap] == 0, cap, 0)], -1), leaf=np.where(kind[cap] == 0, cap, -1), xy_true=xy[cap], K=K,
                        tissue_xy=xy, tissue_type=ctype, tissue_clone=np.r_[clone_t, -np.ones(n_me, int)], tissue_section=sec)
    print(f"spots {S} ({ncell.mean():.1f} cells each, {len(idx) / sec.sum():.0%} of the cells of the section), dissociated {len(cap)} ({int(traced.sum())} traced), genes {G}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
