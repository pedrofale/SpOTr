"""The expression layer and the measurement (environment benchmark_env; numpy only).

Tissue.  The tumor cells at their sites from simulate.py; the non-tumor cells at the layout's other sites, with their types.
Expression.  Cell n's expected counts are library * softmax_g(eta_ng), with Poisson noise, and
    eta_ng = b_g + c_{type(n),g} + sigma_tree * (A h_n)_g + sigma_space * (B f(x_n))_g,
  b_g ~ N(0, 1)                        a baseline shared by all cells,
  c_{t,g} ~ N(0, 1)                    one shift per cell type (tumor and the three non-tumor types),
  h_n                                  for tumor cells, `programs` heritable programs that perform Brownian motion along the true tree, scaled to
                                       unit variance at the leaves (0 for non-tumor cells),
  f(x) = sqrt(2) cos(W x + u)          `programs` smooth random functions of position (random Fourier features, W ~ N(0, 1/l^2), u ~ U(0, 2 pi),
                                       l = length_scale x the section's diameter),
  A, B ~ N(0, 1/programs)              fixed random loadings of the programs on the genes.
Measurement.  The cells are split at random in two halves.  The section's half is measured as by Visium: spots of spot_diameter on a hexagonal
lattice of spot_spacing, a spot's counts the sum of the cells whose centres lie in it, the cells between spots not measured, the spots without
a cell dropped.  Of the other half, a share `capture` is sequenced as dissociated cells, and the lineage of a share `traced` of the sequenced
tumor cells is read.

Writes: Y (spots x genes counts), spots (positions), side and pitch (spot diameter and spacing), ncell (cells per spot), Ncl (spots x clones, the
true tumor cells of each clone), Nme (spots x 3, the true non-tumor cells of each type); for the dissociated cells Xc (counts), kind (0 tumor,
1 non-tumor), me_type, traced, clone, leaf (index of the tumor cells among the tree's leaves), xy_true; K (clones including the pooled one);
and every cell of the tissue: tissue_xy, tissue_type (0 tumor, 1-3 the non-tumor types), tissue_clone (-1 for non-tumor cells), tissue_section
(in the measured half).

Usage: python express.py <params.json> <seed> <sim.npz> <layout.npz> <out.npz>
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree


def visium(xy, sec, spacing, diameter):
    """the spots of a hexagonal lattice over the section that hold a cell of the measured half: (positions, measured cells, their spot)"""
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
    sec = rng.random(len(xy)) < 0.5; diss = ~sec                                                   # the section's half and the dissociated half
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
    print(f"spots {S} ({ncell.mean():.1f} cells each, {len(idx) / sec.sum():.0%} of the section's cells), dissociated {len(cap)} ({int(traced.sum())} traced), genes {G}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
