"""The expression layer and the measurement (run in the spotr environment; numpy only).

Expression.  Cell n's expected counts are L * softmax_g(eta_ng) with Poisson noise, and
    eta_ng = b_g + c_{type(n),g} + sigma_tree * (A h_n)_g + sigma_space * (B f(x_n))_g,
  b_g ~ N(0, 1)                        a baseline shared by all cells,
  c_{t,g} ~ N(0, 1)                    one shift per cell type (tumour and three microenvironment types),
  h_n                                  for tumour cells, `programs` heritable programs that perform Brownian motion along the true tree,
                                       scaled so that their variance at the leaves is 1 (0 for microenvironment cells),
  f(x) = sqrt(2) cos(W x + u)          `programs` smooth random functions of position (random Fourier features, W ~ N(0, 1/l^2), u ~ U(0, 2 pi),
                                       l = length_scale x the section's diameter),
  A, B ~ N(0, 1/programs)              fixed random loadings of the programs on the genes.
Tissue.  The tissue layout (layout.py, scCube): the tumour cells' positions from Cassiopeia's spatial simulator are moved onto the layout's
tumour sites by exact optimal transport (squared Euclidean, both clouds centred and scaled to unit root-mean-square radius), which keeps their
arrangement; the microenvironment cells (fibroblasts, T cells, endothelial cells) are the layout's other cells, at its positions.
Measurement.  The cells are split at random in two halves.  The section's half is measured as by Visium (spots: visium): spots of
spot_diameter (55 um) on a hexagonal lattice of spot_spacing (100 um), a spot's counts the sum of the cells whose centres lie in it, the cells
between spots not measured, the spots without a cell dropped; or (spots: square, the runs until 2026-11-02) pooled into square bins that tile the
section, of side such that a bin holds cells_per_spot cells on average; of the other half, a share `capture` is sequenced, and the
barcodes of a share `traced` of the sequenced tumour cells are read.

Usage: python express.py <params.json> <seed> <sim.npz> <layout.npz> <out.npz>
"""
import sys, json
import numpy as np


def main(pfile, seed, sim, layout, out):
    p = json.load(open(pfile)); rng = np.random.default_rng(int(seed) + 7); z = np.load(sim)
    xy_t, clone_t, K = z["xy"], z["clone"], int(z["K"]); n_t = len(xy_t); G, P = int(p["genes"]), int(p["programs"])
    # heritable programs along the true tree (nodes are in pre-order: a parent before its children)
    parent, time = z["parent"], z["time"]; h = np.zeros((len(parent), P))
    for v in range(len(parent)):
        if parent[v] >= 0:
            h[v] = h[parent[v]] + rng.normal(0.0, np.sqrt(max(time[v] - time[parent[v]], 0.0) / float(z["depth"])), P)
    H = h[z["leaf_node"]]; H = H / max(H.std(), 1e-12)
    # the tissue: tumour cells onto the layout's tumour sites (exact OT, arrangement kept), the microenvironment from the layout
    import ot
    L = np.load(layout); lt = L["type"]; site = L["xy"][lt == 0]; assert len(site) == n_t, (len(site), n_t)
    nz = lambda a: (a - a.mean(0)) / np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()); w = np.full(n_t, 1.0 / n_t)
    xy_t = site[np.asarray(ot.emd(w, w, ot.dist(nz(xy_t), nz(site)), numItermax=10 ** 8)).argmax(1)]
    xy_me = L["xy"][lt > 0]; me_type = lt[lt > 0] - 1; n_me = len(xy_me)
    xy = np.concatenate([xy_t, xy_me]); kind = np.r_[np.zeros(n_t, int), np.ones(n_me, int)]; ctype = np.r_[np.zeros(n_t, int), 1 + me_type]
    # expression
    b = rng.normal(0, 1, G); c = rng.normal(0, 1, (4, G)); A = rng.normal(0, 1 / np.sqrt(P), (G, P)); Bm = rng.normal(0, 1 / np.sqrt(P), (G, P))
    diam = float(np.ptp(xy, 0).max()); W = rng.normal(0, 1 / (float(p["length_scale"]) * diam), (P, 2)); u = rng.uniform(0, 2 * np.pi, P)
    F = np.sqrt(2) * np.cos(xy @ W.T + u); Hall = np.r_[H, np.zeros((n_me, P))]
    eta = b[None, :] + c[ctype] + float(p["sigma_tree"]) * Hall @ A.T + float(p["sigma_space"]) * F @ Bm.T
    pr = np.exp(eta - eta.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True); X = rng.poisson(float(p["library"]) * pr)
    # measurement: the section's half of the cells under the spots, the other half dissociated
    sec = rng.random(len(xy)) < 0.5; diss = ~sec
    if p.get("spots", "square") == "visium":                                                    # spots of spot_diameter on a hexagonal lattice of
        from scipy.spatial import cKDTree                                                          # spot_spacing; cells between spots are not measured
        a, dm = float(p["spot_spacing"]), float(p["spot_diameter"]); lo, hi = xy.min(0), xy.max(0); rows = np.arange(lo[1] + a / 4, hi[1], a * np.sqrt(3) / 2)
        grid = np.array([(x, y) for i, y in enumerate(rows) for x in np.arange(lo[0] + a / 4 + (a / 2) * (i % 2), hi[0], a)])
        dist, near = cKDTree(grid).query(xy[sec]); hit = dist <= dm / 2; idx = np.where(sec)[0][hit]
        used, inv = np.unique(near[hit], return_inverse=True); inv = inv.ravel(); spots = grid[used]; side, shape = dm, "circle"
    else:                                                                                          # square bins of about cells_per_spot cells that
        from scipy.spatial import ConvexHull                                                       # tile the section
        area = ConvexHull(xy).volume; side = np.sqrt(float(p["cells_per_spot"]) * area / sec.sum()); idx = np.where(sec)[0]
        g = np.floor((xy[idx] - xy.min(0)) / side).astype(int); keys, inv = np.unique(g, axis=0, return_inverse=True); inv = inv.ravel()
        spots = xy.min(0) + (keys + 0.5) * side; a, shape = side, "square"
    S = len(spots); Y = np.zeros((S, G), int); np.add.at(Y, inv, X[idx]); ncell = np.bincount(inv, minlength=S)
    tum = kind[idx] == 0; Ncl = np.zeros((S, K)); np.add.at(Ncl, inv[tum], np.eye(K)[clone_t[idx[tum]]])
    Nme = np.zeros((S, 3)); np.add.at(Nme, inv[~tum], np.eye(3)[ctype[idx[~tum]] - 1])
    cap = np.where(diss)[0]; cap = cap[rng.random(len(cap)) < float(p["capture"])]
    traced = (kind[cap] == 0) & (rng.random(len(cap)) < float(p["traced"]))
    leaf = np.where(kind[cap] == 0, cap, -1)                                                        # index into the tree's leaves (tumour cells)
    np.savez_compressed(out, Y=Y, spots=spots, side=side, pitch=a, shape=shape, ncell=ncell, Ncl=Ncl, Nme=Nme, Xc=X[cap], kind=kind[cap], me_type=np.where(kind[cap] == 1, ctype[cap] - 1, -1),
                        traced=traced, clone=np.where(kind[cap] == 0, clone_t[np.where(kind[cap] == 0, cap, 0)], -1), leaf=leaf, xy_true=xy[cap], K=K)
    print(f"spots {S} ({shape}, {side:.0f} um, {ncell.mean():.1f} cells, {len(idx) / sec.sum():.0%} of the section's cells), dissociated {len(cap)} ({int(traced.sum())} traced), genes {G}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
