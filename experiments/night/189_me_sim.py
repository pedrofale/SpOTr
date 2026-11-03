"""Microenvironment and clonal trees in the growth simulator (84_growth_sim.py), for mapping all cells.  2026-10-21.
Pedro, 2026-10-21: "Those cells should definitely also be mapped"; "it seems important to me to account for the
microenvironment cells when using the cancer tree to map the cancer cells too"; "we may only have a clonal tree on the
dissociated side, instead of a fully resolved single cell lineage. Please design the simulator to account for that too".
Design (see vault methods/method-untraced-cells.md):
  tumour      84_.simulate unchanged (same tumours for the same seed); a fraction UNTRACED of tumour cells carries no lineage
  ME cells    F_ME x as many microenvironment cells as tumour cells, of 3 types (proportions 0.5 / 0.3 / 0.2, e.g. macrophage,
              T cell, endothelial), each with its own expression program (25 genes, loadings N(0, ME_STRENGTH)) on its own
              baseline, plus the tumour's location-bound programs at the ME cell's position (the edge program and the
              positional fields, as in 84_: state stays with the place); Poisson counts at the tumour's library size.
              Placement pattern:
                none          no microenvironment
                border        concentrated at the tumour edge (anchor tumour cells drawn with weight exp(-edge / 30 um))
                patchy        (211_) infiltrating one half of the tumour only (anchors drawn from the tumour cells on one side of a
                              line through its centre, random direction, own stream 1.2e7 + seed): the other half is pure cancer
                interspersed  infiltrating evenly (anchor tumour cells drawn uniformly)
                clone         half the major clades infiltrated, half excluded (anchors drawn from the infiltrated clades only)
                structured    (2026-10-27, Pedro: "I'd rather the simulator didn't place the microenvironment randomly") each type
                              where it lives: macrophages at the invasive edge (anchor weight exp(-edge / 30 um)), T cells in N_AGG
                              aggregates (Gaussian weight of scale AGG_UM around tumour cells drawn as centres), endothelial cells
                              evenly; in the margin, immune cells enriched near the tumour and normal epithelium away from it (type
                              weights MARGIN_PROPS x (2 n, 2 n, 1, 1.5 (1 - n)), n = exp(-distance to the tumour / 60 um)); types and
                              anchors from own streams (2.1e7, 2.2e7 + seed), the main streams kept in step with 'interspersed', so the
                              tumour, the type programs and the margin's cells are the same world's and only where each type sits changes
              each ME cell sits at its anchor's position plus N(0, 6 um) jitter.  Own random stream (7e6 + seed).
  trees       single-cell: the true lineage among traced tumour cells (time of the most recent common ancestor Lt).
              clonal: the tree cut where it has 20 lineages among the traced dissociated cells and everything below
              collapsed, Lt_clonal = min(Lt, t_cut): cells whose common ancestor lies after the cut are unresolved members of
              one clone, and the clones keep their true relationships.  Untraced tumour cells: own random stream (8e6 + seed).
  spots       a square lattice over the spatial half of the cells, pitch chosen for about 5 cells per spot (84_'s rule).
  margin      (2026-10-21, Pedro: "The simulation should probably also have a part of the tissue that's mostly microenvironment
              cells"; "Build it that way") a band of tumour-free tissue around the tumour: the free lattice sites nearest the
              tumour, as many as MARGIN x the tumour's sites, each holding Poisson(the tumour region's cells per site) cells of
              4 types (macrophage 0.2, T cell 0.1, endothelial 0.2, normal epithelium 0.5 -- the last only in the margin), with
              the type programs and the location-bound programs at their positions; own random stream (1e7 + seed)
  arms        'tumour'   the current idealised setup: traced tumour cells mapped, spots pooled from tumour cells only
              'ignore'   realistic spots (all cells), only traced tumour cells mapped
              'all'      realistic spots, every dissociated cell mapped (traced, untraced, ME); the tree term only between
                         pairs of traced tumour cells (product weights: any pair with an untraced or ME cell has weight 0)
"""
import importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ME_PROPS, ME_STRENGTH, N_ME_GENES = [0.5, 0.3, 0.2], 1.5, 25
N_ME_TYPES = 4                                               # the 3 types above, plus normal epithelium (margin only)
MARGIN_PROPS = [0.2, 0.1, 0.2, 0.5]                          # in the tumour-free margin: macrophage, T cell, endothelial, normal epithelium
CELL_UM = 12.0
N_AGG, AGG_UM = 3, 60.0                                       # 'structured': T-cell aggregates and their scale


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def location_logit(sim, xy, clades=None):
    """the tumour's location-bound programs (84_: the edge program, and the positional fields if any) evaluated at new
    positions -- state stays with the place, for microenvironment cells as for tumour cells"""
    lp = sim["loc_parts"]; edge = sim["edge_tree"].query(xy)[0]
    out = lp["space_w"] * (np.exp(-edge / 60.0) - 0.5)[:, None] * lp["edge_load"][None, :]
    if lp["pos"] is not None:
        q = lp["pos"]; F = np.stack([(sg * np.exp(-((xy[:, None] - C[None]) ** 2).sum(-1) / (2 * q["pos_scale"] ** 2))).sum(1)
                                     for C, sg in zip(q["centres"], q["signs"])], 1)
        oc = q.get("only_clades")                             # 84_ pos_only_mig: the programs only in those lineages' cells (clades: founder
        mask = 1.0 if oc is None else (np.isin(clades, oc)[:, None] if clades is not None else 0.0)    # per row, -1 for non-tumour)
        out = out + q["pos_w"] * ((F - q["F_mean"]) / q["F_std"]) @ q["Lp"] * mask

    return out


def simulate_me(seed, tree_sd=0.3, pos_w=0.0, mig=0.0, pattern="interspersed", f_me=1.0, untraced=0.3, min_major=60, margin=0.0, pos_only_mig=False, mig_n=None, motility=0.02):
    m84 = mod("m84", "84_growth_sim.py")
    # motility (2026-10-28): runs per cell cycle during growth, the same for every lineage (84_ m_hi = m_lo); 0.02 is the value every
    # world used before (84_'s m_lo default); larger values blur the clones' borders while keeping their territories
    sim = m84.simulate(seed, m_hi=motility, m_lo=motility, pos_w=pos_w, tree_sd=tree_sd, mig=mig, pos_only_mig=pos_only_mig, mig_n=mig_n)
    rng = np.random.default_rng(7 * 10 ** 6 + seed)
    xy_t = sim["xy"]; n_t = len(xy_t); G = sim["counts"].shape[1]; nf = len(sim["motility"])
    sz = np.bincount(sim["clade"], minlength=nf); lab = np.full(nf, -1); major = np.where(sz >= min_major)[0]
    lab[major] = np.arange(len(major)); minor = np.where(lab < 0)[0]; lab[minor] = len(major) + np.arange(len(minor))
    Km = len(major); cl_t = np.minimum(lab[sim["clade"]], Km)
    n_me = 0 if pattern == "none" else int(round(f_me * n_t)); infiltrated = np.array([], int)
    if pattern == "interspersed":
        anchor = rng.integers(0, n_t, n_me)
    elif pattern == "border":
        w = np.exp(-sim["edge"] / 30.0); anchor = rng.choice(n_t, n_me, p=w / w.sum())
    elif pattern == "clone":
        infiltrated = np.sort(rng.choice(Km, Km // 2, replace=False)); pool = np.where(np.isin(cl_t, infiltrated))[0]; anchor = rng.choice(pool, n_me)
    elif pattern == "patchy":                                 # (211_, 2026-10-21) pure-cancer half: anchors only on one side of a line
        th = np.random.default_rng(12 * 10 ** 6 + seed).uniform(0, 2 * np.pi)     # through the tumour's centre, random direction (own stream)
        proj = (xy_t - xy_t.mean(0)) @ np.array([np.cos(th), np.sin(th)]); pool = np.where(proj > np.median(proj))[0]; anchor = rng.choice(pool, n_me)
    elif pattern == "structured":                             # (2026-10-27) each type where it lives; own stream, main stream kept in step
        _ = rng.integers(0, n_t, n_me); rs = np.random.default_rng(21 * 10 ** 6 + seed)
        type_s = rs.choice(len(ME_PROPS), n_me, p=ME_PROPS); anchor = np.empty(n_me, int)
        C = xy_t[rs.choice(n_t, N_AGG, replace=False)]; w_agg = np.exp(-((xy_t[:, None] - C[None]) ** 2).sum(-1) / (2 * AGG_UM ** 2)).sum(1)
        for k, w in enumerate([np.exp(-sim["edge"] / 30.0), w_agg, np.ones(n_t)]):       # macrophages at the edge, T cells in aggregates,
            ii = np.where(type_s == k)[0]; anchor[ii] = rs.choice(n_t, len(ii), p=w / w.sum())   # endothelial cells evenly
    else:
        anchor = np.array([], int)
    xy_me = xy_t[anchor] + rng.normal(0, 6.0, (n_me, 2))
    me_type = rng.choice(len(ME_PROPS), n_me, p=ME_PROPS)
    if pattern == "structured":
        me_type = type_s
    base = rng.normal(0, 0.5, G); load = np.zeros((len(ME_PROPS), G))
    for k in range(len(ME_PROPS)):
        g = rng.choice(G, N_ME_GENES, replace=False); load[k, g] = rng.normal(0, ME_STRENGTH, N_ME_GENES)
    logit = base[None, :] + load[me_type] + location_logit(sim, xy_me)
    rate = np.exp(logit - logit.max(1, keepdims=True)); rate /= rate.sum(1, keepdims=True)
    lib = sim["counts"].sum(1).mean()
    counts_me = rng.poisson(rate * lib)
    traced = np.random.default_rng(8 * 10 ** 6 + seed).random(n_t) >= untraced     # own stream: independent of the ME draws
    xy_m = np.zeros((0, 2)); type_m = np.zeros(0, int); counts_m = np.zeros((0, G))
    if margin > 0:                                           # tumour-free margin: own stream, earlier worlds unchanged
        from scipy.spatial import cKDTree
        rm = np.random.default_rng(10 ** 7 + seed); sites = np.round(xy_t / CELL_UM).astype(int); occ = set(map(tuple, sites))
        R = int(np.ceil(np.sqrt(margin * n_t))) + 5; gx = np.arange(sites[:, 0].min() - R, sites[:, 0].max() + R + 1); gy = np.arange(sites[:, 1].min() - R, sites[:, 1].max() + R + 1)
        grid = np.array([(a, b) for a in gx for b in gy if (a, b) not in occ])
        d = cKDTree(xy_t).query(grid * CELL_UM)[0]; take = grid[np.argsort(d)[: int(round(margin * n_t))]]
        per_site = rm.poisson((n_t + n_me) / n_t, len(take))     # the tumour region's cells per lattice site
        xy_m = np.repeat(take * CELL_UM, per_site, axis=0).astype(float) + rm.normal(0, 3.0, (int(per_site.sum()), 2))
        type_m = rm.choice(N_ME_TYPES, len(xy_m), p=MARGIN_PROPS)
        if pattern == "structured":                          # immune cells near the tumour, normal epithelium away from it (own stream)
            near = np.exp(-cKDTree(xy_t).query(xy_m)[0] / 60.0)
            Pm = np.array(MARGIN_PROPS)[None, :] * np.stack([2 * near, 2 * near, np.ones_like(near), 1.5 * (1 - near)], 1); Pm /= Pm.sum(1, keepdims=True)
            type_m = (Pm.cumsum(1) > np.random.default_rng(22 * 10 ** 6 + seed).random(len(xy_m))[:, None]).argmax(1)
        load4 = np.zeros((N_ME_TYPES, G)); load4[: len(ME_PROPS)] = load
        g4 = rm.choice(G, N_ME_GENES, replace=False); load4[3, g4] = rm.normal(0, ME_STRENGTH, N_ME_GENES)
        lm = base[None, :] + load4[type_m] + location_logit(sim, xy_m); rm_ = np.exp(lm - lm.max(1, keepdims=True)); rm_ /= rm_.sum(1, keepdims=True)
        counts_m = rm.poisson(rm_ * lib)
    n_m = len(xy_m)
    me_load = np.zeros((N_ME_TYPES, G)); me_load[: len(ME_PROPS)] = load
    if margin > 0:
        me_load = load4
    return dict(me_parts=dict(base=base, load=me_load, lib=lib), sim=sim, seed=seed, n_t=n_t, n_me=n_me + n_m, n_margin=n_m, Km=Km, K=Km + 1, pattern=pattern, infiltrated=infiltrated,
                xy=np.vstack([xy_t, xy_me, xy_m]), kind=np.r_[np.zeros(n_t, int), np.ones(n_me + n_m, int)],
                cl=np.r_[cl_t, np.full(n_me + n_m, -1)], me_type=np.r_[np.full(n_t, -1), me_type, type_m], traced=np.r_[traced, np.zeros(n_me + n_m, bool)],
                in_margin=np.r_[np.zeros(n_t + n_me, bool), np.ones(n_m, bool)],
                counts=np.vstack([sim["counts"], counts_me, counts_m]).astype(float), Lt=m84.lca_time(sim), T=float(sim["T"]))


def clonal_cut(Lt, n_clones=20):
    """the cut time at which the tree among these cells has n_clones lineages (bisection; cells whose common ancestor lies
    after the cut are one clone)"""
    from scipy.sparse.csgraph import connected_components
    lo, hi = 0.0, float(Lt.max())
    for _ in range(40):
        mid = 0.5 * (lo + hi); k = connected_components((Lt >= mid).astype(int), directed=False)[0]
        lo, hi = (mid, hi) if k <= n_clones else (lo, mid)
    return lo


def problem_me(S, arm, resolution="single", n_clones=20, beads_per_spot=5.0, rep=0, tree_units="time", capture=None, segmentation=None):
    """the mapping problem for one arm; resolution 'single' or 'clonal' (the tree cut where it has n_clones lineages among
    the traced dissociated cells).  tree_units 'time' (as 190_), 'space2' or 'segment' (corrected 2026-10-21; 'segment' scales to the
    tumour region given by SEGMENTATION, a boolean per spot, the true region by default), or 'space' (AS RUN in 193_-199_, with a
    factor-2 error: AGE, not the tree distance 2 AGE, was matched to the mean spot distance): the tree distances rescaled so that their mean over
    traced pairs equals the mean distance between spots (weighted by spot size) -- plain FGW compares the two on their raw
    scales, and when the traced cells own only part of the spots' capacity nothing stops the tree term from compressing them
    (found in 190_'s example figure).  capture: None (every dissociated cell kept, as 190_ / 193_) or a dict -- each cell of the
    dissociated half kept with probability KEEP x its capture rate / the mean rate, the rate being TUMOUR x a per-clone factor
    exp(N(0, CLONE_SD)) for tumour cells and ME[type] for microenvironment cells (own random stream 9e6 + seed); KEEP with
    every rate 1 keeps that share uniformly, the no-bias control"""
    import ot
    from scipy.spatial.distance import pdist, squareform
    rng = np.random.default_rng(10 * S["seed"] + rep); n_all = len(S["xy"]); perm = rng.permutation(n_all)
    use, side = perm[n_all // 2:], perm[: n_all // 2]
    if arm == "tumour":
        side = side[S["kind"][side] == 0]
    xy = S["xy"]
    def lattice(p):
        x0 = xy[side].min(0); g = np.floor((xy[side] - x0) / p).astype(int); keys, inv = np.unique(g, axis=0, return_inverse=True); return keys, inv.ravel(), x0
    side_all = perm[: n_all // 2]; x0a = xy[side_all].min(0)
    pitches = np.linspace(12, 120, 181)
    occ = [np.bincount(np.unique(np.floor((xy[side_all] - x0a) / p).astype(int), axis=0, return_inverse=True)[1].ravel()).mean() for p in pitches]
    pitch = float(pitches[int(np.argmin(np.abs(np.array(occ) - beads_per_spot)))])
    keys, inv, x0 = lattice(pitch); m = len(keys); spots = x0 + (keys + 0.5) * pitch
    if capture is not None:                                  # dissociation: keep a cell with probability proportional to its capture rate
        rc = np.random.default_rng(9 * 10 ** 6 + S["seed"]); kind_u = S["kind"][use]
        clone_rate = np.exp(rc.normal(0, capture.get("clone_sd", 0.0), S["K"]))
        me_rates = np.r_[capture.get("me", [1.0, 1.0, 1.0]), np.ones(N_ME_TYPES)][:N_ME_TYPES]
        kap = np.where(kind_u == 0, capture.get("tumour", 1.0) * clone_rate[np.maximum(S["cl"][use], 0)], me_rates[np.maximum(S["me_type"][use], 0)])
        keep = rc.random(len(use)) < np.minimum(1.0, capture.get("keep", 0.5) * kap / kap.mean())
        use = use[keep]
    if arm == "all":
        mapped = use
    else:
        mapped = use[(S["kind"][use] == 0) & S["traced"][use]]
    K, nT = S["K"], N_ME_TYPES
    Ncl = np.zeros((m, K)); tum = S["kind"][side] == 0; np.add.at(Ncl, (inv[tum], S["cl"][side][tum]), 1.0)
    Nme = np.zeros((m, nT)); me = S["kind"][side] == 1; np.add.at(Nme, (inv[me], S["me_type"][side][me]), 1.0)
    ncell = np.bincount(inv, minlength=m).astype(float)
    Sc = np.zeros((m, S["counts"].shape[1])); np.add.at(Sc, inv, S["counts"][side])
    lognorm = lambda Cm: np.log1p(Cm / np.maximum(Cm.sum(1, keepdims=True), 1.0) * 1e4)
    Xsc, Xst = lognorm(S["counts"][mapped]), lognorm(Sc); M = ot.dist(Xsc, Xst); M /= M.max()
    n = len(mapped); kind = S["kind"][mapped]; tr = S["traced"][mapped]
    a = tr.astype(float); W = np.outer(a, a); np.fill_diagonal(W, 0.0)
    AGE = np.zeros((n, n)); ti = np.where(tr)[0]; gi = mapped[ti]          # tumour indices are the first n_t of the cell arrays
    Lt = S["Lt"][np.ix_(gi, gi)]
    k_clones = t_cut = None
    if resolution == "clonal":
        from scipy.sparse.csgraph import connected_components
        t_cut = clonal_cut(Lt, n_clones); k_clones = int(connected_components((Lt >= t_cut).astype(int), directed=False)[0])
        Lt = np.minimum(Lt, t_cut)
    age = S["T"] - Lt; np.fill_diagonal(age, 0.0); AGE[np.ix_(ti, ti)] = age
    if tree_units == "space":                                # AS RUN in 193_-199_: AGE scaled to the mean spot distance, so the tree
        q_ = np.bincount(inv, minlength=m).astype(float); q_ /= q_.sum(); Dm = squareform(pdist(spots))   # distance 2 AGE was twice it
        AGE = AGE * float(q_ @ Dm @ q_) / float(age[np.triu_indices(len(ti), 1)].mean())
    elif tree_units in ("space2", "segment"):                 # corrected (2026-10-21): the tree distance 2 AGE has the mean of the
        q_ = np.bincount(inv, minlength=m).astype(float); q_ /= q_.sum(); Dm = squareform(pdist(spots))   # spot distances --
        if tree_units == "segment":                           # over all spots, or over the tumour region's spots (a segmentation;
            seg = Ncl.sum(1) > 0 if segmentation is None else np.asarray(segmentation, bool)    # the true region by default)
            q_ = q_ * seg; q_ /= q_.sum()
        AGE = AGE * float(q_ @ Dm @ q_) / float(2.0 * age[np.triu_indices(len(ti), 1)].mean())
    return dict(M=M, D=squareform(pdist(spots)), q=ncell / ncell.sum(), W=W, WP=W * 2.0 * AGE, AGE=AGE, Xsc=Xsc, Xst=Xst,
                cl=np.where(kind == 0, S["cl"][mapped], S["K"] + S["me_type"][mapped]), K=K, nT=nT, kind=kind, traced=tr,
                me_type=S["me_type"][mapped], xy_true=xy[mapped], spots=spots, pitch=pitch, Ncl=Ncl, Nme=Nme, ncell=ncell,
                n_clones=k_clones, t_cut=t_cut, arm=arm, resolution=resolution, mapped=mapped, use=use, Sc=Sc)


def tree_scale(X):
    """the tree term's scale, calibrated on the rows where the tree acts (traced tumour cells): the doubly-centred spread of
    the expression cost over those rows against that of the tree gradient at the product coupling (70_.scale restricted).
    Calibrating over all rows would let the 0/1 pattern of masked rows inflate the gradient's spread (38x weaker tree term
    with all cells mapped in a smoke test), so alpha would not mean the same strength in every arm."""
    m70 = mod("m70", "70_spotr.py")
    r = np.where(X["traced"] & (X["kind"] == 0))[0]; n = len(X["cl"]); p = np.ones(n) / n; T0 = np.outer(p, X["q"]); D = X["D"]
    g0 = 2.0 * (X["W"][r] @ T0 @ D ** 2 - 2.0 * X["WP"][r] @ T0 @ D)
    return float(m70.dc_std(X["M"][r]) / m70.dc_std(g0))


def fgw_maps(X, alphas):
    """exact FGW maps with the arm's pair weights (alpha 0: exact OT on the expression cost)"""
    import ot
    m103 = mod("m103", "103_aware_learn.py")
    n = len(X["cl"]); p = np.ones(n) / n; D = X["D"]; s = tree_scale(X)
    return {a: (ot.emd(p, X["q"], X["M"]) if a == 0 else m103.solve_exact_w_gpu(X["M"], X["W"] * s, X["WP"] * s, D, D ** 2, p, X["q"], a)) for a in alphas}


def metrics(T, X):
    """truth: placement (median um, argmax spot) and composition (mean total variation) for each group of mapped cells"""
    best = X["spots"][T.argmax(1)]; dist = np.linalg.norm(best - X["xy_true"], axis=1); out = {}
    tr = (X["kind"] == 0) & X["traced"]
    out["traced_err"] = float(np.median(dist[tr]))
    Y = np.eye(X["K"])[X["cl"][tr]]; P = T[tr].T @ Y; P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    has = X["Ncl"].sum(1) > 0; Q = X["Ncl"][has] / X["Ncl"][has].sum(1, keepdims=True)
    out["clone_tv"] = float(0.5 * np.abs(P[has] - Q).sum(1).mean())
    ut = (X["kind"] == 0) & ~X["traced"]
    if ut.any():
        out["untraced_err"] = float(np.median(dist[ut]))
    me = X["kind"] == 1
    if me.any():
        out["me_err"] = float(np.median(dist[me]))
        Ym = np.eye(X["nT"])[X["me_type"][me]]; Pm = T[me].T @ Ym; Pm = Pm / np.maximum(Pm.sum(1, keepdims=True), 1e-300)
        hm = X["Nme"].sum(1) > 0; Qm = X["Nme"][hm] / X["Nme"][hm].sum(1, keepdims=True)
        out["me_type_tv"] = float(0.5 * np.abs(Pm[hm] - Qm).sum(1).mean())
        share = (T[X["kind"] == 0].sum(0)) / np.maximum(T.sum(0), 1e-300); true_share = X["Ncl"].sum(1) / np.maximum(X["ncell"], 1)
        out["tumour_share_err"] = float(np.abs(share - true_share).mean())
        free = X["Ncl"].sum(1) == 0                              # spots without any tumour cell (e.g. the tumour-free margin)
        if free.any():
            out["traced_mass_in_tumour_free"] = float(T[tr][:, free].sum() / max(T[tr].sum(), 1e-300))
            out["tumour_free_spot_share"] = float(X["q"][free].sum())
    return out
