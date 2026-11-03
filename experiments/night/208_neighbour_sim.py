"""The dissociated cells from a neighbouring region: the modalities share subclones, not cells.  2026-10-21.
Pedro, 2026-10-21: "we can't assume that the dissociated cells come from the same slice directly -- the dissociated cells come
from a neighbouring region that should have the same subclonal composition as the spatial section"; "the subclones should be
shared between the two modalities, but not the specific cells"; "Let's stay in 2D and assume that the dissociated cells come from
a very similar neighboring region".  (vault decisions/decision-dissociated-from-neighbouring-region.md)
Until now (84_.problem, 189_.problem_me) one 2D tumour was split at random: half its cells pooled into spots, half dissociated, so
every dissociated cell lay in the section the spots measure.
Design:
  section     189_.simulate_me unchanged (tumour, microenvironment, margin); ALL its cells are pooled into spots, on a lattice
              with about BEADS_PER_SPOT = 10 cells per spot (the pitch the interleaved design had: 5 cells per spot over half the
              cells)
  region      a mirror of the section in new cells, one per section cell (its template):
                tumour  a new leaf branching off the template's lineage at its deepest ancestor u with time <= T (1 - TAU_FRAC);
                        inherited programs prog[u] + N(0, tree_sd^2 (T - t_u) / T), the same loadings and baseline; the
                        location-bound programs at its own position; the template's clone; traced with probability 1 - UNTRACED
                        (own stream).  Two new cells' common ancestor is at min(t_u, t_u', Lt(template, template')).
                ME      a new cell of the template's type with the type program, at its own position
              position: the template's plus N(0, JITTER um) -- "very similar" region, aligned with the section
  dissociated a random half of the region, then the capture bias of 189_ (same rule and stream), so the counts match the
              interleaved design
Own random stream 1.1e7 + seed (+ rep for the half).  Truth, for evaluation only: each dissociated cell's position in the region
(xy_true, aligned with the section), and its subclone in the section: the section's tumour cells sharing its lineage at the
section tree's cut into N_SUB = 20 lineages, or at its branch point if that is earlier (the branch point is usually its template's
last division, so the template alone would otherwise be its only relative there).
Usage: imported (simulate_nb, problem_nb, metrics_nb); python 208_neighbour_sim.py runs a check on one tumour.
"""
import importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TAU_FRAC, JITTER, BEADS_PER_SPOT, N_SUB = 0.1, 6.0, 10.0, 20


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def simulate_nb(seed, tau_frac=TAU_FRAC, jitter=JITTER, untraced=0.3, lib_scale=1.0, section=None, mirror_static_region=False, shuffle_static_region=False, **kw):
    """the section (189_.simulate_me(seed, untraced=untraced, **kw)) and its neighbouring region of new cells.  lib_scale multiplies
    the region's (dissociated side's) library size -- deeper sequencing, e.g. plate-based full-length protocols (added 2026-10-21;
    1 leaves every draw as before).  section: a section already built (e.g. 211_'s intermixed one) in place of simulate_me's"""
    m189 = mod("m189", "189_me_sim.py")
    S = m189.simulate_me(seed, untraced=untraced, **kw) if section is None else section; sim = S["sim"]; ep, mp = sim["expr_parts"], S["me_parts"]
    rng = np.random.default_rng(11 * 10 ** 6 + seed)
    n_all, n_t, T = len(S["xy"]), S["n_t"], S["T"]; parent, ntime, leaves = sim["parent"], sim["ntime"], sim["leaves"]
    xy = S["xy"] + rng.normal(0, jitter, (n_all, 2))
    t_b = T * (1.0 - tau_frac); u = np.empty(n_t, int)
    for j in range(n_t):                                     # deepest ancestor at or before the branch time
        v = parent[leaves[j]]
        while ntime[v] > t_b:
            v = parent[v]
        u[j] = v
    t_u = ntime[u]
    prog = ep["prog"][u] + rng.normal(0, 1, (n_t, ep["prog"].shape[1])) * ep["tree_sd"] * np.sqrt((T - t_u) / T)[:, None]
    loc = m189.location_logit(sim, xy, clades=np.r_[np.asarray(sim["clade"])[:n_t], np.full(n_all - n_t, -1)])   # lineages for pos_only_mig
    if mirror_static_region:                                 # (2026-10-23) the region's non-migrating tumour cells carry the location-bound
        mot = sim["motility"]; static = ~np.isin(np.asarray(sim["clade"])[:n_t], np.where(mot > mot.min())[0])   # programs of the point
        xm = 2.0 * S["xy"][:n_t].mean(0) - xy[:n_t]          # mirrored through the tumour's centre: in the dissociated sample they look like
        loc[:n_t][static] = m189.location_logit(sim, xm, clades=np.asarray(sim["clade"])[:n_t])[static]   # the section far from where they
                                                             # are -- expression misleads them; the section is unchanged; no draws change
    if shuffle_static_region:                                # (2026-10-23) the region's non-migrating tumour cells carry the location-bound
        mot = sim["motility"]; cl_t = np.asarray(sim["clade"])[:n_t]; static = ~np.isin(cl_t, np.where(mot > mot.min())[0])   # programs at the
        rs = np.random.default_rng(16 * 10 ** 6 + seed); xs = xy[:n_t].copy()   # position of a random other cell of their own lineage: the
        for f in np.unique(cl_t[static]):                    # clone's location stays right, its internal arrangement is scrambled (own stream)
            ii = np.where(cl_t == f)[0]; xs[ii] = xy[:n_t][ii[rs.permutation(len(ii))]]
        loc[:n_t][static] = m189.location_logit(sim, xs, clades=cl_t)[static]
    logit_t = ep["base"][None, :] + prog @ ep["load"][: prog.shape[1]] + loc[:n_t]
    me = S["kind"] == 1; logit_m = mp["base"][None, :] + mp["load"][S["me_type"][me]] + loc[me]
    logit = np.vstack([logit_t, logit_m]); rate = np.exp(logit - logit.max(1, keepdims=True)); rate /= rate.sum(1, keepdims=True)
    counts = rng.poisson(rate * lib_scale * np.r_[np.full(n_t, ep["lib"]), np.full(me.sum(), mp["lib"])][:, None]).astype(float)
    traced = np.r_[rng.random(n_t) >= untraced, np.zeros(me.sum(), bool)]
    Lt = np.minimum(np.minimum(t_u[:, None], t_u[None, :]), S["Lt"]); np.fill_diagonal(Lt, T)
    R = dict(xy=xy, kind=S["kind"].copy(), cl=S["cl"].copy(), me_type=S["me_type"].copy(), traced=traced, counts=counts, Lt=Lt, u=u, t_u=t_u,
             template=np.arange(n_all), in_margin=S["in_margin"].copy())
    t_sub = m189.clonal_cut(S["Lt"][:n_t, :n_t], N_SUB)        # the section tree's cut into N_SUB lineages
    return dict(section=S, region=R, seed=seed, K=S["K"], Km=S["Km"], T=T, n_t=n_t, tau_frac=tau_frac, jitter=jitter, t_sub=t_sub)


def problem_nb(NB, arm, resolution="single", n_clones=20, beads_per_spot=BEADS_PER_SPOT, rep=0, tree_units="space2", capture=None, segmentation=None,
               dissoc_frac=0.5, t_cut_fixed=None, drop_section=None, drop_dissoc=None):
    """the mapping problem, in 189_.problem_me's format: spots from every section cell (arm 'tumour': its tumour cells only);
    the dissociated cells a random half of the region, capture-biased; 'all' maps every dissociated cell, otherwise the traced
    tumour cells only.  Extra keys: rel (per mapped tumour cell, the section's tumour cells of its subclone), xy_section.
    drop_section / drop_dissoc (2026-10-24; boolean over the n_all cells, None changes nothing): section cells left out of the spots
    (the spot grid is kept), region cells left out of the dissociated sample after capture (the other cells are the same) -- a clone
    absent from one modality"""
    import ot
    from scipy.spatial.distance import pdist, squareform
    m189 = mod("m189", "189_me_sim.py")
    S, R = NB["section"], NB["region"]; xy_s = S["xy"]; n_all = len(xy_s)
    side = np.arange(n_all) if arm != "tumour" else np.where(S["kind"] == 0)[0]
    pitches = np.linspace(12, 120, 181); x0a = xy_s.min(0)
    occ = [np.bincount(np.unique(np.floor((xy_s - x0a) / p).astype(int), axis=0, return_inverse=True)[1].ravel()).mean() for p in pitches]
    pitch = float(pitches[int(np.argmin(np.abs(np.array(occ) - beads_per_spot)))])
    x0 = xy_s[side].min(0)
    if drop_section is not None:
        side = side[~np.asarray(drop_section, bool)[side]]
    g = np.floor((xy_s[side] - x0) / pitch).astype(int); keys, inv = np.unique(g, axis=0, return_inverse=True)
    inv = inv.ravel(); m = len(keys); spots = x0 + (keys + 0.5) * pitch
    k_use = n_all // 2 if dissoc_frac == 0.5 else int(round(dissoc_frac * n_all))      # dissoc_frac: the share of the region dissociated
    use = np.random.default_rng(11 * 10 ** 6 + 10 * NB["seed"] + rep + 1).permutation(n_all)[:k_use]   # (0.5 as run until 2026-10-22; 1.0 since)
    if capture is not None:                                  # 189_'s capture rule and stream
        rc = np.random.default_rng(9 * 10 ** 6 + NB["seed"]); kind_u = R["kind"][use]
        clone_rate = np.exp(rc.normal(0, capture.get("clone_sd", 0.0), NB["K"]))
        me_rates = np.r_[capture.get("me", [1.0, 1.0, 1.0]), np.ones(m189.N_ME_TYPES)][: m189.N_ME_TYPES]
        kap = np.where(kind_u == 0, capture.get("tumour", 1.0) * clone_rate[np.maximum(R["cl"][use], 0)], me_rates[np.maximum(R["me_type"][use], 0)])
        use = use[rc.random(len(use)) < np.minimum(1.0, capture.get("keep", 0.5) * kap / kap.mean())]
    if drop_dissoc is not None:
        use = use[~np.asarray(drop_dissoc, bool)[use]]
    mapped = use if arm == "all" else use[(R["kind"][use] == 0) & R["traced"][use]]
    K, nT = NB["K"], m189.N_ME_TYPES
    Ncl = np.zeros((m, K)); tum = S["kind"][side] == 0; np.add.at(Ncl, (inv[tum], S["cl"][side][tum]), 1.0)
    Nme = np.zeros((m, nT)); me = S["kind"][side] == 1; np.add.at(Nme, (inv[me], S["me_type"][side][me]), 1.0)
    ncell = np.bincount(inv, minlength=m).astype(float)
    Sc = np.zeros((m, S["counts"].shape[1])); np.add.at(Sc, inv, S["counts"][side])
    lognorm = lambda Cm: np.log1p(Cm / np.maximum(Cm.sum(1, keepdims=True), 1.0) * 1e4)
    Xsc, Xst = lognorm(R["counts"][mapped]), lognorm(Sc); M = ot.dist(Xsc, Xst); M /= M.max()
    n = len(mapped); kind = R["kind"][mapped]; tr = R["traced"][mapped]
    a = tr.astype(float); W = np.outer(a, a); np.fill_diagonal(W, 0.0)
    AGE = np.zeros((n, n)); ti = np.where(tr)[0]; gi = mapped[ti]
    Lt = R["Lt"][np.ix_(gi, gi)]; k_clones = t_cut = None
    if resolution == "clonal":
        from scipy.sparse.csgraph import connected_components
        t_cut = m189.clonal_cut(Lt, n_clones) if t_cut_fixed is None else float(t_cut_fixed)   # t_cut_fixed (2026-10-23): cut at a given time
        k_clones = int(connected_components((Lt >= t_cut).astype(int), directed=False)[0])
        Lt = np.minimum(Lt, t_cut)
    age = NB["T"] - Lt; np.fill_diagonal(age, 0.0); AGE[np.ix_(ti, ti)] = age
    if tree_units in ("space2", "segment"):
        q_ = ncell / ncell.sum(); Dm = squareform(pdist(spots))
        if tree_units == "segment":
            seg = Ncl.sum(1) > 0 if segmentation is None else np.asarray(segmentation, bool); q_ = q_ * seg; q_ /= q_.sum()
        AGE = AGE * float(q_ @ Dm @ q_) / float(2.0 * age[np.triu_indices(len(ti), 1)].mean())
    tum_s = np.where(S["kind"] == 0)[0]
    rel = [tum_s[S["Lt"][mapped[i], tum_s] >= min(R["t_u"][mapped[i]], NB["t_sub"])] if kind[i] == 0 else np.zeros(0, int) for i in range(n)]
    return dict(M=M, D=squareform(pdist(spots)), q=ncell / ncell.sum(), W=W, WP=W * 2.0 * AGE, AGE=AGE, Xsc=Xsc, Xst=Xst,
                cl=np.where(kind == 0, R["cl"][mapped], K + R["me_type"][mapped]), K=K, nT=nT, kind=kind, traced=tr,
                me_type=R["me_type"][mapped], xy_true=R["xy"][mapped], spots=spots, pitch=pitch, Ncl=Ncl, Nme=Nme, ncell=ncell,
                n_clones=k_clones, t_cut=t_cut, arm=arm, resolution=resolution, mapped=mapped, use=use, Sc=Sc, rel=rel, xy_section=xy_s)


def metrics_nb(T, X):
    """189_.metrics (placement against each cell's position in the region, aligned with the section; composition against the
    section's cells per spot), plus the subclone-level truth for traced cells: subclone_err, the median distance (um) from the
    mapped spot to the nearest section cell of its subclone, and in_subclone, the share mapped within one spot pitch of one"""
    m189 = mod("m189", "189_me_sim.py"); out = m189.metrics(T, X)
    tr = np.where((X["kind"] == 0) & X["traced"])[0]; best = X["spots"][T.argmax(1)]
    d = [np.min(np.linalg.norm(X["xy_section"][X["rel"][i]] - best[i], axis=1)) for i in tr if len(X["rel"][i])]
    out["subclone_err"] = float(np.median(d)); out["in_subclone"] = float(np.mean(np.array(d) <= X["pitch"]))
    return out


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    m189, m197, m200 = mod("m189", "189_me_sim.py"), mod("m197", "197_single_relaxed.py"), mod("m200", "200_margin_scale.py")
    NB = simulate_nb(46, pattern="interspersed", margin=1.0); S, R = NB["section"], NB["region"]
    X = problem_nb(NB, "all", "single", capture=m200.CAPTURE["on"]); Xi = m189.problem_me(S, "all", "single", tree_units="space2", capture=m200.CAPTURE["on"])
    tr = (X["kind"] == 0) & X["traced"]
    print(f"section {len(S['xy']):,} cells -> {len(X['q'])} spots, pitch {X['pitch']:.0f} um, {np.median(X['ncell']):.0f} cells per spot; "
          f"dissociated {len(X['cl']):,} ({tr.sum()} traced); interleaved design: {len(Xi['q'])} spots, pitch {Xi['pitch']:.0f}, {len(Xi['cl']):,} dissociated")
    print(f"branch points: median {np.median(NB['T'] - R['t_u'][:NB['n_t']]) / NB['T']:.2f} T before the present; subclone cut at {1 - NB['t_sub'] / NB['T']:.2f} T before; "
          f"section cells in a traced cell's subclone: median {np.median([len(r) for r, k in zip(X['rel'], tr) if k]):.0f}")
    for nm, XX, f in [("neighbouring", X, metrics_nb), ("interleaved", Xi, m189.metrics)]:
        for a in [0.0, 0.3, 0.5]:
            mt = f(m197.solve_single(XX, a, None), XX)
            print(f"   {nm:12s} alpha {a}: traced placement {mt['traced_err']:.0f} um" + (f", to its subclone {mt['subclone_err']:.0f} um (within a spot of it {mt['in_subclone']:.0%})" if "subclone_err" in mt else "") +
                  f", clone mix {mt['clone_tv']:.3f}, ME {mt['me_err']:.0f} um")
