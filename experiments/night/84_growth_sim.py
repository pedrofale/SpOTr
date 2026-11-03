"""A fully controlled tumour simulator for the biology-guided (growth) loss, 2026-10-13.

The planted PEtracer control (66_) cannot test the growth loss: its dispersal is planted at random, unrelated to
fitness.  Here the biology the loss assumes holds by construction.
  growth      2D Eden model on a square lattice: a cell divides only into an empty neighbouring site, so edge cells
              keep dividing and interior ones stop.  The exact division tree is recorded (node times = division times).
  fitness     the Local Branching Index (Neher et al. 2014) of each cell on that tree, so tree-derived fitness is
              high at the edge, as the loss assumes for 4T1 (fitter at the edge).
  clades      the lineages of the first 128 cells; in surface growth most are trapped inside and stay tiny, so the
              ~10 lineages with >= 60 cells are the clades and the rest are pooled into one 'minor' clade.
              Motility is balanced among the lineages on the frontier at founding, which are the ones that survive.
  motility    a random half of the clades make runs at rate M_HI per cell cycle, the others at M_LO, from founding on:
              a run goes a geometric number of sites (mean RUN) in a random direction through occupied tissue, and
              each cell on the path shifts back by one, so the runner travels far and the cells it passes barely
              move (a neighbour swap moves both equally and mostly inside the clade's own sector, so it hardly
              spreads a clade).  The planted difference in spread.
  expression  P inherited programs (Brownian motion along the tree), one program that depends on the distance to
              the edge, Poisson counts; the inherited signal's strength is a knob.
  spatial     half the cells, pooled on a square spot lattice of about five cells per spot; each spot's clade counts
              and distance to the tumour edge are recorded.
simulate() returns the tumour; problem() turns it into the dictionary 70_spotr.learn takes.
Usage: python 84_growth_sim.py   (sanity checks on one tumour per contrast)
"""
import heapq, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CELL_UM = 12.0                                               # lattice spacing: one cell diameter
NB4 = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])


def _run(c, occ, pos, rng, run, allowed=None):
    """cell c runs a geometric number of sites (mean RUN) in a random direction through occupied tissue; each cell
    on its path shifts back by one site.  allowed(site) -> bool confines the run (a niche)."""
    s = pos[c]; dx, dy = NB4[rng.integers(4)]; k = rng.geometric(1.0 / run)
    path = []
    for j in range(1, k + 1):
        q = (s[0] + j * dx, s[1] + j * dy)
        if q not in occ or (allowed is not None and not allowed(q)):
            break
        path.append(q)
    if not path:
        return
    movers = [occ[q] for q in path]
    for q_from, d in zip(path, movers):
        q_to = (q_from[0] - dx, q_from[1] - dy); occ[q_to] = d; pos[d] = q_to
    occ[path[-1]] = c; pos[c] = path[-1]


def simulate(seed, n_cells=3000, n_founders=128, m_hi=0.3, m_lo=0.02, run=4.0, n_genes=200, n_prog=5, tree_sd=0.6,
             space_w=1.0, lib=800, tau_frac=0.125, mig=0.0, min_major=60, pos_w=0.0, n_pos=4, pos_scale=150.0,
             fast=0, adv=1.0, fast_mig=0.0, niche_mig=0.0, niche_frac=0.3, niche_expr=0.0, border_mig=0.0, border_scale=48.0,
             pos_only_mig=False, mig_n=None):
    """niche_expr > 0 (with a niche): cells inside the disc express an extra program of that strength (an
    infiltration-like signature), so the region can be found from spot expression; niche_mig = 0.001 gives the disc
    and its signature with no mixing (a placebo).
    niche_mig > 0: after growth, every cell inside a disc holding NICHE_FRAC of the cells (centred half-way out
    from the centre, e.g. an infiltrated region) makes on average NICHE_MIG runs confined to the disc, whatever its
    clade -- mixing set by place, not by lineage.  The disc is returned as niche = (centre, radius) in um.
    fast > 0: at founding, FAST lineages on the frontier divide ADV times faster (an expanding clade), and after
    growth only they make on average FAST_MIG runs per cell (a diffuse clade among localized ones).
    mig > 0: after growth, half of the major lineages (>= min_major cells, chosen evenly) make on average MIG runs
    per cell -- contrast in spread without touching growth or survival.  pos_w > 0: N_POS smooth spatial programs
    (sums of Gaussian bumps of width POS_SCALE um) in the expression, so expression carries position.  Both draw
    from their own random streams, so the defaults reproduce every earlier tumour exactly.
    border_mig > 0 (2026-10-19): after growth, each cell makes on average border_mig x exp(-d / border_scale) runs, d its
    distance (um) to the tumour edge -- mixing that grows toward the border, whatever the clade (the core/border effect
    measured on PEtracer and KP, section 47); its own random stream.  Dispersal that grows with time since divergence
    needs no new knob: equal motility for every lineage during growth (m_hi = m_lo > 0) makes lineages that split earlier
    more dispersed."""
    rng = np.random.default_rng(seed)
    occ = {(0, 0): 0}                                        # site -> cell
    pos = [(0, 0)]; node = [0]; clade = [-1]
    parent, ntime = [-1], [0.0]                              # tree nodes
    t = 0.0
    frontier = {(0, 0)}
    mot = None

    def is_front(s):
        return s in occ and any((s[0] + dx, s[1] + dy) not in occ for dx, dy in NB4)

    div = None                                               # per-lineage division rate, only with fast > 0
    while len(pos) < n_cells:
        F = list(frontier)
        if div is None:
            dt = rng.exponential(1.0 / len(F)); t += dt
        else:
            wF = div[[clade[occ[q]] for q in F]]; dt = rng.exponential(1.0 / wF.sum()); t += dt
        if mot is not None:                                  # runs during dt, rate m_clade per cell
            rates = mot[np.asarray(clade)]
            n_run = rng.poisson(rates.sum() * dt)
            if n_run:
                who = rng.choice(len(pos), n_run, p=rates / rates.sum())
                for c in who:                                # c runs k sites; the cells on its path shift back one
                    s = pos[c]; dx, dy = NB4[rng.integers(4)]; k = rng.geometric(1.0 / run)
                    path = []
                    for j in range(1, k + 1):
                        q = (s[0] + j * dx, s[1] + j * dy)
                        if q not in occ:
                            break
                        path.append(q)
                    if not path:
                        continue
                    movers = [occ[q] for q in path]
                    for q_from, d in zip(path, movers):
                        q_to = (q_from[0] - dx, q_from[1] - dy); occ[q_to] = d; pos[d] = q_to
                    occ[path[-1]] = c; pos[c] = path[-1]
        s = F[rng.integers(len(F))] if div is None else F[rng.choice(len(F), p=wF / wF.sum())]; c = occ[s]
        empty = [(s[0] + dx, s[1] + dy) for dx, dy in NB4 if (s[0] + dx, s[1] + dy) not in occ]
        s2 = empty[rng.integers(len(empty))]
        v = node[c]; v1, v2 = len(parent), len(parent) + 1
        parent += [v, v]; ntime += [t, t]
        node[c] = v1; d = len(pos); pos.append(s2); node.append(v2); clade.append(clade[c]); occ[s2] = d
        for q in [s, s2] + [(s2[0] + dx, s2[1] + dy) for dx, dy in NB4]:
            if is_front(q):
                frontier.add(q)
            else:
                frontier.discard(q)
        if mot is None and len(pos) == n_founders:            # founders: clades and their motility
            clade = list(range(n_founders))
            fr = np.array(sorted({occ[q] for q in frontier})); inner = np.setdiff1d(np.arange(n_founders), fr)
            hi = np.r_[rng.permutation(fr)[: len(fr) // 2], rng.permutation(inner)[: len(inner) // 2]]
            mot = np.full(n_founders, m_lo); mot[hi] = m_hi     # balanced among the frontier lineages, which survive
            if fast > 0:                                     # the expanding lineages: the most exposed frontier cells
                expo = np.array([sum((pos[c][0] + dx, pos[c][1] + dy) not in occ for dx, dy in NB4) for c in fr])
                fast_ids = fr[np.lexsort((rng.random(len(fr)), -expo))][:fast]
                div = np.ones(n_founders); div[fast_ids] = adv
    T = t
    n = len(pos)
    hi_mig = np.array([], int)                               # the migrating lineages, if any (for pos_only_mig)
    if mig > 0:                                              # migration phase: half the major lineages run
        rng2 = np.random.default_rng(10 ** 6 + seed)
        sz = np.bincount(np.asarray(clade), minlength=len(mot)); major = np.where(sz >= min_major)[0]
        hi = rng2.permutation(major)[: (len(major) // 2 if mig_n is None else mig_n)]; hi_mig = hi   # mig_n (2026-10-23): how many lineages run
        runners = np.where(np.isin(np.asarray(clade), hi))[0]
        for c in rng2.choice(runners, rng2.poisson(mig * len(runners))):
            _run(c, occ, pos, rng2, run)
        mot = mot.copy(); mot[hi] += mig                     # label: growth rate + migration runs per cell
    niche = None
    if niche_mig > 0:
        rng5 = np.random.default_rng(4 * 10 ** 6 + seed)
        P = np.array(pos, float); ctr = P.mean(0); ang = rng5.uniform(0, 2 * np.pi)
        rad_t = np.sqrt(len(P) / np.pi); cen = ctr + 0.5 * rad_t * np.array([np.cos(ang), np.sin(ang)])
        rad = float(np.quantile(np.linalg.norm(P - cen, axis=1), niche_frac))
        inside = lambda q: (q[0] - cen[0]) ** 2 + (q[1] - cen[1]) ** 2 <= rad ** 2
        sites = [q for q in occ if inside(q)]
        for j in rng5.integers(len(sites), size=rng5.poisson(niche_mig * len(sites))):
            _run(occ[sites[j]], occ, pos, rng5, run, allowed=inside)
        niche = (cen * CELL_UM, rad * CELL_UM)
    if fast > 0 and fast_mig > 0:                            # only the expanding lineages migrate
        rng4 = np.random.default_rng(3 * 10 ** 6 + seed)
        runners = np.where(np.isin(np.asarray(clade), fast_ids))[0]
        for c in rng4.choice(runners, rng4.poisson(fast_mig * len(runners))):
            _run(c, occ, pos, rng4, run)
        mot = mot.copy(); mot[fast_ids] += fast_mig
    if border_mig > 0:                                       # mixing toward the border, whatever the clade
        from scipy.spatial import cKDTree as _KD
        rng7 = np.random.default_rng(6 * 10 ** 6 + seed)
        bsites = np.array([q for q in occ if any((q[0] + dx, q[1] + dy) not in occ for dx, dy in NB4)], float)
        dist = _KD(bsites).query(np.array(pos, float))[0] * CELL_UM
        for c in rng7.permutation(np.repeat(np.arange(len(pos)), rng7.poisson(border_mig * np.exp(-dist / border_scale)))):
            _run(c, occ, pos, rng7, run)
    leaf = np.array(node); parent = np.array(parent); ntime = np.array(ntime)
    # leaves end at T: give each cell a leaf node below its current node
    nn = len(parent)
    parent = np.r_[parent, leaf]; ntime = np.r_[ntime, np.full(n, T)]
    leaves = np.arange(nn, nn + n)
    xy = np.array(pos, float) * CELL_UM
    lbi_all = lbi(parent, ntime, leaves, np.ones(n, bool), tau_frac * T)
    # distance to the edge: to the nearest occupied site with an empty neighbour
    from scipy.spatial import cKDTree
    bnd = np.array([s for s in occ if any((s[0] + dx, s[1] + dy) not in occ for dx, dy in NB4)], float) * CELL_UM
    edge_tree = cKDTree(bnd)
    edge = edge_tree.query(xy)[0]
    # expression: inherited programs (BM along the tree), an edge program, Poisson counts
    N = len(parent); bl = np.r_[0.0, ntime[1:] - ntime[parent[1:]]]
    prog = np.zeros((N, n_prog))
    for v in range(1, N):                                    # parents precede children in node numbering
        prog[v] = prog[parent[v]] + rng.normal(0, tree_sd * np.sqrt(max(bl[v], 0) / T), n_prog)
    A = prog[leaves]
    load = np.zeros((n_prog + 1, n_genes))
    for p in range(n_prog + 1):
        g = rng.choice(n_genes, 25, replace=False); load[p, g] = rng.normal(0, 1, 25)
    base = rng.normal(0, 0.5, n_genes)
    edge_prog = np.exp(-edge / 60.0) - 0.5                    # high at the edge
    logit = base + A @ load[:n_prog] + space_w * edge_prog[:, None] * load[n_prog]
    if niche_expr > 0 and niche is not None:                 # the niche's expression signature
        rng6 = np.random.default_rng(5 * 10 ** 6 + seed)
        inside = np.linalg.norm(xy - niche[0], axis=1) <= niche[1]
        g = rng6.choice(n_genes, 25, replace=False); ln = np.zeros(n_genes); ln[g] = np.abs(rng6.normal(0, 1, 25))
        logit = logit + niche_expr * inside[:, None] * ln[None, :]
    pos_parts = None
    if pos_w > 0:                                            # positional programs: smooth fields over the section
        rng3 = np.random.default_rng(2 * 10 ** 6 + seed)
        F = np.zeros((n, n_pos)); Lp = np.zeros((n_pos, n_genes)); Cs, Ss = [], []
        for j in range(n_pos):
            C = rng3.uniform(xy.min(0), xy.max(0), (6, 2)); sg = rng3.choice([-1.0, 1.0], 6); Cs.append(C); Ss.append(sg)
            F[:, j] = (sg * np.exp(-((xy[:, None] - C[None]) ** 2).sum(-1) / (2 * pos_scale ** 2))).sum(1)
            g = rng3.choice(n_genes, 25, replace=False); Lp[j, g] = rng3.normal(0, 1, 25)
        pos_parts = dict(centres=Cs, signs=Ss, F_mean=F.mean(0), F_std=F.std(0), Lp=Lp, pos_w=pos_w, pos_scale=pos_scale)
        F = (F - F.mean(0)) / F.std(0)
        # pos_only_mig (2026-10-22): only the migrating lineages' cells carry the positional programs (an invasive state tied to where
        # they are), so expression can place them but not the static clones; no draws change
        logit = logit + pos_w * (F @ Lp) * (np.isin(np.asarray(clade), hi_mig)[:, None] if pos_only_mig else 1.0)
        pos_parts["only_clades"] = hi_mig if pos_only_mig else None

    rate = np.exp(logit); rate /= rate.sum(1, keepdims=True)
    counts = rng.poisson(rate * lib)
    clone_logit = base + A @ load[:n_prog]                   # the inherited (lineage) part alone, for the truth of 181_ (2026-10-21)
    loc_parts = dict(edge_load=load[n_prog], space_w=space_w, pos=pos_parts)   # the location-bound programs, for 189_ (2026-10-21)
    expr_parts = dict(prog=prog, load=load, base=base, lib=lib, tree_sd=tree_sd)     # for new cells of the same lineages (208_, 2026-10-21)
    return dict(clone_logit=clone_logit, loc_parts=loc_parts, expr_parts=expr_parts, niche=niche, fast=(fast_ids if fast > 0 else np.array([], int)), xy=xy, clade=np.asarray(clade), motility=mot, lbi=lbi_all, tau=tau_frac * T, edge=edge, counts=counts, T=T,
                parent=parent, ntime=ntime, leaves=leaves, edge_tree=edge_tree, bnd=bnd, seed=seed)


def lbi(parent, ntime, leaves, keep, tau):
    """Local Branching Index (Neher et al. 2014) of the kept leaves, on the tree induced by them alone: the
    exponentially discounted tree length around each leaf, by an up and a down pass"""
    N = len(parent); kids = [[] for _ in range(N)]
    alive = np.zeros(N, bool); alive[leaves[keep]] = True
    for v in range(N - 1, 0, -1):                            # children have larger indices than their parents
        if alive[v]:
            alive[parent[v]] = True
    for v in range(1, N):
        if alive[v]:
            kids[parent[v]].append(v)
    bl = np.r_[0.0, ntime[1:] - ntime[parent[1:]]]
    con = lambda k, x: tau * (1 - np.exp(-bl[k] / tau)) + np.exp(-bl[k] / tau) * x
    order = []; st = [0]
    while st:
        v = st.pop(); order.append(v); st.extend(kids[v])
    up = np.zeros(N)
    for v in reversed(order):
        up[v] = sum(con(k, up[k]) for k in kids[v])
    down = np.zeros(N)
    for v in order:
        for k in kids[v]:
            down[k] = con(k, down[v] + up[v] - con(k, up[k]))
    return (up + down)[leaves[keep]]


def lca_time(sim):
    """time of the most recent common ancestor of every pair of cells (block fill over a DFS order)"""
    parent, ntime, leaves = sim["parent"], sim["ntime"], sim["leaves"]
    N = len(parent); kids = [[] for _ in range(N)]
    for v in range(1, N):
        kids[parent[v]].append(v)
    order, span = [], {}
    st = [(0, False)]
    while st:
        v, done = st.pop()
        if done:
            span[v] = (span[kids[v][0]][0], span[kids[v][-1]][1]) if kids[v] else (len(order), len(order) + 1)
            if not kids[v]:
                order.append(v)
            continue
        st.append((v, True))
        for k in reversed(kids[v]):
            st.append((k, False))
    n = len(order); Lt = np.zeros((n, n))
    for v in range(N):
        ch = kids[v]
        for i in range(len(ch)):
            a0, a1 = span[ch[i]]
            for j in range(i + 1, len(ch)):
                b0, b1 = span[ch[j]]
                Lt[a0:a1, b0:b1] = ntime[v]; Lt[b0:b1, a0:a1] = ntime[v]
    pos_of = {leaf: i for i, leaf in enumerate(order)}
    idx = np.array([pos_of[l] for l in leaves])
    Lt = Lt[np.ix_(idx, idx)]; np.fill_diagonal(Lt, sim["T"])
    return Lt


def problem(sim, rep=0, cov=0.2, beads_per_spot=5.0, min_major=60):
    """the dictionary 70_spotr.learn takes: half the cells dissociated, the other half pooled into spots"""
    import ot
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist, squareform
    spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
    m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
    rng = np.random.default_rng(10 * sim["seed"] + rep)
    n = len(sim["xy"]); perm = rng.permutation(n); use, side = perm[n // 2:], perm[: n // 2]
    xy = sim["xy"]; nf = len(sim["motility"])
    sz = np.bincount(sim["clade"], minlength=nf); lab = np.full(nf, -1); major = np.where(sz >= min_major)[0]
    lab[major] = np.arange(len(major)); minor = np.where(lab < 0)[0]; lab[minor] = len(major) + np.arange(len(minor))
    Km = len(major); cl = np.minimum(lab[sim["clade"]], Km)                         # the minor lineages pooled: clade Km
    mot = np.r_[sim["motility"][major], np.nan]
    pitches = np.linspace(20, 120, 201)
    def lattice(p):
        x0 = xy[side].min(0); g = np.floor((xy[side] - x0) / p).astype(int)
        keys, inv = np.unique(g, axis=0, return_inverse=True)
        return keys, inv.ravel(), x0
    occ = [np.bincount(lattice(p)[1]).mean() for p in pitches]
    pitch = float(pitches[int(np.argmin(np.abs(np.array(occ) - beads_per_spot)))])
    keys, inv, x0 = lattice(pitch); m = len(keys); spots = x0 + (keys + 0.5) * pitch
    K = Km + 1
    Nspot = np.zeros((m, K)); np.add.at(Nspot, (inv, cl[side]), 1.0)
    S = np.zeros((m, sim["counts"].shape[1])); np.add.at(S, inv, sim["counts"][side])
    lognorm = lambda C: np.log1p(C / np.maximum(C.sum(1, keepdims=True), 1.0) * 1e4)
    Xsc, Xst = lognorm(sim["counts"][use].astype(float)), lognorm(S)
    M = ot.dist(Xsc, Xst); M /= M.max()
    Lt = lca_time(sim)[np.ix_(use, use)]
    AGE = sim["T"] - Lt; np.fill_diagonal(AGE, 0.0)
    nb = Nspot.sum(1)
    J = rng.choice(m, max(6, int(round(cov * m))), replace=False)
    gperm = rng.permutation(len(use))
    return dict(M=M, Xsc=Xsc, Xst=Xst, use=use, D=squareform(pdist(spots)), q=nb / nb.sum(), NC=m55.clone_size(AGE).astype(float),
                rho=len(use) / ConvexHull(xy).volume, cl=cl[use], K=K, Nspot=Nspot, spots=spots, xy_true=xy[use],
                AGE=AGE, J=J, L0=None, c0=None, inB=np.r_[mot[:Km] > sim["motility"].min(), False], Kmaj=Km,
                fast_clades=np.array([lab[f] for f in sim.get("fast", []) if lab[f] < Km], int),
                spot_in_niche=(np.linalg.norm(spots - sim["niche"][0], axis=1) <= sim["niche"][1]) if sim.get("niche") else None,
                cell_in_niche=(np.linalg.norm(xy[use] - sim["niche"][0], axis=1) <= sim["niche"][1]) if sim.get("niche") else None, motility=mot, tag=f"growth_s{sim['seed']}",
                fitness=lbi(sim["parent"], sim["ntime"], sim["leaves"], np.isin(np.arange(n), use), sim["tau"])[np.argsort(np.argsort(use))], edge_spot=sim["edge_tree"].query(spots)[0], edge_true=sim["edge"][use],
                growth_fit=gperm[: len(use) // 2], growth_val=gperm[len(use) // 2:], pitch=pitch,
                S_counts=S)                                                    # the spots' raw summed counts (2026-10-20)


def fit_law(sim):
    """L and c of r^2 = L^2 + c^2 n / (2 pi rho), fitted on a simulated tumour's TRUE positions (a reference tumour)"""
    from scipy.optimize import nnls
    from scipy.spatial import ConvexHull
    spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
    m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
    xy = sim["xy"]; n = len(xy); rho = n / ConvexHull(xy).volume
    AGE = sim["T"] - lca_time(sim); np.fill_diagonal(AGE, 0.0)
    NC = m55.clone_size(AGE); iu = np.triu_indices(n, 1)
    x = NC[iu] / (2 * np.pi * rho); y = ((xy[iu[0]] - xy[iu[1]]) ** 2).sum(1) / 2.0
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 25))); b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    cnt = np.bincount(b); ok = cnt > 100
    X_ = np.bincount(b, x)[ok] / cnt[ok]; Y_ = np.bincount(b, y)[ok] / cnt[ok]; w = np.sqrt(cnt[ok])
    coef, _ = nnls(np.c_[np.ones(ok.sum()), X_] * w[:, None], Y_ * w)
    pred = np.sqrt(coef[0] + coef[1] * X_)
    return float(np.sqrt(coef[0])), float(np.sqrt(coef[1])), float(np.median(np.abs(pred - np.sqrt(Y_)) / np.sqrt(Y_)))


if __name__ == "__main__":
    import time
    from scipy.stats import spearmanr
    for m_hi in [0.02, 0.1, 0.3, 1.0]:
        t0 = time.time()
        sim = simulate(1, m_hi=m_hi)
        xy, cl, K = sim["xy"], sim["clade"], int(sim["clade"].max()) + 1
        spread = np.array([np.sqrt(((xy[cl == k] - xy[cl == k].mean(0)) ** 2).sum(1).mean()) for k in range(K)])
        hi = sim["motility"] > sim["motility"].min()
        print(f"m_hi {m_hi:4.1f}: {len(xy)} cells, T {sim['T']:.1f} cycles, radius {np.sqrt(len(xy) / np.pi) * CELL_UM:.0f} um | "
              f"LBI vs edge Spearman {spearmanr(sim['lbi'], sim['edge']).statistic:+.2f} | clade RMS spread hi {spread[hi].mean():.0f} "
              f"lo {spread[~hi].mean():.0f} um | {time.time() - t0:.0f}s", flush=True)
    L, c, e = fit_law(simulate(999, m_hi=0.02))
    print(f"law on a reference tumour (no contrast): L {L:.0f} um, c {c:.2f}, median error {100 * e:.0f}%")
