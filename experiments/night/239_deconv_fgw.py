"""Deconvolution through the map with a tree term: does tree-space concordance give the spots the right subclone proportions where
expression cannot?  2026-10-25.
FIXED BEFORE RUNNING (settings revised once after a smoke test; see 'Revised' below).
Pedro, 2026-10-24: the analysis goal is a spot deconvolution in terms of cell-state programs learned from the dissociated cells,
written through the map so that the tree carries clonal information into it ([[method-deconvolution-fgw]]); "please set this up with a
simulation example that shows that this works in obtaining a deconvolution of the spots where the spots have the correct subclone
proportions, unlike what would happen if we did not use the tree-space concordance GW part. Show it both for the case where we have a
lineage tree, and for the case where we have a clone tree, and for the case where we simply have clones. Vary the level at which each
clone has clone-specific expression. Compare with method that do not use the tree. Let's nail this." (exp-unmatched-oct12 section 84)

World: 208_'s neighbouring-region design (interspersed microenvironment, a tumour-free margin, compact clones, the whole region
dissociated, capture keep 0.5); expression whose main spatial signal is the core/border state (84_'s edge program) and blind to position
otherwise (pos_w 0); clone-specific expression = 84_'s lineage programs, Brownian along the tree with sd tree_sd in
{0, 0.15, 0.3, 0.6, 1.2} (tree_sd only scales the drift: a seed has the same tumour, tree and clones at every level); seeds 46, 47.
Subclones: 189_'s labels (the major founder lineages, >= 60 cells, and one label for the small lineages); the truth is each spot's
section tumour cells' clone mix.
The model ([[method-deconvolution-fgw]], [[decision-states-frozen-from-cells]]):
  stage A  Poisson NMF of the dissociated cells' counts, 12 programs, with a multiplicative deviation per major clone fitted on its
           traced cells (pooled over the clone); each cell's profile mu_n = (theta_n W) o exp(Delta_c(n)), at its kind's mean library;
           frozen
  pass 1   NNLS of each spot's normalised counts on the dissociated kinds' mean profiles (tumour, each microenvironment type): tumour
           share t_m and type shares e_me; per-kind spot capacities q_m f_T t_m (traced), q_m (1 - f_T) t_m (untraced), q_m e_me (each
           type), q_m the cells per spot (as in every run so far); cells of a kind share its total
  pass 2   minimise L(gamma) + lambda G(gamma) - eps H(gamma): L the multinomial NLL of each spot's counts given the library-weighted
           mixture of the profiles mapped to it; G the GW term on traced pairs; eps = EPS_REL x the doubly-centred spread of L's gradient
           at the product map; G scaled so that lambda = 1 equates the spreads of the two gradients on the traced rows at the product
           map; mirror descent (Sinkhorn on omega x gradient - (1 - omega) eps log gamma_t); continuation along LAMBDAS; lambda chosen
           by the knee of L and the tree mismatch, each min-max normalised between the ends of the grid (216_'s rule)
  trees    'lineage' the single-cell tree; 'clone tree' the tree cut at the founders (within a clone the cut, across clones the tree);
           'clones' a partition (0 within a clone, one constant across clones); distances scaled so their mean over traced pairs is the
           mean distance between spots weighted by pass 1's tumour content
No-tree comparators: the same model at lambda 0; 'NNLS clones' (each spot on the clones' and the microenvironment types' mean profiles:
per-spot deconvolution with subclones as types); exact OT on the expression cost; Tangram.  Reference: the current cost (FGW on (1 - alpha) M,
balanced, one capacity per spot, the same trees and scale, alpha by 216_'s elbow over ALPHAS_M); the floor sampling allows (the
dissociated cells at their true positions).
Readouts: clone_tv (189_.metrics: the mean over spots holding tumour cells of the total variation between the map's traced clone mix
and the section's), the same over 3 x 3 spot windows, clone-density agreement (per major clone the correlation across tumour spots of
mapped mass and true count, weighted by clone size), the traced cells' median placement, the tumour-share and microenvironment-type errors.
Expectations stated before running (per tree; both seeds):
  E71  for tree_sd >= 0.15 the model with the lineage tree, at the knee, has a lower clone_tv than every no-tree comparator
  E72  its margin over the best no-tree comparator is largest at the weakest nonzero clone signal and shrinks as the signal grows
  E73  E71 with the clone tree and with clones only (predicted: holds for the clone tree, weaker for clones only)
  E74  read: tree_sd 0 -- nothing in expression orients the clones, and a coherent layout in the wrong orientation can be worse per spot
       than a blurred one
  E75  read: the model against the current cost with the same tree
Revised before running, after a smoke test (seed 46, tree_sd 0.3; spotr_out/239_smoke.log): mirror descent at omega 0.5 oscillated at
lambda 0 (1.75 of the map's mass moved at every step and the spot NLL never settled), so the knee rule, fed a non-monotone fit, chose
lambda 0.  Now: the entropic best response to the gradient, then the step in {1, 1/2, ..., 1/1024, 0} that lowers the whole objective most
(a generalised conditional gradient: the objective never rises), J 60 steps per lambda; the knee's fit term is the lambda-0 objective
(spot NLL + eps x negative entropy), which the lambda-0 map minimises, so it rises with lambda; a 'global mix' reference (every tumour spot
given the dissociated traced cells' clone mix) is added.  Indicative only, from the unconverged smoke run: at lambda about 1 the three
trees gave a subclone error of 0.26-0.27 against 0.54-0.78 for the no-tree comparators, 0.32-0.42 for the current cost, floor 0.18.
Usage: python 239_deconv_fgw.py [procs] [SUMMARY|SMOKE]    (writes spotr_out/deconv_fgw.json)
"""
import os, sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np
HERE = Path(__file__).resolve().parent
TREE_SDS = [0.0, 0.15, 0.3, 0.6, 1.2]
SEEDS = [46, 47]
TREES = ["lineage", "clone tree", "clones"]
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 10.0]
ALPHAS_M = [0.0, 0.1, 0.2, 0.3, 0.5, 0.95]
N_PROG, NMF_ITERS = 12, 300
EPS_REL, J, IT0, IT1 = 0.02, 60, 1000, 150
ETAS = [2.0 ** -k for k in range(11)] + [0.0]
NOTREE = ["lambda 0", "NNLS clones", "OT", "Tangram"]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def stage_a(Xc, clone, iters=NMF_ITERS, seed=0):
    """Poisson NMF, X_ng ~ Poisson((theta_n W)_g E_ng), E = exp(Delta_c) on the rows of clone c (clone >= 0), 1 elsewhere; returns each
    cell's normalised profile and the clones' deviations"""
    rng = np.random.default_rng(seed); N, G = Xc.shape
    th = rng.gamma(1.0, 1.0, (N, N_PROG)) * Xc.sum(1, keepdims=True) / N_PROG; W = rng.gamma(1.0, 1.0, (N_PROG, G)); W /= W.sum(1, keepdims=True)
    E = np.ones((N, G)); groups = {c: np.where(clone == c)[0] for c in np.unique(clone[clone >= 0])}
    for it in range(iters):
        Q = Xc / ((th @ W) * E + 1e-12); th *= ((Q * E) @ W.T) / np.maximum(E @ W.T, 1e-12)
        Q = Xc / ((th @ W) * E + 1e-12); W *= (th.T @ (Q * E)) / np.maximum(th.T @ E, 1e-12)
        s = W.sum(1); W /= s[:, None]; th *= s[None, :]
        if it % 10 == 9:
            B = th @ W
            for rows in groups.values():
                E[rows] = (Xc[rows].sum(0) + 0.5) / (B[rows].sum(0) + 0.5)
    prof = (th @ W) * E
    blind = th @ W                                            # (243_) the clone-blind profiles, without the deviations
    return prof / prof.sum(1, keepdims=True), {int(c): np.log(E[rows[0]]) for c, rows in groups.items()}, blind / blind.sum(1, keepdims=True)


def pass1(X, Xc):
    """NNLS of the spots' normalised counts on the dissociated kinds' mean profiles; returns per-spot cell shares (tumour, each ME type)"""
    from scipy.optimize import nnls
    kind, mt = X["kind"], X["me_type"]
    groups = [kind == 0] + [(kind == 1) & (mt == e) for e in range(X["nT"])]; present = np.array([g.any() for g in groups])
    Pm = np.stack([Xc[g].sum(0) / Xc[g].sum() for g in groups if g.any()], 1); lib = np.array([Xc[g].sum(1).mean() for g in groups if g.any()])
    B = X["Sc"] / np.maximum(X["Sc"].sum(1, keepdims=True), 1.0)
    w = np.array([nnls(Pm, b)[0] for b in B]) / lib; w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    sh = np.zeros((len(B), len(groups))); sh[:, present] = w
    return sh, present


def pass1_clone(X, Xc):
    """(242_, 2026-10-25) pass 1 with the tumour represented by each clone's profile (traced cells of every label with >= 5 cells) instead
    of one tumour profile; the tumour share is the sum of the clone components"""
    from scipy.optimize import nnls
    kind, mt, tr = X["kind"], X["me_type"], (X["kind"] == 0) & X["traced"]
    tg = [tr & (X["cl"] == k) for k in range(X["K"]) if (tr & (X["cl"] == k)).sum() >= 5]
    mg = [(kind == 1) & (mt == e) for e in range(X["nT"])]; present = np.array([True] + [g.any() for g in mg])
    gs = tg + [g for g in mg if g.any()]; Pm = np.stack([Xc[g].sum(0) / Xc[g].sum() for g in gs], 1); lib = np.array([Xc[g].sum(1).mean() for g in gs])
    B = X["Sc"] / np.maximum(X["Sc"].sum(1, keepdims=True), 1.0)
    w = np.array([nnls(Pm, b)[0] for b in B]) / lib; w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    sh = np.zeros((len(B), 1 + X["nT"])); sh[:, 0] = w[:, : len(tg)].sum(1); sh[:, 1:][:, present[1:]] = w[:, len(tg):]
    return sh, present


def pass1_oracle(X):
    """(242_) the section's true cell shares per spot (tumour, each ME type present among the dissociated cells): a ceiling for pass 1"""
    present = np.array([True] + [((X["kind"] == 1) & (X["me_type"] == e)).any() for e in range(X["nT"])])
    sh = np.zeros((len(X["q"]), 1 + X["nT"])); sh[:, 0] = X["Ncl"].sum(1); sh[:, 1:] = X["Nme"]; sh[:, ~present] = 0.0
    return sh / np.maximum(sh.sum(1, keepdims=True), 1e-12), present


def tree_dists(X, NB, t_f, w_spot):
    """traced-by-traced target distances for the three trees, scaled so their mean over traced pairs is the w-weighted mean spot distance"""
    R = NB["region"]; ti = np.where((X["kind"] == 0) & X["traced"])[0]; gi = X["mapped"][ti]; Lt = R["Lt"][np.ix_(gi, gi)]; T = NB["T"]
    raw = {"lineage": 2.0 * (T - Lt), "clone tree": 2.0 * (T - np.minimum(Lt, t_f)), "clones": (Lt < t_f).astype(float)}
    w = w_spot / w_spot.sum(); target = float(w @ X["D"] @ w); off = ~np.eye(len(ti), dtype=bool); out = {}
    for k, d in raw.items():
        d = d.copy(); np.fill_diagonal(d, 0.0); out[k] = d * target / d[off].mean()
    return ti, out


def window(X):
    s = X["spots"]; p = X["pitch"]
    return ((np.abs(s[:, None, 0] - s[None, :, 0]) <= 1.01 * p) & (np.abs(s[:, None, 1] - s[None, :, 1]) <= 1.01 * p)).astype(float)


def readout(T, X, Nb, m189, P_clone=None):
    """composition readouts against the section's spots; P_clone (spots x K, unnormalised clone mass) replaces the map for NNLS"""
    K, Km = X["K"], X["K"] - 1; Ncl = X["Ncl"]; has = Ncl.sum(1) > 0; out = {}
    if T is not None:
        met = m189.metrics(T, X); out.update(traced_err=met["traced_err"], tumour_share_err=met.get("tumour_share_err"), me_type_tv=met.get("me_type_tv"))
        tr = (X["kind"] == 0) & X["traced"]; P_clone = T[tr].T @ np.eye(K)[X["cl"][tr]]
    nrm = lambda A: A / np.maximum(A.sum(1, keepdims=True), 1e-300)
    out["clone_tv"] = float(0.5 * np.abs(nrm(P_clone[has]) - nrm(Ncl[has])).sum(1).mean())
    Pw, Qw = Nb @ P_clone, Nb @ Ncl; hw = Qw.sum(1) > 0
    out["clone_tv_win"] = float(0.5 * np.abs(nrm(Pw[hw]) - nrm(Qw[hw])).sum(1).mean())
    ks = [k for k in range(Km) if Ncl[:, k].sum() >= 20]
    cs = [np.corrcoef(P_clone[has, k], Ncl[has, k])[0, 1] if P_clone[has, k].std() > 0 else 0.0 for k in ks]
    out["agree"] = float(np.average(cs, weights=[Ncl[:, k].sum() for k in ks])) if ks else float("nan")
    return out


def nnls_clones(X, Xc):
    """per-spot NNLS on the major clones' (traced cells), the small lineages' and the ME types' mean profiles: clone mass per spot"""
    from scipy.optimize import nnls
    K = X["K"]; tr = (X["kind"] == 0) & X["traced"]; prof, lab = [], []
    for k in range(K):
        g = tr & (X["cl"] == k)
        if g.sum() >= 5:
            prof.append(Xc[g].sum(0) / Xc[g].sum()); lab.append(k)
    for e in range(X["nT"]):
        g = (X["kind"] == 1) & (X["me_type"] == e)
        if g.any():
            prof.append(Xc[g].sum(0) / Xc[g].sum()); lab.append(-1)
    B = X["Sc"] / np.maximum(X["Sc"].sum(1, keepdims=True), 1.0); w = np.array([nnls(np.stack(prof, 1), b)[0] for b in B])
    P = np.zeros((len(B), K))
    for j, k in enumerate(lab):
        if k >= 0:
            P[:, k] += w[:, j] * X["ncell"]
    return P


def make_solver(mu, Y, D, kinds_rows, kind_of_row, idx, eps):
    """mirror descent on L + lambda G - eps H with per-kind column capacities, in JAX; returns jitted solve and numpy helpers"""
    m195 = mod("m195", "195_typed_semirelaxed.py"); jax = m195._jax(); import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    mu_j, Y_j, D_j = jnp.asarray(mu), jnp.asarray(Y), jnp.asarray(D); D2 = D_j * D_j; Yt = Y_j.sum(1); L = mu_j.sum(1)
    kor = jnp.asarray(kind_of_row); rows = [jnp.asarray(r) for r in kinds_rows]; idx_j = jnp.asarray(idx)

    def lik(G):
        P = G.T @ mu_j + 1e-300; Pm = P.sum(1)
        return -(mu_j @ (Y_j / P).T) + L[:, None] * (Yt / Pm)[None, :], -(Y_j * jnp.log(P / Pm[:, None])).sum()

    def tree(G, Wc, WPc, WQc):
        Ts = G[idx_j]; A = -2.0 * (WPc @ Ts @ D_j) + Wc @ Ts @ D2 + (WQc @ Ts.sum(1))[:, None]
        return A, (Ts * A).sum() / Ts.sum() ** 2

    def sink(Cp, loga, logq, f, gk, iters):
        def body(_, c):
            f, gk = c
            f = eps * loga - eps * logsumexp((gk[kor] - Cp) / eps, axis=1)
            gk = jnp.stack([eps * logq[k] - eps * logsumexp((f[r][:, None] - Cp[r]) / eps, axis=0) for k, r in enumerate(rows)])
            return (f, gk)
        return jax.lax.fori_loop(0, iters, body, (f, gk))

    etas = jnp.asarray(ETAS)

    def parts(G, Wc, WPc, WQc):
        """spot NLL, the tree energy (Omega-weighted, scaled by c) and eps x negative entropy"""
        P = G.T @ mu_j + 1e-300; nll = -(Y_j * jnp.log(P / P.sum(1)[:, None])).sum()
        Ts = G[idx_j]; A = -2.0 * (WPc @ Ts @ D_j) + Wc @ Ts @ D2 + (WQc @ Ts.sum(1))[:, None]
        return nll, (Ts * A).sum(), eps * jnp.sum(G * (jnp.log(jnp.maximum(G, 1e-300)) - 1.0))

    def solve(G, f, gk, loga, logq, lam, Wc, WPc, WQc, Jn, cold):
        def grad(G):
            CL, _ = lik(G); A, _ = tree(G, Wc, WPc, WQc)
            return CL + lam * 2.0 * jnp.zeros_like(CL).at[idx_j].set(A)
        if cold:
            f, gk = sink(grad(G), loga, logq, f, gk, IT0)
        def step(carry, _):
            G, f, gk = carry; C = grad(G); f, gk = sink(C, loga, logq, f, gk, IT1)
            Dl = jnp.exp((f[:, None] + gk[kor] - C) / eps) - G       # towards the entropic best response
            def phi(e):
                nll, Ge, ent = parts(G + e * Dl, Wc, WPc, WQc); return nll + lam * Ge + ent
            vals = jax.lax.map(phi, etas); e = etas[jnp.argmin(vals)]; Gn = G + e * Dl
            nll, Ge, ent = parts(Gn, Wc, WPc, WQc)
            return (Gn, f, gk), jnp.stack([nll + lam * Ge + ent, nll, Ge, ent, e, jnp.abs(Gn - G).sum()])
        (G, f, gk), trace = jax.lax.scan(step, (G, f, gk), None, length=Jn)
        return G, f, gk, trace
    return jax.jit(solve, static_argnums=(9, 10)), lik, tree, parts


def build(seed, tree_sd):
    m189, m208 = mod("m189", "189_me_sim.py"), mod("m208", "208_neighbour_sim.py")
    NB = m208.simulate_nb(seed, pattern="interspersed", margin=1.0, tree_sd=tree_sd)
    X = m208.problem_nb(NB, "all", "single", capture=dict(keep=0.5), dissoc_frac=1.0)
    t_f = float(NB["section"]["sim"]["ntime"][2 * 127 - 1]); Xc = NB["region"]["counts"][X["mapped"]].astype(float)
    return NB, X, Xc, t_f, m189


def job(args):
    seed, tree_sd = args[:2]; keep = len(args) > 2 and args[2]           # keep: also return the maps (float32), for the example figure
    opts = args[3] if len(args) > 3 else {}                                # (242_) pass1: 'nnls' (as run in 239_), 'clone', 'oracle'; model_only
    import warnings; warnings.filterwarnings("ignore")
    t0 = time.time(); m70, m89, m195 = mod("m70", "70_spotr.py"), mod("m89", "89_growth_example_figure.py"), mod("m195", "195_typed_semirelaxed.py")
    NB, X, Xc, t_f, m189 = build(seed, tree_sd); n, M_ = len(X["cl"]), len(X["q"]); tum = X["kind"] == 0; tr = tum & X["traced"]; Km = X["K"] - 1
    out = dict(seed=seed, tree_sd=tree_sd, n=n, spots=M_, traced=int(tr.sum()), arms={}, sweeps={}, timing={}); maps = {}
    # stage A: profiles frozen from the dissociated cells
    clone = np.where(tr & (X["cl"] < Km), X["cl"], -1); prof, dev, prof_b = stage_a(Xc, clone)
    halves = []
    for c in dev:                                             # split-half reliability of each major clone's pooled deviation
        rows = np.where(clone == c)[0]; a, b = rows[::2], rows[1::2]; base = Xc[tum & ~X["traced"]].sum(0) + 0.5
        la = np.log((Xc[a].sum(0) + 0.5) / Xc[a].sum()) - np.log(base / base.sum()); lb = np.log((Xc[b].sum(0) + 0.5) / Xc[b].sum()) - np.log(base / base.sum())
        halves.append(np.corrcoef(la, lb)[0, 1])
    out["clone_signature_split_half"] = float(np.mean(halves))
    # pass 1: kind shares, capacities, row masses
    sh, present = {"nnls": pass1, "clone": pass1_clone}[opts.get("pass1", "nnls")](X, Xc) if opts.get("pass1", "nnls") != "oracle" else pass1_oracle(X)
    kinds = [tr, tum & ~X["traced"]] + [(X["kind"] == 1) & (X["me_type"] == e) for e in range(X["nT"]) if present[1 + e]]
    sh_me = sh[:, 1:][:, present[1:]]; t_hat = sh[:, 0]; fT = tr.sum() / tum.sum(); q = X["q"]
    qk = np.vstack([q * fT * t_hat, q * (1 - fT) * t_hat] + [q * sh_me[:, j] for j in range(sh_me.shape[1])])
    kinds_rows = [np.where(k)[0] for k in kinds]; kind_of_row = np.zeros(n, int)
    for j, r in enumerate(kinds_rows):
        kind_of_row[r] = j
    a = np.array([qk[kind_of_row[i]].sum() / len(kinds_rows[kind_of_row[i]]) for i in range(n)])
    true_t = X["Ncl"].sum(1) / np.maximum(X["ncell"], 1); out["pass1_tumour_share_err"] = float(np.abs(t_hat - true_t).mean())
    lib = np.zeros(n)
    for r in kinds_rows:
        lib[r] = Xc[r].sum(1).mean()
    mu = prof * lib[:, None]; Y = X["Sc"]
    # the product map, the entropic level and the trees' scale
    logq = np.log(np.maximum(qk, 1e-300)); logG0 = np.log(a)[:, None] + logq[kind_of_row] - np.log(qk.sum(1))[kind_of_row][:, None]
    G0 = np.exp(logG0); P0 = G0.T @ mu; CL0 = -(mu @ (Y / P0).T) + mu.sum(1)[:, None] * (Y.sum(1) / P0.sum(1))[None, :]
    CLc = CL0                                                # (243_) eps and c from the gradient with the clone-blind profiles
    if opts.get("eps") == "blind":
        mb = prof_b * lib[:, None]; Pb = G0.T @ mb; CLc = -(mb @ (Y / Pb).T) + mb.sum(1)[:, None] * (Y.sum(1) / Pb.sum(1))[None, :]
    eps = EPS_REL * float(m70.dc_std(CLc)); ti, dt = tree_dists(X, NB, t_f, q * t_hat)
    solve, lik, treef, parts = make_solver(mu, Y, X["D"], kinds_rows, kind_of_row, ti, eps); Nb = window(X)
    import jax.numpy as jnp
    Om = 1.0 - np.eye(len(ti)); Ts0 = G0[ti]
    def tree_arrays(d):
        A0 = -2.0 * ((Om * d) @ Ts0 @ X["D"]) + Om @ Ts0 @ (X["D"] ** 2) + ((Om * d * d) @ Ts0.sum(1))[:, None]
        c = float(m70.dc_std(CLc[ti]) / m70.dc_std(2.0 * A0)); return jnp.asarray(c * Om), jnp.asarray(c * Om * d), jnp.asarray(c * Om * d * d), c
    out["timing"]["setup"] = time.time() - t0
    # pass 2: lambda 0 (no tree), then each tree's sweep from it
    W0 = tree_arrays(dt["lineage"])
    lg, f, gk, trc = solve(jnp.asarray(G0), jnp.zeros(n), jnp.zeros((len(kinds_rows), M_)), jnp.asarray(np.log(a)), jnp.asarray(logq), 0.0, *W0[:3], J, True)
    start = (lg, f, gk); T0map = np.asarray(lg); out["arms"]["lambda 0"] = readout(T0map, X, Nb, m189)
    rowH = -(T0map / a[:, None] * np.log(np.maximum(T0map / a[:, None], 1e-300))).sum(1); out["lambda0_effective_spots"] = float(np.median(np.exp(rowH)))
    gm = np.bincount(X["cl"][tr], minlength=X["K"]).astype(float); out["arms"]["global mix"] = readout(None, X, Nb, m189, P_clone=np.outer(X["Ncl"].sum(1) > 0, gm))
    out["lambda0_trace"] = np.asarray(trc).tolist(); out["timing"]["lambda0"] = time.time() - t0
    for tree in TREES:
        Wc, WPc, WQc, c = tree_arrays(dt[tree]); state = start; sw = {}
        for lam in LAMBDAS:
            if lam == 0.0:
                Tm = T0map; tr_ = trc
            else:
                lg, f, gk, tr_ = solve(*state, jnp.asarray(np.log(a)), jnp.asarray(logq), float(lam), Wc, WPc, WQc, J, False)
                state = (lg, f, gk); Tm = np.asarray(lg)
            nll, Ge, ent = (float(v) for v in parts(jnp.asarray(Tm), Wc, WPc, WQc)); Qm = float(np.asarray(treef(jnp.asarray(Tm), Wc, WPc, WQc)[1]))
            tr_ = np.asarray(tr_); sw[str(lam)] = dict(nll=nll, ent=ent, E=nll + ent, Q=Qm / c, last_step=float(tr_[-1, 4]), last_change=float(tr_[-1, 5]),
                                                      **readout(Tm, X, Nb, m189))
            if keep:
                maps[(tree, lam)] = Tm.astype(np.float32)
        lo, hi = str(LAMBDAS[0]), str(LAMBDAS[-1]); Ln = lambda k: (sw[k]["E"] - sw[lo]["E"]) / max(sw[hi]["E"] - sw[lo]["E"], 1e-12)
        Qn = lambda k: (sw[k]["Q"] - sw[hi]["Q"]) / max(sw[lo]["Q"] - sw[hi]["Q"], 1e-12); knee = min(sw, key=lambda k: Ln(k) + Qn(k))
        out["sweeps"][tree] = sw; out["arms"][f"model, {tree}"] = dict(sw[knee], knee=float(knee))
        out["timing"][tree] = time.time() - t0
    if opts.get("model_only"):
        out["timing"]["total"] = time.time() - t0; out["pass1"] = opts.get("pass1", "nnls"); return out
    # no-tree comparators and references
    P_nnls = nnls_clones(X, Xc); out["arms"]["NNLS clones"] = readout(None, X, Nb, m189, P_clone=P_nnls)
    BL = m89.baselines(X, seed)
    out["arms"]["OT"] = readout(BL["Expression only (OT)"], X, Nb, m189); out["arms"]["Tangram"] = readout(BL["Tangram"], X, Nb, m189)
    out["arms"]["floor (true positions)"] = readout(m89.oracle(X), X, Nb, m189)
    epsM = m195.EPS_REL * float(m70.dc_std(X["M"])); aM = np.full(n, 1.0 / n)
    for tree in TREES:
        Wf = np.zeros((n, n)); Wf[np.ix_(ti, ti)] = Om; WPf = np.zeros((n, n)); WPf[np.ix_(ti, ti)] = Om * dt[tree]
        Xt = dict(X); Xt.update(W=Wf, WP=WPf); cM = m189.tree_scale(Xt); Tp = None; E, Qd, rd = {}, {}, {}
        for al in ALPHAS_M:
            Tp = m195.fgw_sinkhorn(X["M"], Wf, WPf, X["D"], aM, X["q"], al, cM, epsM, None, T_init=Tp)[0]
            E[al] = float((X["M"][tr] * Tp[tr]).sum()); Ts = Tp[ti]
            A = -2.0 * ((Om * dt[tree]) @ Ts @ X["D"]) + Om @ Ts @ (X["D"] ** 2) + ((Om * dt[tree] ** 2) @ Ts.sum(1))[:, None]
            Qd[al] = float((Ts * A).sum() / Ts.sum() ** 2); rd[al] = readout(Tp, X, Nb, m189)
        lo, hi = ALPHAS_M[0], ALPHAS_M[-1]
        el = min(ALPHAS_M, key=lambda x: (E[x] - E[lo]) / max(E[hi] - E[lo], 1e-12) + (Qd[x] - Qd[hi]) / max(Qd[lo] - Qd[hi], 1e-12))
        out["arms"][f"current cost, {tree}"] = dict(rd[el], elbow=el); out["sweeps"][f"current cost, {tree}"] = {str(k): v for k, v in rd.items()}
        if keep and tree == "lineage":
            Tp = None
            for al in ALPHAS_M[: ALPHAS_M.index(el) + 1]:      # the elbow's map again (the continuation is deterministic)
                Tp = m195.fgw_sinkhorn(X["M"], Wf, WPf, X["D"], aM, X["q"], al, cM, epsM, None, T_init=Tp)[0]
            maps[("current cost", "lineage")] = Tp.astype(np.float32)
        del Wf, WPf, Xt
    out["timing"]["total"] = time.time() - t0
    if keep:
        maps.update({"OT": BL["Expression only (OT)"], "Tangram": BL["Tangram"], "floor": m89.oracle(X), "lambda 0": T0map})
        out.update(maps=maps, P_nnls=P_nnls, X=X, NB=NB, kinds_rows=kinds_rows, t_hat=t_hat)
    return out


def summary(res):
    by = {(r["seed"], r["tree_sd"]): r for r in res}
    arms = [f"model, {t}" for t in TREES] + NOTREE + [f"current cost, {t}" for t in TREES] + ["global mix", "floor (true positions)"]
    for key in ["clone_tv", "clone_tv_win", "agree"]:
        print(f"######## {key} (mean of seeds {SEEDS}; knee lambda / elbow alpha in brackets for the tree arms)")
        print("   tree_sd       : " + " | ".join(f"{a:>22s}" for a in arms))
        for sd in TREE_SDS:
            rr = [by[(s, sd)] for s in SEEDS if (s, sd) in by]
            if not rr:
                continue
            cells = []
            for a in arms:
                v = np.mean([r["arms"][a][key] for r in rr]); tag = ""
                par = "knee" if a.startswith("model") else ("elbow" if a.startswith("current") else None)
                if par:
                    tag = " [" + "/".join("%g" % r["arms"][a][par] for r in rr) + "]"
                cells.append(f"{v:.3f}{tag}")
            print(f"   {sd:<13g} : " + " | ".join(f"{c:>22s}" for c in cells))
    print("######## the expectations (clone_tv, per seed)")
    for t in TREES:
        ok, margins = [], {}
        for sd in TREE_SDS:
            for s in SEEDS:
                if (s, sd) not in by:
                    continue
                r = by[(s, sd)]; best = min(r["arms"][a]["clone_tv"] for a in NOTREE); m = best - r["arms"][f"model, {t}"]["clone_tv"]
                margins.setdefault(sd, []).append(m)
                if sd >= 0.15:
                    ok.append(m > 0)
        print(f"   {t:10s}: below every no-tree comparator in {sum(ok)}/{len(ok)} worlds with tree_sd >= 0.15; margin over the best no-tree comparator by tree_sd: "
              + ", ".join(f"{sd:g}: {np.mean(v):+.3f}" for sd, v in margins.items()))
    print("######## read: the knee against the truth-best lambda (clone_tv, mean of seeds)")
    for t in TREES:
        line = []
        for sd in TREE_SDS:
            rr = [by[(s_, sd)] for s_ in SEEDS if (s_, sd) in by]
            if rr:
                best = [min(r["sweeps"][t].values(), key=lambda v: v["clone_tv"])["clone_tv"] for r in rr]
                bl = [min(r["sweeps"][t], key=lambda k: r["sweeps"][t][k]["clone_tv"]) for r in rr]
                line.append(f"{sd:g}: knee {np.mean([r['arms'][f'model, {t}']['clone_tv'] for r in rr]):.3f} / best {np.mean(best):.3f} at lambda {'/'.join(bl)}")
        print(f"   {t:10s}: " + "; ".join(line))
    print("   pass 1 tumour-share error: " + ", ".join(f"{r['seed']}/{r['tree_sd']:g}: {r['pass1_tumour_share_err']:.3f}" for r in res))
    print("   clone signature split-half correlation by tree_sd: " + ", ".join(f"{sd:g}: {np.mean([by[(s, sd)]['clone_signature_split_half'] for s in SEEDS if (s, sd) in by]):.2f}" for sd in TREE_SDS))


def smoke():
    """one world, timing and sanity checks before the settings are fixed"""
    t0 = time.time(); r = job((46, 0.3)); print(json.dumps({k: v for k, v in r.items() if k not in ("sweeps", "lambda0_trace")}, indent=1, default=float))
    tr = np.array(r["lambda0_trace"]); print("lambda 0 trace (objective, step, change):", [(f"{x[0]:.8g}", f"{x[4]:.3g}", f"{x[5]:.2e}") for x in tr[::5]])
    for t in TREES:
        print(t, {k: (round(v["clone_tv"], 3), round(v["E"], 1), round(v["Q"], 1), f"{v['last_step']:.2g}", f"{v['last_change']:.1e}") for k, v in r["sweeps"][t].items()})
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    t0 = time.time(); procs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    fn = HERE / "spotr_out" / "deconv_fgw.json"
    if len(sys.argv) > 2 and sys.argv[2] == "SMOKE":
        smoke(); sys.exit()
    if len(sys.argv) > 2 and sys.argv[2] == "SUMMARY":
        res = json.load(open(fn))
    else:
        with get_context("spawn").Pool(procs) as pool:
            res = pool.map(job, [(s, sd) for sd in TREE_SDS for s in SEEDS])
        json.dump(res, open(fn, "w"), indent=1, default=float)
    summary(res)
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
