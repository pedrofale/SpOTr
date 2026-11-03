"""354_: the lineage tree as a nested decomposition of each clone's spatial scatter -- one learned spread per level, no pair counted twice.
2026-10-30.  Pedro approved ("Yes go ahead") after 347_/348_: replacing the clone compaction by tree-distance matching made the lineage worse
than the clone tree (347_: 0.451 against 0.426 per-spot error, 2/12), and adding the matching to the compaction (348_) counted the same
within-clade distances twice -- its gain equals doubling the clone-label weight (0.308 against 0.302) and it lost PEtracer M3-4 (0.267
against 0.332).
Model (the map deterministic, as 344_; clone labels and clone tree exactly 347_'s):
  lineage  within clone k, the clone's subtree (average linkage on the cells' tree distances) cut into nested partitions Pi_1 > Pi_2 > Pi_3
           (at most 2, 4, 8 groups, groups below MIN_GROUP cells merged into their nearest sibling); with S_t the scatter around the group
           centres at level t (S_0 the clone's), the scatter splits into non-overlapping parts P_t = S_(t-1) - S_t (t = 1..T) and P_(T+1) = S_T,
           and  G_k = sum_t P_t / (2 tau_t^2),  each tau_t^2 ~ InvGamma(1, b0), set to its posterior mean (b0 + P_t / 2) / dof_t, dof_t the
           number of new groups at that level (cells minus groups for the last); in GW form a nested masking matrix -- a pair sharing a level-t
           group of n_g cells carries weight (1/(2 tau_(t+1)^2) - 1/(2 tau_t^2)) / n_g, summed over the levels it shares.  Equal spreads at
           every level give clone labels exactly; plus the clone tree's cross-clade matching.
Readouts besides 347_'s: every map saved; in simulations and PEtracer's half designs, the per-spot error at sub-clade resolution (levels 1-3)
against the dissociated cells' own true spots.
FIXED BEFORE RUNNING:
  E203  simulations, 12 oriented world-seeds: lineage below clone tree in >= 9 of 12 at clone level (per-spot subclone error), and in
        >= 10 of 12 at sub-clade level 2
  E204  PEtracer: lineage >= clone labels - 0.01 in >= 3 of 4 designs (per-clade PCC), above no tree in 4 of 4
  E205  embryo: lineage's clone-tissue agreement >= clone labels' - 0.005
Usage: python 354_c2l_cgw_anova.py [sim|petracer|embryo|htan ...]    (writes spotr_out/spotr_c2l_cgw_anova/<problem>.json)
347_'s docstring follows.
347_: the clade-aware GW term split into within- and cross-clade parts, each with its own learned weight, and the weights made
consistent with the spreads they are learned from; every annotated cell type kept and no gene cap.  2026-10-30.
Why (Pedro, 2026-10-30, after 346_): PEtracer's elongated clade is lost and clades sit in wrong places -- 346_'s clades come out at
0.56 of their true spread on M3-4 (per-clade PCC 0.265) where 323_'s clone clouds kept them at 0.88 (0.335).  Cause: 346_ puts
w_k = 1/(2 n_k s_k^2) on every ORDERED pair, 319_/323_ on every unordered pair, so 346_'s term is twice the Gaussian whose spread s_k it
learns; each round compacts the clades further.  And: "we need to calibrate our model so that using the lineage tree is always the
best (since its more information), followed by clone tree, and then clone labels, and then vanilla cell2location."  HTAN: hepatocytes
(4 nuclei) were merged into "other" (MIN_TYPE 30) and the liver genes fell outside the 500-gene cap, so tumour nuclei filled the liver.
Model (the map deterministic, as 344_):  G(phi) = sum over ordered traced pairs of E[ W1 D^2 - 2 W2 D ] + const, built from
  clone labels  within clade k: compaction, weight w_k = 1/(4 n_k s_k^2)            [target distance 0]
  clone tree    the same, plus across clades: v_0 (a_0 C^T - D)^2, v_0 = 1/(4 n_T t_0^2)   [clone-to-clone distances]
  lineage       across clades as the clone tree; within clade k the compaction is REPLACED by v_k (a_k C^T - D)^2, v_k = 1/(4 n_k t_k^2)
                [the cells' tree distances: sister cells near, cousins farther, the clade never below what its tree allows]
  each spread (s_k^2, t_k^2, t_0^2) ~ InvGamma(1, b0) and set to its posterior mean given the map after each round, as 344_;
  the scales a_k, a_0 by least squares on E D.  Each resolution adds information to the one before.
Also: MIN_TYPE 1 (every annotated cell type its own reference type), no gene cap (all fitted genes), each arm's abundances saved.
FIXED BEFORE RUNNING:
  E193  simulations, 12 oriented world-seeds: lineage < clone tree in >= 10 of 12; clone tree <= clone labels + 0.005 in >= 10 of 12;
        clone labels < no tree in 12 of 12 (no tree < cell2location alone holds already)
  E194  PEtracer: clone labels' per-clade PCC on M3-4 >= 0.32 (323_'s level); lineage above no tree in 4 of 4; lineage >= clone labels
        - 0.01 in >= 3 of 4; the second-largest clade's (the stripe) learned spread with the lineage tree >= 0.8 of its true spread
  E195  HTAN HT260C1: SpOTr's tumour mass in non-tumour spots below 0.15 (346_: 0.29); agreement with the copy-number signal >= 0.66
  E196  embryo: clone-tissue agreement of the lineage tree above no tree's and clone labels'
Usage: python 347_c2l_cgw_split.py [sim|embryo|petracer|htan ...]    (writes spotr_out/spotr_c2l_cgw_split/<problem>.json)
346_'s docstring follows.
346_: 344_ with the map's line search fixed, rerun on everything (simulations, embryo, PEtracer, HTAN).  2026-10-30.
Why: on the embryo every GCG step of every tree arm was rejected in 344_ (step 0 in all 96; 345_), and the no-tree fit stalled from round 6
while still improving, so the clone-label and lineage maps were the warm start, identical to six decimals.  Two numerical defects: the step
grid stopped at 0.03, and each candidate's objective was a float32 total (~2e6 on the embryo) compared with another, whose rounding hides the
gains of a small step.  Fix (numerical only; the model, the weights and every setting are 344_'s): each candidate is scored by its change from
not moving, summed term by term -- the likelihood through log1p of the change of each spot mean, which is affine in the map, and the entropy
elementwise -- and the grid runs down to 0.001.
FIXED BEFORE RUNNING: 344_'s E185-E188 unchanged, now as E189-E192 on these fits:
  E189  simulations, 12 oriented world-seeds: each tree arm below the no-tree arm in >= 11 of 12, the lineage tree <= clone labels + 0.01 in >= 9 of 12
  E190  embryo: clone-tissue agreement of the lineage tree above the no-tree arm's and >= clone labels'
  E191  PEtracer: per-clade PCC of the lineage tree above the no-tree arm's in 4 of 4 designs and >= clone labels - 0.01 in >= 3 of 4
  E192  HTAN HT260C1: agreement with the spots' copy-number signal of clone labels above the no-tree arm's
  and: steps are taken in the tree arms on the embryo (share of nonzero steps > 0.5).
344_'s docstring follows.
SpOTr with the clade-aware Gromov-Wasserstein term and clade weights learned from a prior (option 1) -- the map a deterministic function;
on the simulations, the embryo, PEtracer and HTAN.  2026-10-30.
Pedro, 2026-10-30, after 342_ (fixed weights from one rule: best in simulation, but on PEtracer and the embryo the tree made the maps worse
than without it -- clades pulled to about half their true spread): "Let's go with option 1" (clade weights with a prior from which a posterior
is learned), and "We don't need to compare to SpOTr-DM, right? That's old and we discarded that approach."
Model (342_'s, the weights learned):
  phi*(theta, w) = argmin_{phi in Pi(theta)}  -log p(Y | theta, phi) + G(phi; w) + sum phi log phi,   cell2location's likelihood (324_),
  G(phi; w) = sum_k w_k sum_{n<n' in k} E D^2(pi_n, pi_n')                                   [within each clade, target 0: compaction]
            + w_T sum_{n<n'} E (s C^T_nn' - D(pi_n, pi_n'))^2                                  [clone tree / lineage tree only: all traced pairs]
  clade weights w_k = 1 / (2 n_k sigma_k^2), sigma_k^2 ~ InvGamma(1, b0): after each map round, sigma_k^2 = (b0 + S_k / (2 n_k)) / (n_k - 1),
  S_k = sum_{n<n' in k} E D^2 under the map (321_'s clone clouds);  tree weight w_T = 1 / (2 n_T tau^2), tau^2 ~ InvGamma(1, b0):
  tau^2 = (b0 + R / (2 n_T)) / (n_T - 1), R = sum_{n<n'} E (s C^T - D)^2, and s by least squares on E D (C^T the clone tree's or the
  lineage's distances, as 342_).  Clone labels: the first term only.  The weights start at the map fitted without the tree.
Comparators: no tree, cell2location alone, DestVI alone, Tangram alone, exact OT, the floor (SpOTr-DM dropped).
FIXED BEFORE RUNNING:
  E185  simulations, 12 oriented world-seeds: per-spot subclone error of each tree arm below the no-tree arm's in >= 11 of 12, and the lineage
        tree <= clone labels + 0.01 in >= 9 of 12
  E186  embryo: clone-tissue agreement of the lineage tree above the no-tree arm's and >= clone labels'
  E187  PEtracer: per-clade PCC of the lineage tree above the no-tree arm's in 4 of 4 designs and >= clone labels - 0.01 in >= 3 of 4
  E188  HTAN HT260C1: agreement with the spots' copy-number signal of clone labels above the no-tree arm's
  read  the learned clade spreads against the truth (simulations, PEtracer); w_T and s
Usage: python 346_c2l_cgw_ls.py [sim|embryo|petracer|htan ...]    (writes spotr_out/spotr_c2l_cgw_post/<problem>.json)
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE / "spotr_out"; OUT = O / "spotr_c2l_cgw_anova"; LEVELS, MIN_GROUP = 3, 10; DATA = HERE.parents[1] / "data"
N_OUTER, J, IT0, IT1, ADAM0, ADAM1, G_MAX, MIN_TYPE, THETA = 8, 12, 300, 100, 1000, 200, 10 ** 9, 1, 0.5
ETAS = [1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0]
SIMS = [(w, s) for s in [46, 47] for w in ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]]; PET = ["half:M3-4", "half:M3-1", "half:M3-2", "serial:M3-1>M3-2"]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def fit(Y, Gme, gu, fT, Xh, kc, D, xy, clone=None, CT=None, init=None, mode="clones", lev=None):
    """324_'s multi-kind cell2location fit with the clade-aware GW term whose weights are learned (see the docstring); clone=None: no tree;
    CT: the tree's distances between traced cells (None: clone labels, the within-clade term only); init = (theta, phi) warm start"""
    import jax, jax.numpy as jnp, optax
    from functools import partial
    from jax.scipy.special import logsumexp, gammaln
    S, G = Y.shape; E = Gme.shape[0]; T = gu.shape[0]; n = Xh.shape[0]; rows = [np.where(kc == c)[0] for c in range(T)]; nc = np.array([len(r) for r in rows], float)
    f32 = lambda a: jnp.asarray(np.asarray(a, np.float32))
    Yj, Gj, guj, fTj, Xj, kcj = f32(Y), f32(Gme if E else np.zeros((1, G))), f32(gu), f32(fT), f32(Xh), jnp.asarray(kc); mc = float(Y.mean())
    xc = np.asarray(xy, float) - np.asarray(xy, float).mean(0); Lj, sqj, Dj = f32(xc), f32((xc ** 2).sum(1)), f32(D); b0 = 1e-3 * float(np.median(D[D > 0])) ** 2
    th = init[0] if init is not None else dict(logN=f32(np.full(S, np.log(10.0))), z=f32(np.zeros((S, E + T))), logm=f32(np.zeros(G)), logy=f32(np.zeros(S)),
                                               logs=f32(np.full(G, np.log(1e-3 * mc))), logr=f32(np.full(G, np.log(10.0))))
    def parts(th):
        w = jnp.exp(th["logN"])[:, None] * jax.nn.softmax(th["z"], axis=1); return w[:, :E], w[:, E:]
    def caps_of(th):
        wt = parts(th)[1]; return (wt / wt.sum(0, keepdims=True) * f32(nc)[None, :]).T
    def mu_of(th, phi, caps):
        we, wt = parts(th); A = (wt * fTj[None, :] / jnp.maximum(caps.T, 1e-12))[:, kcj].T
        expr = (we @ Gj if E else 0.0) + (wt * (1.0 - fTj)[None, :]) @ guj + (phi * A).T @ Xj
        return (jnp.exp(th["logm"])[None, :] * expr + jnp.exp(th["logs"])[None, :]) * jnp.exp(th["logy"])[:, None]
    def loglik(th, phi, caps):
        mu = mu_of(th, phi, caps); r = jnp.exp(th["logr"])[None, :]
        return jnp.sum(gammaln(Yj + r) - gammaln(r) - gammaln(Yj + 1.0) + r * jnp.log(r / (r + mu)) + Yj * jnp.log(mu / (r + mu) + 1e-30))
    def logprior(th):
        return (jnp.sum(9.0 * th["logN"] - jnp.exp(th["logN"])) + jnp.sum(19.0 * th["logy"] - 20.0 * jnp.exp(th["logy"])) - 0.5 * jnp.sum(th["logm"] ** 2)
                - jnp.sum((th["logs"] - np.log(1e-3 * mc)) ** 2) / 8.0 - jnp.sum((th["logr"] - np.log(10.0)) ** 2) / 8.0)
    opt = optax.adam(0.02); st = opt.init(th)
    @partial(jax.jit, static_argnums=4)
    def adam(th, st, phi, caps, steps):
        def body(c, _):
            th, st = c; g = jax.grad(lambda t: -(loglik(t, phi, caps) + logprior(t)))(th); up, st = opt.update(g, st, th); return (optax.apply_updates(th, up), st), None
        (th, st), _ = jax.lax.scan(body, (th, st), None, length=steps); return th, st
    def sink(Cst, logcap, f, g, iters):
        def body(_, s):
            f, g = s; f = -logsumexp(g[kcj] - Cst, axis=1)
            g = jnp.stack([logcap[c] - logsumexp(f[rows[c], None] - Cst[rows[c]], axis=0) for c in range(T)]); return (f, g)
        return jax.lax.fori_loop(0, iters, body, (f, g))
    def pD2(p):
        return (p @ sqj)[:, None] + p.sum(1)[:, None] * sqj[None, :] - 2.0 * (p @ Lj) @ Lj.T
    ent = lambda p: jnp.sum(p * jnp.log(jnp.maximum(p, 1e-30)))
    @jax.jit
    def gcg(phi, f, g, th, caps, W1, W2, rr):
        logcap = jnp.log(jnp.maximum(caps, 1e-30))
        def step(c, _):
            phi, f, g = c; Lt = 2.0 * rr[:, None] + 2.0 * (W1 @ pD2(phi)) - 4.0 * (W2 @ (phi @ Dj))
            Cst = -jax.grad(lambda p: loglik(th, p, caps))(phi) + Lt; f, g = sink(Cst, logcap, f, g, IT1); dl = jnp.exp(f[:, None] + g[kcj] - Cst) - phi
            q1 = jnp.sum(dl * Lt); q2 = jnp.sum(dl * (W1 @ pD2(dl))) - 2.0 * jnp.sum(dl * (W2 @ (dl @ Dj)))
            mu0 = mu_of(th, phi, caps); dmu = mu_of(th, phi + dl, caps) - mu0; r = jnp.exp(th["logr"])[None, :]       # the spot means are affine in the map
            def dobj(e):                                                                            # objective change from not moving, term by term
                dll = jnp.sum(Yj * jnp.log1p(e * dmu / mu0) - (Yj + r) * jnp.log1p(e * dmu / (r + mu0)))
                pe = jnp.maximum(phi + e * dl, 1e-30); p0 = jnp.maximum(phi, 1e-30); return -dll + e * q1 + e * e * q2 + jnp.sum(pe * jnp.log(pe) - p0 * jnp.log(p0))
            vals = jnp.stack([dobj(e) for e in ETAS]); e = jnp.asarray(ETAS)[jnp.argmin(vals)]
            return (phi + e * dl, f, g), jnp.stack([vals.min(), e, vals[-1] - vals.min()])
        (phi, f, g), tr = jax.lax.scan(step, (phi, f, g), None, length=J); return phi, f, g, tr
    @jax.jit
    def moments(phi):                                                                               # E D and E D^2 between traced cells (cells independent)
        return phi @ Dj @ phi.T, pD2(phi) @ phi.T
    use = clone is not None; on = (clone >= 0) if use else np.zeros(n, bool); Kc = int(clone.max()) + 1 if use else 0
    same = on[:, None] & on[None, :] & ((clone[:, None] == clone[None, :]) if use else False); np.fill_diagonal(same, False); pair = on[:, None] & on[None, :]; np.fill_diagonal(pair, False)
    nk = np.bincount(clone[on], minlength=Kc).astype(float) if use else np.zeros(0); nT = float(on.sum()); c0 = np.maximum(clone, 0) if use else None
    diff = pair & ~same
    def ls(c, ed, ed2):
        """least-squares scale of the tree distances to E D, and the summed residual E (a c - D)^2"""
        a = float((c * ed).sum() / max((c * c).sum(), 1e-30)); return a, float((a * a * c * c - 2.0 * a * c * ed + ed2).sum())
    def weights(phi):
        """posterior means of the spreads at the current map; W1 (on E D^2), W2 (on E D) and r over ordered pairs, and a record"""
        ED, ED2 = (np.asarray(x, float) for x in moments(jnp.asarray(phi, jnp.float32)))
        W1 = np.zeros((n, n)); W2 = np.zeros((n, n)); A = np.zeros((n, n)); s2 = np.zeros(Kc); rec = {}
        for k in range(Kc):
            i = np.where(clone == k)[0]; m = len(i)
            if m < 2:
                continue
            iu = np.triu_indices(m, 1); ed, ed2 = ED[np.ix_(i, i)][iu], ED2[np.ix_(i, i)][iu]
            s2[k] = (b0 + ed2.sum() / (2.0 * m)) / (m - 1.0)                                           # the clade's spread (read out in every arm)
            if mode == "lineage":
                parts = [np.zeros(m, int)] + [lev[i, t] for t in range(lev.shape[1])]; keep = [parts[0]]
                for q in parts[1:]:
                    if len(np.unique(q)) > len(np.unique(keep[-1])):
                        keep.append(q)
                E2 = np.zeros((m, m)); E2[iu] = ed2; E2 = E2 + E2.T
                def scat(q):
                    return sum(E2[np.ix_(q == g, q == g)].sum() / 2.0 / (q == g).sum() for g in np.unique(q))
                S_ = [scat(q) for q in keep]; J_ = [len(np.unique(q)) for q in keep]; T_ = len(keep) - 1
                P_ = [S_[t - 1] - S_[t] for t in range(1, T_ + 1)] + [S_[T_]]; dof = [J_[t] - J_[t - 1] for t in range(1, T_ + 1)] + [m - J_[T_]]
                tau2 = [(b0 + max(P, 0.0) / 2.0) / max(df, 1) for P, df in zip(P_, dof)]
                blk = np.zeros((m, m))
                for t, q in enumerate(keep):
                    c = 1.0 / (2.0 * tau2[t]) - (1.0 / (2.0 * tau2[t - 1]) if t >= 1 else 0.0)
                    for g in np.unique(q):
                        sel = q == g; blk[np.ix_(sel, sel)] += c / sel.sum()
                W1[np.ix_(i, i)] = 0.5 * blk; rec.setdefault("tau2", []).append(tau2)
            else:
                W1[np.ix_(i, i)] = 1.0 / (4.0 * m * s2[k])
        rec["s2"] = s2.tolist()
        if mode in ("clone_tree", "lineage"):
            iu = np.triu(diff, 1); a0, R0 = ls(CT[iu], ED[iu], ED2[iu]); t20 = (b0 + R0 / (2.0 * nT)) / max(nT - 1.0, 1.0); v0 = 1.0 / (4.0 * nT * t20)
            W1 = np.where(diff, v0, W1); A = np.where(diff, a0 * CT, A); rec.update(a0=a0, t20=t20, v0=v0)
        np.fill_diagonal(W1, 0.0); W2 = W1 * A; return f32(W1), f32(W2), f32((W2 * A).sum(1)), rec
    caps = caps_of(th); f, g = jnp.zeros(n, jnp.float32), jnp.zeros((T, S), jnp.float32)
    if init is None:
        f, g = sink(jnp.zeros((n, S), jnp.float32), jnp.log(caps), f, g, IT0); phi = jnp.exp(f[:, None] + g[kcj]); th, st = adam(th, st, phi, caps, ADAM0)
    else:
        phi = f32(init[1])
    zero = (f32(np.zeros((n, n))), f32(np.zeros((n, n))), f32(np.zeros(n)), {})
    W1, W2, rr, rec = weights(np.asarray(phi)) if use else zero; trace = []
    for it in range(N_OUTER):
        caps = caps_of(th); lphi = jnp.log(jnp.maximum(phi, 1e-30)); f, g = sink(-lphi, jnp.log(caps), f, g, IT0); phi = jnp.exp(f[:, None] + g[kcj] + lphi)
        phi, f, g, tr = gcg(phi, f, g, th, caps, W1, W2, rr); th, st = adam(th, st, phi, caps, ADAM1)
        if use:
            W1, W2, rr, rec = weights(np.asarray(phi))
        tr = np.asarray(tr, float); trace.append(dict(loglik=float(loglik(th, phi, caps)), steps=tr[:, 1].tolist(), gain=tr[:, 2].tolist(), **rec))
        print(f"      outer {it}: loglik {trace[-1]['loglik']:.5g}, steps {np.round(tr[:, 1], 2).tolist()}, gain {tr[:, 2].sum():.3g}", flush=True)
    return dict(phi=np.asarray(phi, float), th=th, trace=trace, weights=rec)


def tree_mats(kind, clone, Kc, d_within, Dk, target, area):
    """Omega (W1), Omega o C^T (W2), (Omega o C^T^2) 1 (r) for one resolution; d_within: the cells' tree distances (used within clones),
    Dk: clone-to-clone distances (between clones); C^T rescaled so its mean over traced pairs equals `target`"""
    n = len(clone); on = clone >= 0; c0 = np.maximum(clone, 0); same = on[:, None] & on[None, :] & (clone[:, None] == clone[None, :]); diff = on[:, None] & on[None, :] & ~same
    nk = np.bincount(clone[on], minlength=Kc).astype(float); nT = float(on.sum())
    if kind == "clones":
        CT = np.zeros((n, n)); th = 1.0
    else:
        Dk = np.asarray(Dk, float); off = Dk[np.ix_(c0, c0)]
        if kind == "lineage":
            within = np.asarray(d_within, float)
        else:                                                                                       # clone tree: each clone's mean internal distance
            mk = np.array([np.asarray(d_within, float)[np.ix_(clone == k, clone == k)][~np.eye(int(nk[k]), dtype=bool)].mean() if nk[k] > 1 else 0.0 for k in range(Kc)])
            within = mk[c0][:, None] * np.ones(n)[None, :]
        mo = off[diff].mean() if diff.any() else 1.0; off = off * target / max(mo, 1e-12); within = np.array(within, float, copy=True)
        for k in range(Kc):                                                                         # within clone k: a compact disc's mean distance
            i = np.where(clone == k)[0]
            if len(i) > 1:
                blk = within[np.ix_(i, i)]; mw = blk[~np.eye(len(i), dtype=bool)].mean(); disc = 0.905 * np.sqrt(area * nk[k] / (np.pi * nT))
                within[np.ix_(i, i)] = blk * disc / max(mw, 1e-12)
        CT = np.where(same, within, np.where(diff, off, 0.0)); np.fill_diagonal(CT, 0.0); th = THETA
    W1 = np.where(same, th / np.maximum(nk[c0][:, None], 1.0), np.where(diff, (1.0 - th) / nT, 0.0)); np.fill_diagonal(W1, 0.0)
    W2 = W1 * CT; return dict(W1=W1, W2=W2, r=(W2 * CT).sum(1)), CT


def tree_full(kind, clone, Kc, d_cells, Dk, target):
    """the unmasked GW term over all traced pairs (weight 1 / n_T per pair) with the clone tree's or the lineage's distances, rescaled so their
    mean over traced pairs equals `target`"""
    n = len(clone); on = clone >= 0; c0 = np.maximum(clone, 0); same = on[:, None] & on[None, :] & (clone[:, None] == clone[None, :]); nT = float(on.sum())
    off = np.asarray(Dk, float)[np.ix_(c0, c0)]
    if kind == "lineage":
        CT = np.where(same, np.asarray(d_cells, float), off)
    else:
        nk = np.bincount(clone[on], minlength=Kc); mk = np.array([np.asarray(d_cells, float)[np.ix_(clone == k, clone == k)][~np.eye(int(nk[k]), dtype=bool)].mean() if nk[k] > 1 else 0.0 for k in range(Kc)])
        CT = np.where(same, mk[c0][:, None] * np.ones(n)[None, :], off)
    pair = on[:, None] & on[None, :]; CT = np.where(pair, CT, 0.0); np.fill_diagonal(CT, 0.0); m = CT[pair & ~np.eye(n, dtype=bool)].mean(); CT = CT * target / max(m, 1e-12)
    W1 = np.where(pair, 1.0 / nT, 0.0); np.fill_diagonal(W1, 0.0); W2 = W1 * CT; return dict(W1=W1, W2=W2, r=(W2 * CT).sum(1))


def tree_ct(kind, clone, Kc, d_cells, Dk, target):
    """C^T between traced cells for the clone tree (each clone's mean internal distance within, clone-to-clone distances between) or the
    lineage tree (the cells' distances within, the clone-to-clone distances between), rescaled to mean `target` over traced pairs"""
    return tree_full(kind, clone, Kc, d_cells, Dk, target)["W2"] * float((clone >= 0).sum())


def sd_dc(A, rows):
    """standard deviation of the doubly centred matrix over the given rows (the part of a cost the marginals do not absorb)"""
    B = A[rows]; B = B - B.mean(1, keepdims=True) - B.mean(0, keepdims=True) + B.mean(); return float(B.std())


def spreads(P, xy):
    """each clone's RMS distance of its mapped mass from its centre (um or section units)"""
    P = np.asarray(P, float); m = P.sum(0); c = (P.T @ xy) / np.maximum(m[:, None], 1e-300)
    return [float(np.sqrt((P[:, k] * ((xy - c[k]) ** 2).sum(1)).sum() / max(m[k], 1e-300))) for k in range(P.shape[1])]


def problem(src, name):
    """the inputs of 324_'s multi-kind fit, the clones, the trees and the scoring for one problem"""
    from scipy.spatial.distance import pdist, squareform
    m321, f289 = mod("m321", "321_c2l_profiles.py"), mod("f289", "289_field_metrics.py")
    if src in ("sim", "petracer"):
        if src == "sim":
            w, seed = name; m268, m239, m189 = mod("m268", "268_lineage_cv.py"), mod("m239", "239_deconv_fgw.py"), mod("m189", "189_me_sim.py")
            NB, X, Xc, t_f, _ = m268.build(seed, **mod("m270", "270_lineage_cv_concordance.py").WORLDS[w]); Y = X["Sc"]; d = m239.tree_dists(X, NB, t_f, X["q"])[1]["lineage"]; Nb = m239.window(X)
        else:
            P = mod("m276", "276_petracer.py").build(name, 0); X, Xc, Y = P["X"], P["Xc"], P["Y"]; d = np.asarray(P["d_raw"], float)
        Xc, Y = np.asarray(Xc, float), np.asarray(Y, float); tum = X["kind"] == 0; tr = tum & X["traced"]; K = X["K"]; cl = X["cl"][tr]; clone = np.where(cl < K - 1, cl, -1); Kc = K - 1
        me = np.asarray(X["me_type"]); types = sorted(set(me[~tum].tolist())); Gme = np.stack([Xc[(~tum) & (me == e)].mean(0) for e in types])
        gu = (Xc[tum & ~tr].mean(0) if (tum & ~tr).any() else Xc[tr].mean(0))[None, :]; fT = np.array([tr.sum() / max(tum.sum(), 1)])
        prof = m321.profiles(Xc[tr], clone)["within"]; kc = np.zeros(int(tr.sum()), int); xy = np.asarray(X["spots"], float); D = np.asarray(X["D"], float); Ncl = X["Ncl"]; onehot = np.eye(K)[cl]
        Dk = np.array([[d[np.ix_(clone == a_, clone == b_)].mean() if a_ != b_ else 0.0 for b_ in range(Kc)] for a_ in range(Kc)])
        def score(Pm):
            r = f289.scores(Pm, Ncl)
            if src == "sim":
                r.update({k: v for k, v in m239.readout(None, X, Nb, m189, P_clone=Pm).items() if k in ("clone_tv", "clone_tv_win")})
            return r
        return dict(Y=Y, Gme=Gme, gu=gu, fT=fT, Xh=prof, kc=kc, D=D, xy=xy, clone=clone, Kc=Kc, d=d, Dk=Dk, to_mass=lambda phi: phi.T @ onehot, score=score, trees=["clones", "clone_tree", "lineage"], types=[str(t) for t in types] + ["tumour"], Xt=Xc[tr])
    z = dict(np.load(DATA / ("embryo/embryo2a_e95.npz" if src == "embryo" else f"htan/{name}.npz"), allow_pickle=True))
    Xc, Y = z["Xc_fit"].astype(float), z["Ys_fit"].astype(float); ct = np.asarray(z["cell_type"]).astype(str); u, cnt = np.unique(ct, return_counts=True); ct = np.where(np.isin(ct, u[cnt < MIN_TYPE]), "other", ct)
    inform = np.asarray(z["clone_informative"], bool); clone_all = z["clone"].astype(int); traced = inform[clone_all]; remap = -np.ones(len(inform), int); remap[inform] = np.arange(inform.sum()); Kc = int(inform.sum())
    L = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1.0) * 1e4); gsel = np.argsort(-L.var(0))[: min(G_MAX, Xc.shape[1])]; Xg, Yg = Xc[:, gsel], Y[:, gsel]
    kinds = sorted(set(ct.tolist())); tk = [k for k in kinds if (traced & (ct == k)).any()]; ek = [k for k in kinds if k not in tk]
    Gme = np.stack([Xg[ct == k].mean(0) for k in ek]) if ek else np.zeros((0, Xg.shape[1]))
    gu = np.stack([Xg[(ct == k) & ~traced].mean(0) if ((ct == k) & ~traced).any() else Xg[(ct == k) & traced].mean(0) for k in tk]); fT = np.array([((ct == k) & traced).sum() / (ct == k).sum() for k in tk])
    ti = np.where(traced)[0]; kc = np.array([tk.index(k) for k in ct[ti]]); clone = remap[clone_all[ti]]
    prof = m321.profiles(Xg[ti], clone)["within"]; xy = np.asarray(z["xy"], float); D = squareform(pdist(xy)); onehot = np.eye(Kc)[clone]
    out = dict(Y=Yg, Gme=Gme, gu=gu, fT=fT, Xh=prof, kc=kc, D=D, xy=xy, clone=clone, Kc=Kc, to_mass=lambda phi: phi.T @ onehot, types=list(ek) + list(tk), Xt=Xg[ti])
    if src == "embryo":
        nj = dict(np.load(DATA / "embryo" / "embryo2a_nj.npz", allow_pickle=True)); n = len(ti); dw = np.zeros((n, n))
        for kk, k in enumerate(np.where(inform)[0]):                                              # within-clone patristic distances from the NJ blocks
            S_ = nj[f"sigma_{k}"].astype(float); dg = np.diag(S_); i = np.where(clone == kk)[0]; dw[np.ix_(i, i)] = dg[:, None] + dg[None, :] - 2.0 * S_
        Dk = np.asarray(nj["clone_dist"], float)[np.ix_(inform, inform)]; ann = z["annotation"].astype(str); lab = z["cell_type"].astype(str)[ti]
        tis = sorted(set(ann.tolist())); At = np.stack([(ann == t).astype(float) for t in tis], 1)
        Qk = np.stack([np.array([(lab[clone == k] == t).mean() for t in tis]) for k in range(Kc)]); wk = np.bincount(clone, minlength=Kc) / len(clone)
        def score(Pm):
            Pm = np.asarray(Pm, float); Pk = (Pm.T @ At); Pk = Pk / np.maximum(Pk.sum(1, keepdims=True), 1e-300)
            return dict(clone_tissue_agreement=float((wk * (1.0 - 0.5 * np.abs(Pk - Qk).sum(1))).sum()))
        out.update(d=dw, Dk=Dk, score=score, trees=["clones", "clone_tree", "lineage"], ann=ann, lab=lab)
    else:
        m335 = mod("m335", "335_htan_c2l.py"); out.update(d=None, Dk=None, score=lambda Pm: m335.ref_scores(name, Pm), trees=["clones"])
    return out


def nested_levels(d, clone, Kc, L=LEVELS, mmin=MIN_GROUP):
    """per traced cell, its group at levels 1..L of its clone's subtree (average linkage on the tree distances, at most 2^l groups,
    groups below mmin merged into the nearest sibling within the same parent); -1 outside the major clones; groups numbered globally"""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    n = len(clone); lev = -np.ones((n, L), int); gid = 0
    for k in range(Kc):
        i = np.where(clone == k)[0]
        if len(i) < 2:
            continue
        dk = np.asarray(d, float)[np.ix_(i, i)]; Z = linkage(squareform(dk, checks=False), method="average"); par = np.zeros(len(i), int)
        for l in range(L):
            lab = fcluster(Z, t=2 ** (l + 1), criterion="maxclust"); new = np.zeros(len(i), int)
            for pg in np.unique(par):
                sel = np.where(par == pg)[0]; ls_ = lab[sel]; us, cs = np.unique(ls_, return_counts=True); big = us[cs >= mmin]
                if len(big) == 0:
                    big = us[[np.argmax(cs)]]
                for u in us:
                    if u not in big:                                                              # merge into the nearest big sibling
                        a = sel[ls_ == u]; ls_[ls_ == u] = big[np.argmin([dk[np.ix_(a, sel[ls_ == b_])].mean() for b_ in big])]
                for u in np.unique(ls_):
                    new[sel[ls_ == u]] = gid; gid += 1
            lev[i, l] = new; par = new
    return lev


def subclade_errors(phi, lev, xy_true, spots):
    """per-spot error (TV) at each level, the map's group composition per spot against that of the cells' own true spots"""
    from scipy.spatial import cKDTree
    s_true = cKDTree(spots).query(xy_true)[1]; out = {}
    for l in range(lev.shape[1]):
        ok = lev[:, l] >= 0; g = np.unique(lev[ok, l], return_inverse=True)[1]; H = np.eye(g.max() + 1)[g]
        T = np.zeros((len(spots), H.shape[1])); np.add.at(T, s_true[ok], H); Pm = np.asarray(phi, float)[ok].T @ H
        has = T.sum(1) > 0; Tn = T[has] / T[has].sum(1, keepdims=True); Pn = Pm[has] / np.maximum(Pm[has].sum(1, keepdims=True), 1e-300)
        out[f"subclade_tv_l{l + 1}"] = float(0.5 * np.abs(Tn - Pn).sum(1).mean())
    return out


def abund(th):
    """each spot's abundances, untraced types then traced kinds (cell2location's w), float32"""
    z = np.asarray(th["z"], float); e = np.exp(z - z.max(1, keepdims=True)); return (np.exp(np.asarray(th["logN"], float))[:, None] * e / e.sum(1, keepdims=True)).astype(np.float32)


def truth_xy(src, name):
    """the traced dissociated cells' true positions (simulations, PEtracer's half designs), else None"""
    if src == "sim":
        w, seed = name; X = mod("m268", "268_lineage_cv.py").build(seed, **mod("m270", "270_lineage_cv_concordance.py").WORLDS[w])[1]
    elif src == "petracer" and name.startswith("half"):
        X = mod("m276", "276_petracer.py").build(name, 0)["X"]
    else:
        return None
    tr = (X["kind"] == 0) & X["traced"]; return np.asarray(X["xy_true"], float)[tr]


def run(src, name, f):
    import warnings; warnings.filterwarnings("ignore")
    t0 = time.time(); p = problem(src, name); out = json.load(open(f)) if f.exists() else dict(meta=dict(src=src, problem=str(name)), arms={}, masses={})
    args = (p["Y"], p["Gme"], p["gu"], p["fT"], p["Xh"], p["kc"], p["D"], p["xy"])
    print(f"  {name}: no tree", flush=True); r0 = fit(*args); P0 = p["to_mass"](r0["phi"]); s = p["score"](P0); np.save(f.with_name(f.stem + "_notree_abund.npy"), abund(r0["th"])); out["meta"]["types"] = p["types"]
    out["arms"]["notree"] = dict(scores=s, spreads=spreads(P0, p["xy"]), trace=r0["trace"]); out["masses"]["notree"] = P0.tolist()
    print(f"  {name} notree: {s}  [{time.time() - t0:.0f}s]", flush=True)
    q = r0["phi"].sum(0); q = q / q.sum(); target = float(q @ p["D"] @ q)
    lev = nested_levels(p["d"], p["clone"], p["Kc"]) if p.get("d") is not None else None; xyt = truth_xy(src, name)
    def sub(phi):
        return subclade_errors(phi, lev, xyt, p["xy"]) if (lev is not None and xyt is not None) else {}
    out["arms"]["notree"]["scores"].update(sub(r0["phi"])); np.save(f.with_name(f.stem + "_notree_phi.npy"), r0["phi"].astype(np.float16))
    for arm in p["trees"]:
        CT = None if arm == "clones" else tree_ct(arm, p["clone"], p["Kc"], p["d"], p["Dk"], target)
        r = fit(*args, clone=p["clone"], CT=CT, init=(r0["th"], r0["phi"]), mode=arm, lev=lev); P = p["to_mass"](r["phi"]); s = p["score"](P); s.update(sub(r["phi"])); np.save(f.with_name(f.stem + f"_{arm}_abund.npy"), abund(r["th"]))
        out["arms"][arm] = dict(scores=s, weights=r["weights"], spreads=spreads(P, p["xy"]), trace=r["trace"]); out["masses"][arm] = P.tolist()
        if True:
            np.save(f.with_name(f.stem + f"_{arm}_phi.npy"), r["phi"].astype(np.float16))
        json.dump(out, open(f, "w"), default=float); wt = r["weights"]
        print(f"  {name} {arm}: {s}" + (f", a0 {wt['a0']:.3g}, v0 {wt['v0']:.3g}" if "v0" in wt else "") + f"  [{time.time() - t0:.0f}s]", flush=True)
    if src in ("embryo", "htan"):
        np.save(f.with_name(f.stem + "_notree_phi.npy"), r0["phi"].astype(np.float16))
    out["done"] = True; json.dump(out, open(f, "w"), default=float)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); which = sys.argv[1:] or ["sim", "embryo", "petracer", "htan"]; t0 = time.time()
    jobs = {"sim": [("sim", ws, f"{ws[0]}_{ws[1]}") for ws in SIMS], "embryo": [("embryo", "e95", "embryo_e95")],
            "petracer": [("petracer", p_, "petracer_" + p_.replace(":", "_").replace(">", "_to_")) for p_ in PET], "htan": [("htan", "HT260C1-Th1K1Fc2U1Z1Bs1", "htan_HT260C1")]}
    for w in which:
        for src, name, stem in jobs[w]:
            f = OUT / f"{stem}.json"
            if f.exists() and json.load(open(f)).get("done"):
                continue
            print(f"{stem}: start", flush=True); run(src, name, f); print(f"{stem}: done ({time.time() - t0:.0f}s)", flush=True)
