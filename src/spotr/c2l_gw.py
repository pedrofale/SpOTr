"""SpOTr's model: cell2location's spot likelihood with a map of the traced dissociated cells, and the clade-aware Gromov-Wasserstein
term with learned spreads (clone labels, clone tree, or the lineage tree as a nested decomposition of each clone's scatter).

Moved verbatim from SpOTr/experiments/night/354_c2l_cgw_anova.py and 321_c2l_profiles.py (profiles) (2026-10-30), the method of record (vault: exp-unmatched-oct12 sections
123-124, decision-clade-aware-gw-learned-weights); the only additions are the `fixed_s2` ablation and `verbose` in `fit`.

One phase (decision-single-phase-fit, 2026-10-31): every arm -- no tree, clone labels, clone tree, lineage -- is fitted from the same start (the
capacities' uniform map, ADAM0 steps on cell2location's parameters, the spreads read off that map) and alternates map steps, cell2location's
parameters and the spreads for N_OUTER = 16 rounds; no arm starts from another's fit (354_ ran 8 rounds without the tree, then 8 with it).
"""
import numpy as np

N_OUTER, J, IT0, IT1, ADAM0, ADAM1 = 16, 12, 300, 100, 1000, 200
ETAS = [1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0]
LEVELS, MIN_GROUP = 3, 10
SINK_TOL, SINK_MAX = 1e-4, 5000                                                                  # rows of the map to 1e-4 (section 149)


def fit(Y, Gme, gu, fT, Xh, kc, D, xy, clone=None, CT=None, init=None, mode="clones", lev=None, fixed_s2=None, verbose=True, spread="expected", jensen="off", totals="fixed", cross="pairs", tree_scale=1.0, spread_prior=None, diag=False, map_temp=1.0, structure=None, alpha_tree=0.0, tree_d=None, cell_cost=None, X_cells=None, beta=0.0, untraced="free", clone_tau=1.0, cell_metric="cosine", tree_target="linear", tree_n=None, omega=None, capacity="share"):
    """324_'s multi-kind cell2location fit with the clade-aware GW term whose weights are learned (see the docstring); clone=None: no tree;
    CT: the tree's distances between traced cells (None: clone labels, the within-clade term only); init = (theta, phi) warm start;
    mode: "clones" | "clone_tree" | "lineage"; lev: the nested groups of each traced cell (nested_levels), for mode "lineage";
    fixed_s2: per-clade spreads to use instead of the learned ones in the clone compaction (an ablation; None: learned);
    spread: what the spreads (and the clone tree's residual) are read from -- "expected": the expected distances between the cells under the map,
    which add each cell's own map uncertainty (E|x_i - x_j|^2 = |E x_i - E x_j|^2 + tr Var x_i + tr Var x_j); "positions": the distances between
    the cells' expected positions, without it (exp-unmatched-oct12 section 130);
    jensen: the expected log-likelihood under the map, to second order -- "off": the likelihood of the map's average traced profile (the plug-in,
    which for a concave likelihood exceeds the average likelihood by a gap that grows as the map mixes unlike cells into a spot); "full": minus
    (1/2) sum_sg lambda_sg V_sg, with V_sg = sum_n phi_ns (1 - phi_ns) (x_ng - hbar_sg)^2 / a_s^2 the variance of spot s's mean traced profile under
    the map and lambda_sg = I_sg (m_g y_s w_sc f_c)^2 (I the negative binomial's Fisher information), in the map's step (lambda held fixed within a
    step) and in cell2location's; "map": in the map's step only (exp-unmatched-oct12 section 131);
    totals: "fixed" (default, the method of record): every traced cell's row sums to 1, so each clone's total is its share of the dissociated cells;
    "free" (an option, exp-unmatched-oct12 section 150): only each spot's traced capacity is kept (semi-relaxed transport) -- in every spot the
    capacity goes to the cells in proportion to exp(-cost), so a clone's share there is its number of cells times how well it explains the spot, and
    each clone's total follows the spots, with the dissociated sample's composition as the prior weight; for subclone-level profiles (cells of a
    clone alike), where only the clones' totals can move; the spreads are read from the row-normalised map;
    spread "switch_guard" (an option, section 169): as "switch", but a clade's spreads are read from positions only while its cells have
    committed -- their average own variance under the map below half of that plus the variance of their expected positions; otherwise from
    expected distances (W3's collapse, sections 164 and 166, came from switching clades whose cells never commit);
    spread "switch" (an option, exp-unmatched-oct12 section 163): "expected" for the first half of the outer rounds, while the map forms (at the
    uniform start every cell's expected position is the section's centre, which "positions" reads as a collapsed clone, section 130), then
    "positions", so the spreads stop counting each cell's map uncertainty as physical spread (sections 161-162);
    spread_prior: None (default: the spreads' InvGamma(1, b0) prior, nearly uninformative) or k > 0, a prior that the lineage form's finest tree
    groups are compact (section 163): the finest level's spread gets an inverse-gamma prior whose mean is the spread of compact discs -- each
    finest group's cells packed at the density of traced cells in the tumour (rho = traced cells / (spot area x the effective number of spots
    holding traced capacity); a disc of n cells has per-dimension variance n / (4 pi rho)) -- with k times the data's degrees of freedom as
    pseudo-observations (k = 1: as much weight as the map);
    cell_cost: None, "raw" or "resid" -- FGW's per-nucleus expression cost added to the map's step (exp-unmatched-oct12 section 172):
    M_ns = the distance between log1p(CPM(X_n)) and log1p(CPM(Y_s)) for the traced cells' own counts X_cells (n x genes, the genes of Y),
    cell_metric "cosine" (the method from 2026-11-02, Pedro: "let's use cosine" -- scDOT's starting cost and Tangram's similarity are cosine,
    and it ignores the depth of nuclei and spots; decision-cosine-expression-cost) or "sqeuclidean" (the squared Euclidean distance, every run
    before 2026-11-02); "raw" the spot's
    whole counts, "resid" the spot's counts minus the likelihood's current microenvironment + untraced + background (its tumour part,
    recomputed each round); scaled once at the start map so its double-centred spread equals the likelihood cost's, times beta / (1 - beta);
    structure: None (the method: weights from learned spreads) or a fixed structure for the GW term, with one strength alpha_tree in [0, 1)
    (exp-unmatched-oct12 section 171): "dm" -- distance matching, every pair of traced cells (clone >= 0) with unit weight and target
    a d_ij (tree_d: the cells' tree distances; a = mean spot distance / mean tree distance); "compact" -- every pair within a clade, target 0;
    the unit weights scaled once, at the start map, so that the tree cost's double-centred spread equals the likelihood cost's, then times
    alpha_tree / (1 - alpha_tree); fixed for the whole fit;
    capacity: "count" (default) -- each spot receives the traced cells of a kind in proportion to the deconvolution's number of cells of that
    kind in the spot, w_mc; "share" (an option, 2026-11-02, exp-unmatched-oct12 section 181) -- in proportion to the kind's share of the spot's
    cells, w_mc / N_m, as if every spot held the same number of cells (the deconvolution's absolute cell counts confound cells with detection;
    on HT260C1 they put half as many cancer cells per spot in B's territory as in A's, so B2's cells overflowed into the border); the cancer
    part of the spot's expression keeps the deconvolution's amount w_mc either way;
    tree_target (structure "dm"; an option, 2026-11-02, Pedro: "We should say that that's an option"): "linear" (default) -- the target
    separation of two cells is a times their tree distance (a = mean spot distance / mean tree distance); "growth" -- it is the mean
    separation of two cells of the clade founded at their most recent common ancestor, packed at the density of traced cells in the tissue,
    sqrt(n_nn' / (4 rho)) (n_nn' that clade's number of traced cells: tree_n, or computed from an ultrametric tree_d by mrca_sizes; rho =
    traced cells / (spot pitch^2 x the effective number of spots holding traced capacity)): no free parameter, and the targets stay on the
    tissue's scale where the linear target keeps growing with tree time (the growth law of exp-unmatched-oct12 sections 2-10, with no mixing
    length);
    omega (an option, 2026-11-02, Pedro: "we can probably show that they can be tuned to help if the user knows that some clades behave
    differently"): per-clade weights for the structural term, a dict {clone: weight} or an array over clones (default 1 for every clone):
    Omega_nn' = omega_{k_n} omega_{k_n'} times the structure's own pair weights, applied after the strength is calibrated, so that the other
    clades keep their weight -- 0 maps a clade by expression alone (all of its pairs dropped), below 1 holds it more loosely (a clade known to
    form a stripe or an invasive front), above 1 more tightly; for structured deviations that expression can place, not for scattered
    migrants, whose clade's centre stays the best guess (exp-unmatched-oct12 sections 17, 19); weights are set by the user, not learned
    (sections 10, 18);
    map_temp: the weight of the likelihood in the map's step only (cost and line search; cell2location's own fit untouched) -- default 1, the
    method; below 1 the map follows only part of the likelihood's evidence about where each cell goes (exp-unmatched-oct12 section 168: on
    HTAN most of that evidence does not reproduce across halves of the genes);
    diag: record, at the last round, the size of each term in the map's cost for a typical cell (section 165): the range of the likelihood's
    cost (minus its gradient in the map, nats per cell moved) and of the tree term's cost over the cell's 20 cheapest spots by likelihood and
    over all spots (5-95%), medians over cells, and the cells' median effective number of spots;
    totals "clone" (an option, exp-unmatched-oct12 section 179): each clone's total over the section is learned, every cell of a clone keeping the
    same mass (each nucleus placed whole; only the Kc totals move): the clone shares rho start at the dissociated sample's (pi_k = n_k / n),
    and after each round move (in proportion space, by the step of 0, 0.05, ..., 1 that lowers the likelihood + entropy + prior most) towards the
    clone totals of a clones x spots transport: the likelihood's linearised cost averaged over each clone's cells (all of a clone's cells grow
    together; a log-sum-exp over cells let each spot's best-fitting nucleus decide, and A's 1,969 won every spot), each spot's capacity met exactly, each clone's total
    held to pi_k n by a KL penalty of weight clone_tau (unbalanced Sinkhorn; entropy weight 1 as the map's); clone_tau -> infinity is
    "fixed"; clone_tau = 1 weighs the dissociated composition like a multinomial sample of the section's tumour, to second order; the tree
    and the expression cost are left out of the totals (their per-cell levels are arbitrary offsets, and the tree term grows with a clone's
    mass squared -- neither matters while the totals are fixed); they act on where each clone's cells go
    (a first version from the clones' mean row potentials collapsed to one clone in a round, section 179);
    each cell of clone k then has row sum rho_k / pi_k; one kind of traced cells only;
    untraced: "tied" (the earlier default) -- each tumour kind's abundance w_s split by its fixed traced fraction fT into untraced, w_s (1 - fT) gu,
    and traced, w_s fT hbar_s, so every spot's tumour is (1 - fT) untraced by construction; "free" (exp-unmatched-oct12 section 174) -- the untraced
    tumour is a type of its own with a free abundance per spot, as in cell2location (gu appended to Gme, fT set to 1), so each spot's traced
    capacity is its own abundance; abund() then returns the microenvironment types, the untraced types, the traced kinds;
    tree_scale: a diagnostic multiplier on the tree term's learned weights (default 1: the method; exp-unmatched-oct12 section 162 -- is the
    tree used too weakly?), not a tuning knob of the method;
    cross: the clone tree's and the lineage's term between clades -- "pairs" (default): every pair of cells in different clades against the
    tree's distance (scale a0 by least squares, weight v0 from the residuals); "blocks" (an option, tested and not adopted, section 154): only the
    clades' mean distances against the tree's, up to a scale refitted inside the term.  Either way the term is off when the data have exactly two
    subclones (Pedro, 2026-11-01: "shouldnt the only time this applies be when we literally only have 2 subclones?"): a tree of two leaves has no
    relative distances to give, and the per-pair term then only penalises the spread of the distances between the two, pulling a compact clade
    into the middle of a spread one (HT268B1 K3 at 1% of the counts: purity 0.54 against the clone labels' 0.97); the clone tree is then the clone
    labels.  Subclones are counted as the tree's leaves: the clades, or for the lineage form the groups of cells at tree distance 0 from one
    another (HT260C1's lineage form, A and B with B1 and B2 nested in B, has three; a cell-level tree has one per cell)"""
    import jax, jax.numpy as jnp, optax
    from functools import partial
    from jax.scipy.special import logsumexp, gammaln
    assert untraced in ("tied", "free") and capacity in ("count", "share")
    if untraced == "free":                                                                          # the untraced tumour as a type of its own with
        Gme = np.vstack([np.asarray(Gme, float).reshape(-1, Y.shape[1]), np.asarray(gu, float)])    # a free abundance per spot (cell2location's);
        fT = np.ones_like(np.asarray(fT, float))                                                    # each kind's abundance is then all traced
    S, G = Y.shape; E = Gme.shape[0]; T = gu.shape[0]; n = Xh.shape[0]; rows = [np.where(kc == c)[0] for c in range(T)]; nc = np.array([len(r) for r in rows], float)
    f32 = lambda a: jnp.asarray(np.asarray(a, np.float32))
    Yj, Gj, guj, fTj, Xj, kcj = f32(Y), f32(Gme if E else np.zeros((1, G))), f32(gu), f32(fT), f32(Xh), jnp.asarray(kc); mc = float(Y.mean())
    xc = np.asarray(xy, float) - np.asarray(xy, float).mean(0); Lj, sqj, Dj = f32(xc), f32((xc ** 2).sum(1)), f32(D); b0 = 1e-3 * float(np.median(D[D > 0])) ** 2
    th = init[0] if init is not None else dict(logN=f32(np.full(S, np.log(10.0))), z=f32(np.zeros((S, E + T))), logm=f32(np.zeros(G)), logy=f32(np.zeros(S)),
                                               logs=f32(np.full(G, np.log(1e-3 * mc))), logr=f32(np.full(G, np.log(10.0))))
    def parts(th):
        w = jnp.exp(th["logN"])[:, None] * jax.nn.softmax(th["z"], axis=1); return w[:, :E], w[:, E:]
    def caps_of(th):                                                                               # each kind's capacity per spot, floored at a
        wt = parts(th)[1]                                                                           # millionth of its mean and renormalised (a kind
        if capacity == "share":                                                                     # the kind's share of the spot's cells, at a common
            wt = wt / jnp.exp(th["logN"])[:, None]                                                  # number of cells per spot
        c = (wt / wt.sum(0, keepdims=True) * f32(nc)[None, :]).T
        c = jnp.maximum(c, 1e-6 * f32(nc)[:, None] / S); return c * (f32(nc) / c.sum(1))[:, None]     # with ~0 capacity at most spots overflows Sinkhorn)
    def mu_of(th, phi, caps):
        we, wt = parts(th); A = (wt * fTj[None, :] / jnp.maximum(caps.T, 1e-12))[:, kcj].T
        expr = (we @ Gj if E else 0.0) + (wt * (1.0 - fTj)[None, :]) @ guj + (phi * A).T @ Xj
        return (jnp.exp(th["logm"])[None, :] * expr + jnp.exp(th["logs"])[None, :]) * jnp.exp(th["logy"])[:, None]
    def loglik(th, phi, caps):
        mu = mu_of(th, phi, caps); r = jnp.exp(th["logr"])[None, :]
        return jnp.sum(gammaln(Yj + r) - gammaln(r) - gammaln(Yj + 1.0) + r * jnp.log(r / (r + mu)) + Yj * jnp.log(mu / (r + mu) + 1e-30))
    X2j = Xj ** 2
    def lam_of(th, phi, caps):                                                                     # lambda_csg = I_sg (m_g y_s w_sc f_c)^2, per traced kind
        mu = mu_of(th, phi, caps); r = jnp.exp(th["logr"])[None, :]; I = r / (mu * (mu + r)); wt = parts(th)[1]; sc = jnp.exp(th["logm"])[None, :] * jnp.exp(th["logy"])[:, None]
        return jnp.stack([I * (sc * (wt[:, c] * fTj[c])[:, None]) ** 2 for c in range(T)])
    def pen(phi, lam, caps):                                                                       # (1/2) sum lambda V: the Jensen gap of the plug-in, to second order
        tot = 0.0
        for c in range(T):
            P = phi[rows[c]]; Q = P * (1.0 - P); X = Xj[rows[c]]; a = jnp.maximum(caps[c], 1e-12); hb = (P.T @ X) / a[:, None]
            V = ((Q.T @ X2j[rows[c]]) - 2.0 * hb * (Q.T @ X) + hb ** 2 * Q.sum(0)[:, None]) / (a ** 2)[:, None]; tot = tot + 0.5 * jnp.sum(lam[c] * V)
        return tot
    def logprior(th):
        return (jnp.sum(9.0 * th["logN"] - jnp.exp(th["logN"])) + jnp.sum(19.0 * th["logy"] - 20.0 * jnp.exp(th["logy"])) - 0.5 * jnp.sum(th["logm"] ** 2)
                - jnp.sum((th["logs"] - np.log(1e-3 * mc)) ** 2) / 8.0 - jnp.sum((th["logr"] - np.log(10.0)) ** 2) / 8.0)
    free = totals == "free"; assert totals in ("fixed", "free", "clone"); ctot = totals == "clone"
    if ctot:
        assert clone is not None and T == 1 and (np.asarray(clone) >= 0).all(), "totals='clone': one traced kind, every traced cell in a clone"
    opt = optax.adam(0.02); st = opt.init(th)
    @partial(jax.jit, static_argnums=4)
    def adam(th, st, phi, caps, steps):
        def body(c, _):
            th, st = c; g = jax.grad(lambda t: -(loglik(t, phi, caps) + logprior(t)) + (pen(phi, lam_of(t, phi, caps), caps) if jensen == "full" else 0.0))(th)
            up, st = opt.update(g, st, th); return (optax.apply_updates(th, up), st), None
        (th, st), _ = jax.lax.scan(body, (th, st), None, length=steps); return th, st
    def sink(Cst, logcap, f, g, iters, logr):
        """log-domain Sinkhorn: at least `iters` sweeps, then on until every cell's row sums to 1 within SINK_TOL (at most SINK_MAX sweeps); the
        column (capacity) update comes last, so an unconverged solve leaves the rows off -- and with them each clone's total mass and the spreads
        read from the map (2026-10-31, exp-unmatched-oct12 section 149: HT268B1 K3's B rows summed to 1.37 after a fixed 100 sweeps)"""
        def body(_, s):
            f, g = s; f = (logr - logsumexp(g[kcj] - Cst, axis=1)) if not free else f                # rows sum to exp(logr) (1 unless totals "clone")
            g = jnp.stack([logcap[c] - logsumexp(f[rows[c], None] - Cst[rows[c]], axis=0) for c in range(T)]); return (f, g)
        def err(f, g):
            return 0.0 if free else jnp.max(jnp.abs(jnp.exp(logsumexp(f[:, None] + g[kcj] - Cst, axis=1) - logr) - 1.0))
        f, g = jax.lax.fori_loop(0, iters, body, (f, g))
        def more(s):
            f, g, k = s; f, g = jax.lax.fori_loop(0, 10, body, (f, g)); return (f, g, k + 10)
        f, g, _ = jax.lax.while_loop(lambda s: (s[2] < SINK_MAX) & (err(s[0], s[1]) > SINK_TOL), more, (f, g, iters)); return f, g
    def pD2(p):
        return (p @ sqj)[:, None] + p.sum(1)[:, None] * sqj[None, :] - 2.0 * (p @ Lj) @ Lj.T
    ent = lambda p: jnp.sum(p * jnp.log(jnp.maximum(p, 1e-30)))
    assert cross in ("blocks", "pairs")
    Kc_ = int(clone.max()) + 1 if clone is not None else 0; nk_ = np.bincount(clone[clone >= 0], minlength=Kc_).astype(float) if clone is not None else np.zeros(0)
    if clone is not None and mode == "lineage" and CT is not None:                                  # leaves: cells at tree distance 0 from one another
        from scipy.sparse.csgraph import connected_components
        tr_ = np.where(clone >= 0)[0]; Z = np.asarray(CT, float)[np.ix_(tr_, tr_)] <= 1e-9 * max(float(np.max(CT)), 1e-30)
        leaves = int(connected_components(Z, directed=False)[0])
    else:
        leaves = int((nk_ > 0).sum())
    two = leaves <= 2                                                                               # exactly two subclones: no term between them
    use_blk = clone is not None and mode in ("clone_tree", "lineage") and cross == "blocks" and int((nk_ > 0).sum()) >= 3 and not two
    Ohj = f32(np.eye(max(Kc_, 1))[np.maximum(clone, 0)] * (clone >= 0)[:, None] if clone is not None else np.zeros((n, 1)))
    NNj = f32(np.maximum(np.outer(nk_, nk_), 1.0) if clone is not None else np.ones((1, 1)))
    def blk(p, Bw, Bt):
        """sum over ordered pairs of clades of Bw_kl (a Bt_kl - Dbar_kl)^2, Dbar the clades' mean expected distance under the map, a the
        weighted least-squares scale (so only relative distances count)"""
        pn = p / jnp.maximum(p.sum(1, keepdims=True), 1e-30); rho = Ohj.T @ pn; Dm = (rho @ Dj @ rho.T) / NNj
        a = jnp.sum(Bw * Bt * Dm) / jnp.maximum(jnp.sum(Bw * Bt * Bt), 1e-30); return jnp.sum(Bw * (a * Bt - Dm) ** 2)
    @jax.jit
    def gcg(phi, f, g, th, caps, W1, W2, rr, Bw, Bt, Mb, logr):
        logcap = jnp.log(jnp.maximum(caps, 1e-30))
        def step(c, _):
            phi, f, g = c; Lt = 2.0 * rr[:, None] + 2.0 * (W1 @ pD2(phi)) - 4.0 * (W2 @ (phi @ Dj)) + Mb
            jen = jensen != "off"; lam = jax.lax.stop_gradient(lam_of(th, phi, caps)) if jen else None; pen0 = pen(phi, lam, caps) if jen else 0.0
            gb = jax.grad(blk)(phi, Bw, Bt) if use_blk else 0.0; b0_ = blk(phi, Bw, Bt) if use_blk else 0.0
            Cst = -float(map_temp) * jax.grad(lambda p: loglik(th, p, caps) - (pen(p, lam, caps) if jen else 0.0))(phi) + Lt + gb; f, g = sink(Cst, logcap, f, g, IT1, logr); dl = jnp.exp(f[:, None] + g[kcj] - Cst) - phi
            q1 = jnp.sum(dl * Lt); q2 = jnp.sum(dl * (W1 @ pD2(dl))) - 2.0 * jnp.sum(dl * (W2 @ (dl @ Dj)))
            mu0 = mu_of(th, phi, caps); dmu = mu_of(th, phi + dl, caps) - mu0; r = jnp.exp(th["logr"])[None, :]       # the spot means are affine in the map
            def dobj(e):                                                                            # objective change from not moving, term by term
                dll = jnp.sum(Yj * jnp.log1p(e * dmu / mu0) - (Yj + r) * jnp.log1p(e * dmu / (r + mu0)))
                pe = jnp.maximum(phi + e * dl, 1e-30); p0 = jnp.maximum(phi, 1e-30); dpen = (pen(phi + e * dl, lam, caps) - pen0) if jen else 0.0
                db = (blk(phi + e * dl, Bw, Bt) - b0_) if use_blk else 0.0
                return float(map_temp) * (-dll + dpen) + db + e * q1 + e * e * q2 + jnp.sum(pe * jnp.log(pe) - p0 * jnp.log(p0))
            vals = jnp.stack([dobj(e) for e in ETAS]); e = jnp.asarray(ETAS)[jnp.argmin(vals)]
            return (phi + e * dl, f, g), jnp.stack([vals.min(), e, vals[-1] - vals.min()])
        (phi, f, g), tr = jax.lax.scan(step, (phi, f, g), None, length=J); return phi, f, g, tr
    @jax.jit
    def moments(phi):                                                                               # E D and E D^2 between traced cells (cells independent)
        return phi @ Dj @ phi.T, pD2(phi) @ phi.T
    use = clone is not None; on = (clone >= 0) if use else np.zeros(n, bool); Kc = int(clone.max()) + 1 if use else 0
    same = on[:, None] & on[None, :] & ((clone[:, None] == clone[None, :]) if use else False); np.fill_diagonal(same, False); pair = on[:, None] & on[None, :]; np.fill_diagonal(pair, False)
    nk = np.bincount(clone[on], minlength=Kc).astype(float) if use else np.zeros(0); nT = float(on.sum()); c0 = np.maximum(clone, 0) if use else None
    pi_ = nk / max(nk.sum(), 1.0) if use else None; lrho = np.log(np.maximum(pi_, 1e-300)) if use else None   # totals "clone": the clone shares, from pi
    diff = pair & ~same
    def ls(c, ed, ed2):
        """least-squares scale of the tree distances to E D, and the summed residual E (a c - D)^2"""
        a = float((c * ed).sum() / max((c * c).sum(), 1e-30)); return a, float((a * a * c * c - 2.0 * a * c * ed + ed2).sum())
    pitch_ = float(np.median(np.sort(np.asarray(D, float), 1)[:, 1])) if S > 1 else 1.0               # the spots' spacing
    def weights(phi, rnd=0):
        """posterior means of the spreads at the current map (to be used in outer round rnd); W1 (on E D^2), W2 (on E D) and r over ordered
        pairs, and a record"""
        est = spread if spread not in ("switch", "switch_guard") else ("expected" if rnd < N_OUTER // 2 else "positions")
        colm = np.asarray(phi, float).sum(0); rho = float(np.asarray(phi, float).sum()) / (pitch_ ** 2 * colm.sum() ** 2 / max((colm ** 2).sum(), 1e-30))
        guard = {}
        if est == "positions":                                                                  # where the cells are, not the map's uncertainty about them
            mu = np.asarray(jnp.asarray(phi, jnp.float32) @ Lj, float); sq = (mu ** 2).sum(1); ED2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * mu @ mu.T, 0.0); ED = np.sqrt(ED2)
            if spread == "switch_guard":                                                        # positions only for clades whose cells have committed:
                pn = np.asarray(phi, float); pn = pn / np.maximum(pn.sum(1, keepdims=True), 1e-30)  # each cell's own variance under the map, against the
                EDe, ED2e = (np.asarray(x, float) for x in moments(jnp.asarray(pn, jnp.float32)))   # spread of the cells' expected positions, per clade
                wv = pn @ np.asarray(sqj, float) - (mu ** 2).sum(1); ok = np.ones(n, bool)
                for k in range(Kc):
                    i = np.where(clone == k)[0]
                    if len(i) > 1:
                        w_ = float(wv[i].mean()); b_ = float(((mu[i] - mu[i].mean(0)) ** 2).sum(1).mean()); guard[k] = w_ / max(w_ + b_, 1e-30); ok[i] = guard[k] < 0.5
                both = ok[:, None] & ok[None, :]; ED2 = np.where(both, ED2, ED2e); ED = np.where(both, ED, EDe)
        else:
            pn = np.asarray(phi, float); pn = pn / np.maximum(pn.sum(1, keepdims=True), 1e-30)        # each cell's position distribution
            ED, ED2 = (np.asarray(x, float) for x in moments(jnp.asarray(pn, jnp.float32)))
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
                if spread_prior:                                                                   # compact finest groups: discs at density rho
                    qf = keep[T_]; tp = sum(float((qf == g).sum()) ** 2 for g in np.unique(qf)) / (4.0 * np.pi * rho) / max(dof[T_], 1)
                    al = float(spread_prior) * max(dof[T_], 1); tau2[T_] = (al * tp + b0 + max(P_[T_], 0.0) / 2.0) / (al + max(dof[T_], 1))
                    rec.setdefault("tau2_prior", []).append(tp)
                blk = np.zeros((m, m))
                for t, q in enumerate(keep):
                    c = 1.0 / (2.0 * tau2[t]) - (1.0 / (2.0 * tau2[t - 1]) if t >= 1 else 0.0)
                    for g in np.unique(q):
                        sel = q == g; blk[np.ix_(sel, sel)] += c / sel.sum()
                W1[np.ix_(i, i)] = 0.5 * blk; rec.setdefault("tau2", []).append(tau2)
            else:
                W1[np.ix_(i, i)] = 1.0 / (4.0 * m * (s2[k] if fixed_s2 is None else float(fixed_s2[k])))
        rec["s2"] = s2.tolist(); rec["estimator"] = est; rec["rho"] = rho
        if guard:
            rec["uncommitted_share"] = {int(k): float(v) for k, v in guard.items()}
        Bw = np.zeros((max(Kc, 1), max(Kc, 1))); Bt = np.zeros_like(Bw)
        if mode in ("clone_tree", "lineage"):
            iu = np.triu(diff, 1); a0, R0 = ls(CT[iu], ED[iu], ED2[iu]); t20 = (b0 + R0 / (2.0 * nT)) / max(nT - 1.0, 1.0); v0 = 1.0 / (4.0 * nT * t20)
            rec.update(a0=a0, t20=t20, v0=v0, cross=cross if not two else "off (two subclones)", leaves=leaves)
            if two:
                pass
            elif cross == "pairs":
                W1 = np.where(diff, v0, W1); A = np.where(diff, a0 * CT, A)
            elif use_blk:                                                                          # the tree's distance between every two clades
                for k in range(Kc):
                    for l in range(Kc):
                        if k != l and nk[k] > 0 and nk[l] > 0:
                            Bt[k, l] = float(CT[np.ix_(clone == k, clone == l)].mean())
                Bw = v0 * np.outer(nk, nk); np.fill_diagonal(Bw, 0.0); Bw = np.where(Bt > 0, Bw, 0.0)
        np.fill_diagonal(W1, 0.0); W1 = W1 * float(tree_scale); W2 = W1 * A; return f32(W1), f32(W2), f32((W2 * A).sum(1)), f32(Bw), f32(Bt), rec
    caps = caps_of(th); f, g = jnp.zeros(n, jnp.float32), jnp.zeros((T, S), jnp.float32); logr = jnp.zeros(n, jnp.float32)
    if init is None:
        f, g = sink(jnp.zeros((n, S), jnp.float32), jnp.log(caps), f, g, IT0, logr); phi = jnp.exp(f[:, None] + g[kcj]); th, st = adam(th, st, phi, caps, ADAM0)
    else:
        phi = f32(init[1])
    zero = (f32(np.zeros((n, n))), f32(np.zeros((n, n))), f32(np.zeros(n)), f32(np.zeros((max(Kc, 1), max(Kc, 1)))), f32(np.zeros((max(Kc, 1), max(Kc, 1)))), {})
    W1, W2, rr, Bw, Bt, rec = weights(np.asarray(phi), 0) if use else zero; trace = []
    if structure is not None:                                                                       # a fixed structure, one strength
        trn = np.where(clone >= 0)[0]
        if structure == "dm":
            U = np.zeros((n, n)); U[np.ix_(trn, trn)] = 1.0; dd = np.asarray(tree_d, float); iu_ = np.triu_indices(len(trn), 1)
            assert tree_target in ("linear", "growth")
            if tree_target == "linear":
                a_ = float(np.asarray(D, float)[np.triu_indices(S, 1)].mean() / max(dd[np.ix_(trn, trn)][iu_].mean(), 1e-12)); Tg = a_ * dd
            else:                                                                                   # the clade at the common ancestor, packed
                nm_ = np.asarray(tree_n, float) if tree_n is not None else mrca_sizes(dd)
                colm_ = np.asarray(caps, float).sum(0); rho_ = float(colm_.sum()) / (pitch_ ** 2 * colm_.sum() ** 2 / max(float((colm_ ** 2).sum()), 1e-30))
                Tg = np.sqrt(nm_ / (4.0 * rho_))
        else:
            U = ((clone[:, None] == clone[None, :]) & (clone[:, None] >= 0)).astype(float); Tg = np.zeros((n, n))
        np.fill_diagonal(U, 0.0); U = U / max(U.sum(), 1e-30) * n; W2u = U * Tg; rru = (W2u * Tg).sum(1)
        Cl = -np.asarray(jax.grad(lambda p_: loglik(th, p_, caps))(phi), float)
        Ltu = np.asarray(2.0 * f32(rru)[:, None] + 2.0 * (f32(U) @ pD2(phi)) - 4.0 * (f32(W2u) @ (phi @ Dj)), float)
        dcs = lambda C: float((C - C.mean(1, keepdims=True) - C.mean(0, keepdims=True) + C.mean()).std())
        cal = dcs(Cl[trn]) / max(dcs(Ltu[trn]), 1e-30); lam = cal * float(alpha_tree) / max(1.0 - float(alpha_tree), 1e-9)
        if omega is not None:                                                                       # per-clade weights, after the calibration
            om_ = np.ones(n)
            for i_ in trn:
                om_[i_] = float(omega.get(int(clone[i_]), 1.0)) if isinstance(omega, dict) else float(np.asarray(omega, float)[int(clone[i_])])
            U = U * np.outer(om_, om_); W2u = U * Tg; rru = (W2u * Tg).sum(1)
        W1, W2, rr = f32(lam * U), f32(lam * W2u), f32(lam * rru); Bw = f32(np.zeros((max(Kc, 1), max(Kc, 1)))); Bt = Bw
        rec = dict(structure=structure, alpha_tree=float(alpha_tree), calibration=cal, strength=lam, tree_target=tree_target,
                   omega=(None if omega is None else (dict(omega) if isinstance(omega, dict) else list(np.asarray(omega, float)))))
    Mb = f32(np.zeros((n, S)))
    if cell_cost is not None and beta > 0:                                                          # FGW's per-nucleus expression cost
        Xn_ = np.asarray(X_cells, float); Lc = f32(np.log1p(Xn_ / np.maximum(Xn_.sum(1, keepdims=True), 1.0) * 1e4)); Lc2 = jnp.sum(Lc ** 2, 1)
        assert cell_metric in ("cosine", "sqeuclidean")
        Lcn = Lc / jnp.maximum(jnp.sqrt(Lc2), 1e-12)[:, None]                                       # unit rows, for the cosine distance
        def M_of(t):
            if cell_cost == "resid":
                we_, wt_ = parts(t); rest = (jnp.exp(t["logm"])[None, :] * ((we_ @ Gj if E else 0.0) + (wt_ * (1.0 - fTj)[None, :]) @ guj) + jnp.exp(t["logs"])[None, :]) * jnp.exp(t["logy"])[:, None]
                Ys_ = jnp.maximum(Yj - rest, 0.0)
            else:
                Ys_ = Yj
            Ls = jnp.log1p(Ys_ / jnp.maximum(Ys_.sum(1, keepdims=True), 1.0) * 1e4)
            if cell_metric == "cosine":
                return 1.0 - Lcn @ (Ls / jnp.maximum(jnp.linalg.norm(Ls, axis=1, keepdims=True), 1e-12)).T
            return Lc2[:, None] + jnp.sum(Ls ** 2, 1)[None, :] - 2.0 * Lc @ Ls.T
        M0 = np.asarray(M_of(th), float); Cl0 = -np.asarray(jax.grad(lambda p_: loglik(th, p_, caps))(phi), float)
        dcs_ = lambda C: float((C - C.mean(1, keepdims=True) - C.mean(0, keepdims=True) + C.mean()).std())
        beff = dcs_(Cl0) / max(dcs_(M0), 1e-30) * float(beta) / max(1.0 - float(beta), 1e-9); Mb = f32(beff * M0)
        if isinstance(rec, dict):
            rec = dict(rec, cell_cost=cell_cost, cell_metric=cell_metric, beta=float(beta), cell_cost_scale=beff)
    for it in range(N_OUTER):
        caps = caps_of(th); lphi = jnp.log(jnp.maximum(phi, 1e-30)); f, g = sink(-lphi, jnp.log(jnp.maximum(caps, 1e-30)), f, g, IT0, logr); phi = jnp.exp(f[:, None] + g[kcj] + lphi)
        if cell_cost == "resid" and beta > 0:
            Mb = f32(beff * np.asarray(M_of(th), float))
        phi, f, g, tr = gcg(phi, f, g, th, caps, W1, W2, rr, Bw, Bt, Mb, logr); th, st = adam(th, st, phi, caps, ADAM1)
        if ctot:                                                                                    # the clone shares: half-way to the clones x spots solve
            Cc = np.asarray(-float(map_temp) * jax.grad(lambda p_: loglik(th, p_, caps))(phi), float)   # the likelihood alone: totals answer to the counts
            lZ = np.stack([np.log(nk[k]) - Cc[c0 == k].mean(0) for k in range(Kc)]); lc = np.log(np.maximum(np.asarray(caps, float)[0], 1e-30))   # a clone grows as one
            Fk, gs = np.zeros(Kc), np.zeros(S); kt = float(clone_tau) / (float(clone_tau) + 1.0); lpn = np.log(pi_ * n)
            for _ in range(500):                                                                    # unbalanced Sinkhorn: clone totals KL(tau) to pi n, spots exact
                Fk = kt * (lpn - np.logaddexp.reduce(gs[None, :] + lZ, axis=1)); gs = lc - np.logaddexp.reduce(Fk[:, None] + lZ, axis=0)
            tg_ = np.logaddexp.reduce(Fk[:, None] + gs[None, :] + lZ, axis=1); tg_ -= np.logaddexp.reduce(tg_)
            lph_ = jnp.log(jnp.maximum(phi, 1e-30)); lcap_ = jnp.log(jnp.maximum(caps, 1e-30)); best_ = None
            for e_ in (0.0, 0.05, 0.15, 0.35, 0.7, 1.0):                                            # line search on the true objective (the linearised
                lr_ = np.log(np.maximum((1.0 - e_) * np.exp(lrho) + e_ * np.exp(tg_), 1e-12)); lr_ = lr_ - np.logaddexp.reduce(lr_); lg_ = f32((lr_ - np.log(pi_))[c0])
                f_, g_ = sink(-lph_, lcap_, f, g, IT0, lg_); pe_ = jnp.exp(f_[:, None] + g_[kcj] + lph_)
                ob_ = float(-float(map_temp) * loglik(th, pe_, caps) + jnp.sum(pe_ * jnp.log(jnp.maximum(pe_, 1e-30)))) \
                      + float(clone_tau) * n * float(np.sum(np.exp(lr_) * (lr_ - np.log(pi_))))
                if verbose:
                    print(f"        rho step {e_}: shares {np.round(np.exp(lr_), 3).tolist()}, objective {ob_:.6g}", flush=True)
                if best_ is None or ob_ < best_[0]:
                    best_ = (ob_, lr_, e_)
            lrho = best_[1]; logr = f32((lrho - np.log(pi_))[c0]); rho_step = best_[2]
        if use and structure is None:
            W1, W2, rr, Bw, Bt, rec = weights(np.asarray(phi), it + 1)
        tr = np.asarray(tr, float); pz = float(pen(phi, lam_of(th, phi, caps), caps)) if jensen != "off" else None
        if diag and it == N_OUTER - 1:                                                             # how big is each term in the map's cost
            Cl = -np.asarray(jax.grad(lambda p_: loglik(th, p_, caps))(phi), float)
            Lt_ = np.asarray(2.0 * rr[:, None] + 2.0 * (W1 @ pD2(phi)) - 4.0 * (W2 @ (phi @ Dj)), float) if use else np.zeros_like(Cl)
            k_ = min(20, S); top = np.argpartition(Cl, k_ - 1, axis=1)[:, :k_]; rg = lambda C, ix: np.ptp(np.take_along_axis(C, ix, 1), axis=1)
            ph_ = np.asarray(phi, float); ph_ = ph_ / np.maximum(ph_.sum(1, keepdims=True), 1e-300)
            rec = dict(rec, diag_like_range_top20=float(np.median(rg(Cl, top))), diag_tree_range_top20=float(np.median(rg(Lt_, top))),
                       diag_like_range_all=float(np.median(np.quantile(Cl, 0.95, 1) - np.quantile(Cl, 0.05, 1))), diag_tree_range_all=float(np.median(np.quantile(Lt_, 0.95, 1) - np.quantile(Lt_, 0.05, 1))),
                       diag_eff_spots=float(np.median(np.exp(-(np.where(ph_ > 0, ph_ * np.log(np.maximum(ph_, 1e-300)), 0.0)).sum(1)))))
        trace.append(dict(loglik=float(loglik(th, phi, caps)), jensen_gap=pz, steps=tr[:, 1].tolist(), gain=tr[:, 2].tolist(),
                          row_err=float(jnp.max(jnp.abs(phi.sum(1) / jnp.exp(logr) - 1.0))), **rec, **({"clone_share": np.exp(lrho).tolist(), "clone_step": rho_step} if ctot else {}),
                          **({"clone_mass": np.bincount(c0[on], weights=np.asarray(phi.sum(1))[on], minlength=Kc).tolist()} if (free and use) else {})))
        if verbose:
            print(f"      outer {it}: loglik {trace[-1]['loglik']:.5g}, steps {np.round(tr[:, 1], 2).tolist()}, gain {tr[:, 2].sum():.3g}"
                  + (f", clone shares {np.round(np.exp(lrho), 3).tolist()} (step {rho_step})" if ctot else ""), flush=True)
    costs = {}                                                                                      # the elbow's two curves (Methods: strengths of the terms)
    if cell_cost is not None and beta > 0:
        costs["expression"] = float(jnp.sum(M_of(th) * phi))                                        # the unweighted expression cost of the final map
    if structure is not None:                                                                       # the unweighted structural term, S = <gamma, L> / 2
        costs["structure"] = float(0.5 * jnp.sum(phi * (2.0 * f32(rru)[:, None] + 2.0 * (f32(U) @ pD2(phi)) - 4.0 * (f32(W2u) @ (phi @ Dj)))))
    return dict(phi=np.asarray(phi, float), th=th, trace=trace, weights=rec, costs=costs)


def mrca_sizes(d, tol=1e-9):
    """for an ultrametric distance matrix between cells (a time-calibrated tree, or a cophenetic one), the number of cells in the clade founded
    at each pair's most recent common ancestor; polytomies (several clades joining at one height) count the whole clade they form"""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    d = np.asarray(d, float); n = len(d); Z = linkage(squareform(d, checks=False), method="single")
    members = {i: [i] for i in range(n)}; out = np.ones((n, n)); t = 0
    while t < len(Z):
        h = Z[t, 2]; grp = []
        while t < len(Z) and abs(Z[t, 2] - h) <= tol * max(1.0, abs(h)):
            grp.append(t); t += 1
        comps, new = {}, []
        for g in grp:                                                                              # merge the group, keeping the components it joins
            a, b = int(Z[g, 0]), int(Z[g, 1])
            ca = comps.pop(a, None) or [members[a]]; cb = comps.pop(b, None) or [members[b]]
            members.pop(a); members.pop(b); members[n + g] = sum(ca + cb, [])
            comps[n + g] = ca + cb; new.append(n + g)
        for cid, cl in comps.items():                                                              # pairs across the joined components
            size = sum(len(c) for c in cl)
            for x in range(len(cl)):
                for y in range(x + 1, len(cl)):
                    out[np.ix_(cl[x], cl[y])] = size; out[np.ix_(cl[y], cl[x])] = size
    return out


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


def ct_target(D):
    """the scale C^T is brought to before the fit: the mean distance between spots (any constant does: each round refits the tree distances'
    scale a0 by least squares, so the term does not depend on it)"""
    D = np.asarray(D, float); return float(D.sum() / max(D.shape[0] * (D.shape[0] - 1), 1))


def spreads(P, xy):
    """each clone's RMS distance of its mapped mass from its centre (um or section units)"""
    P = np.asarray(P, float); m = P.sum(0); c = (P.T @ xy) / np.maximum(m[:, None], 1e-300)
    return [float(np.sqrt((P[:, k] * ((xy - c[k]) ** 2).sum(1)).sum() / max(m[k], 1e-300))) for k in range(P.shape[1])]


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


def profiles_denoised(X_all, traced, labels=None, signatures=None, n_latent=10, seed=0, max_epochs=None):
    """each traced cell's posterior-mean expression from scVI fitted on every dissociated cell, with no neighbour averaging
    (decision-no-knn-profiles, exp-unmatched-oct12 section 165): rho_n, the decoder's normalised expression averaged over 25 posterior samples;
    put on counts per cell either by the traced cells' mean library size (signatures=None: references that are raw means) or, gene by gene, so
    that each label's mean equals its signature (signatures: {label: profile}; labels: the traced cells' labels) -- cell2location's signature
    for the label's mean, scVI's posterior for each cell's deviation from it.  X_all: counts of every dissociated cell; traced: boolean mask or
    indices of the traced cells, in the order of the returned rows."""
    import scvi, anndata as ad
    X_all = np.asarray(X_all, float); traced = np.asarray(traced); ti = np.where(traced)[0] if traced.dtype == bool else traced.astype(int)
    a = ad.AnnData(X_all.astype(np.float32)); a.var_names = [f"g{j}" for j in range(X_all.shape[1])]; a.obs_names = [f"c{i}" for i in range(len(X_all))]
    scvi.settings.seed = seed; scvi.model.SCVI.setup_anndata(a); m = scvi.model.SCVI(a, n_latent=n_latent, gene_likelihood="nb")
    m.train(max_epochs=max_epochs, enable_progress_bar=False)
    rho = np.asarray(m.get_normalized_expression(a, indices=ti, library_size=1.0, n_samples=25, return_mean=True), float)
    if signatures is None:
        return rho * X_all[ti].sum(1).mean()
    labels = np.asarray(labels).astype(str); out = np.empty_like(rho)
    for l in np.unique(labels):
        j = labels == l; out[j] = rho[j] / np.maximum(rho[j].mean(0, keepdims=True), 1e-12) * np.asarray(signatures[l], float)[None, :]
    return out


def profiles(Xt, clone, k=15, se2=False):
    """NOT TO BE USED for SpOTr's profiles (decision-no-knn-profiles, 2026-11-01: kNN means of raw counts are too noisy; use
    profiles_denoised or subclone signatures).  Kept so that earlier scripts still run.
    cell: 319_'s across-clone kNN means; within: kNN means within each clone (cells without a clone: across); clone: clone means;
    with se2, also within_se2: each within profile's squared standard error, the variance over its k neighbours over k (exp-unmatched-oct12
    section 132)"""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    L = np.log1p(Xt / np.maximum(Xt.sum(1, keepdims=True), 1.0) * 1e4); Z = PCA(n_components=min(30, L.shape[1] - 1), random_state=0).fit_transform(L - L.mean(0))
    idx = NearestNeighbors(n_neighbors=min(k, len(Xt))).fit(Z).kneighbors(Z, return_distance=False); cell = Xt[idx].mean(1)
    within, clm = cell.copy(), cell.copy()
    def sem2(rows_, nb):                                                                            # in chunks: n x k x G at once is large
        out = np.zeros((len(rows_), Xt.shape[1]))
        for a0 in range(0, len(rows_), 256):
            blk = Xt[nb[a0:a0 + 256]]; out[a0:a0 + 256] = blk.var(1, ddof=1) / blk.shape[1] if blk.shape[1] > 1 else 0.0
        return out
    se = sem2(np.arange(len(Xt)), idx) if se2 else None
    for c in np.unique(clone[clone >= 0]):
        r = np.where(clone == c)[0]; kk = min(k, len(r)); j = NearestNeighbors(n_neighbors=kk).fit(Z[r]).kneighbors(Z[r], return_distance=False)
        within[r] = Xt[r][j].mean(1); clm[r] = Xt[r].mean(0)
        if se2:
            se[r] = sem2(r, r[j])
    out = dict(cell=cell, within=within, clone=clm)
    if se2:
        out["within_se2"] = se
    return out
