"""SpOTr's engine: a joint fit of a spot deconvolution model and an entropic Fused Gromov-Wasserstein map of the dissociated cancer cells.

The spots follow the negative binomial model of cell2location, in which the expected counts of each spot sum reference signatures weighted by
their abundances; the cancer part of each spot is the sum of the clone signatures of the cells mapped to it.  The map minimises

    F(theta, gamma) = -log p(Y | theta, gamma) - log p(theta) + beta <C^F, gamma> + alpha S(gamma) + <gamma, log gamma>

over maps that place each cancer cell exactly once and give each spot a number of cancer cells proportional to its cancer share.  C^F is the
cosine distance between each cell's and each spot's cancer part in log-normalised expression, and S the structural term: distance matching
between tree distances and spot distances, or, with clone labels only, the spread of each clone over the spots.  The strengths of the
expression cost and of the structural term are set against the likelihood at the start of the fit (beta / (1 - beta) and
alpha / (1 - alpha) times the likelihood's spread).  The map is updated by generalised conditional gradient, with the linearised objective
solved by Sinkhorn in the log domain, in alternation with Adam steps on the deconvolution's parameters.  Supplementary Note 1 of the paper
derives the linearised structural term.
"""
import numpy as np

N_OUTER, J, IT0, IT1, ADAM0, ADAM1 = 16, 12, 300, 100, 1000, 200                                # rounds; map updates, Sinkhorn sweeps and Adam steps
ETAS = [1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0]                                     # the map's step sizes
SINK_TOL, SINK_MAX = 1e-4, 5000                                                                  # Sinkhorn: rows to within 1e-4, at most 5,000 sweeps


def fit(Y, G_ref, g_untraced, X_sig, X_cells, xy, clone, alpha=0.2, beta=0.5, tree_d=None, omega=None, verbose=False):
    """Fit the deconvolution and the map jointly.

    Y           spots x genes counts.
    G_ref       reference types x genes signatures of the microenvironment.
    g_untraced  1 x genes signature of the cancer cells without a clone (a reference type of its own), or None when every cancer cell
                is mapped.
    X_sig       mapped cells x genes: each mapped cell's clone signature.
    X_cells     mapped cells x genes: each mapped cell's own counts (for the expression cost).
    xy          spots x 2 positions.
    clone       each mapped cell's clone (0 .. K - 1).
    alpha       structural importance in [0, 1); beta the expression cost's weight in [0, 1).
    tree_d      mapped cells x mapped cells tree distances (distance matching), or None (clone labels: each clone drawn together).
    omega       optional {clone: weight} on the structural term (0: that clone mapped by expression alone).
    Returns dict(phi: cells x spots map, th: the deconvolution's parameters, costs: the unweighted expression and structural costs of the
    final map, trace: per round log-likelihood and steps).
    """
    import jax, jax.numpy as jnp, optax
    from functools import partial
    from jax.scipy.special import logsumexp, gammaln
    from scipy.spatial.distance import pdist, squareform
    Y, xy, clone = np.asarray(Y, float), np.asarray(xy, float), np.asarray(clone, int)
    Gme = np.asarray(G_ref, float).reshape(-1, Y.shape[1])
    if g_untraced is not None:                                                                  # untraced cancer cells: a type of their own
        Gme = np.vstack([Gme, np.asarray(g_untraced, float).reshape(1, -1)])
    D = squareform(pdist(xy)); S, G = Y.shape; E = Gme.shape[0]; n = np.asarray(X_sig).shape[0]
    f32 = lambda a: jnp.asarray(np.asarray(a, np.float32))
    Yj, Gj, Xj = f32(Y), f32(Gme), f32(X_sig); mc = float(Y.mean())
    xc = xy - xy.mean(0); Lj, sqj, Dj = f32(xc), f32((xc ** 2).sum(1)), f32(D)
    th = dict(logN=f32(np.full(S, np.log(10.0))), z=f32(np.zeros((S, E + 1))), logm=f32(np.zeros(G)), logy=f32(np.zeros(S)),
              logs=f32(np.full(G, np.log(1e-3 * mc))), logr=f32(np.full(G, np.log(10.0))))

    def parts(th):                                                                              # abundances: reference types, cancer cells
        w = jnp.exp(th["logN"])[:, None] * jax.nn.softmax(th["z"], axis=1); return w[:, :E], w[:, E:]

    def caps_of(th):                                                                            # each spot's number of mapped cells: its cancer share,
        wt = parts(th)[1] / jnp.exp(th["logN"])[:, None]                                        # floored at a millionth of the mean
        c = (wt / wt.sum(0, keepdims=True) * float(n)).T
        c = jnp.maximum(c, 1e-6 * float(n) / S); return c * (float(n) / c.sum(1))[:, None]

    def mu_of(th, phi, caps):
        we, wt = parts(th); A = (wt / jnp.maximum(caps.T, 1e-12))[:, 0]
        expr = we @ Gj + (phi * A[None, :]).T @ Xj
        return (jnp.exp(th["logm"])[None, :] * expr + jnp.exp(th["logs"])[None, :]) * jnp.exp(th["logy"])[:, None]

    def loglik(th, phi, caps):
        mu = mu_of(th, phi, caps); r = jnp.exp(th["logr"])[None, :]
        return jnp.sum(gammaln(Yj + r) - gammaln(r) - gammaln(Yj + 1.0) + r * jnp.log(r / (r + mu)) + Yj * jnp.log(mu / (r + mu) + 1e-30))

    def logprior(th):                                                                           # cell2location's priors
        return (jnp.sum(9.0 * th["logN"] - jnp.exp(th["logN"])) + jnp.sum(19.0 * th["logy"] - 20.0 * jnp.exp(th["logy"])) - 0.5 * jnp.sum(th["logm"] ** 2)
                - jnp.sum((th["logs"] - np.log(1e-3 * mc)) ** 2) / 8.0 - jnp.sum((th["logr"] - np.log(10.0)) ** 2) / 8.0)

    opt = optax.adam(0.02); st = opt.init(th)

    @partial(jax.jit, static_argnums=4)
    def adam(th, st, phi, caps, steps):
        def body(c, _):
            th, st = c; g = jax.grad(lambda t: -(loglik(t, phi, caps) + logprior(t)))(th)
            up, st = opt.update(g, st, th); return (optax.apply_updates(th, up), st), None
        (th, st), _ = jax.lax.scan(body, (th, st), None, length=steps); return th, st

    def sink(Cst, logcap, f, g, iters):
        """log-domain Sinkhorn: at least `iters` sweeps, then on until every row sums to 1 within SINK_TOL"""
        def body(_, s):
            f, g = s; f = -logsumexp(g[0] - Cst, axis=1); g = jnp.stack([logcap[0] - logsumexp(f[:, None] - Cst, axis=0)]); return (f, g)
        def err(f, g):
            return jnp.max(jnp.abs(jnp.exp(logsumexp(f[:, None] + g[0] - Cst, axis=1)) - 1.0))
        f, g = jax.lax.fori_loop(0, iters, body, (f, g))
        def more(s):
            f, g, k = s; f, g = jax.lax.fori_loop(0, 10, body, (f, g)); return (f, g, k + 10)
        f, g, _ = jax.lax.while_loop(lambda s: (s[2] < SINK_MAX) & (err(s[0], s[1]) > SINK_TOL), more, (f, g, iters)); return f, g

    def pD2(p):                                                                                 # p (C^S)^2 from the spots' coordinates
        return (p @ sqj)[:, None] + p.sum(1)[:, None] * sqj[None, :] - 2.0 * (p @ Lj) @ Lj.T

    @jax.jit
    def gcg(phi, f, g, th, caps, W1, W2, rr, Mb):
        """J conditional gradient steps of the map: the linearised objective solved by Sinkhorn, the step chosen on the exact objective"""
        logcap = jnp.log(jnp.maximum(caps, 1e-30))
        def step(c, _):
            phi, f, g = c; Lt = 2.0 * rr[:, None] + 2.0 * (W1 @ pD2(phi)) - 4.0 * (W2 @ (phi @ Dj)) + Mb
            Cst = -1.0 * jax.grad(lambda p: loglik(th, p, caps))(phi) + Lt; f, g = sink(Cst, logcap, f, g, IT1); dl = jnp.exp(f[:, None] + g[0] - Cst) - phi
            q1 = jnp.sum(dl * Lt); q2 = jnp.sum(dl * (W1 @ pD2(dl))) - 2.0 * jnp.sum(dl * (W2 @ (dl @ Dj)))
            mu0 = mu_of(th, phi, caps); dmu = mu_of(th, phi + dl, caps) - mu0; r = jnp.exp(th["logr"])[None, :]   # spot means are affine in the map
            def dobj(e):                                                                        # the objective's change for a step e
                dll = jnp.sum(Yj * jnp.log1p(e * dmu / mu0) - (Yj + r) * jnp.log1p(e * dmu / (r + mu0)))
                pe = jnp.maximum(phi + e * dl, 1e-30); p0 = jnp.maximum(phi, 1e-30)
                return -dll + e * q1 + e * e * q2 + jnp.sum(pe * jnp.log(pe) - p0 * jnp.log(p0))
            vals = jnp.stack([dobj(e) for e in ETAS]); e = jnp.asarray(ETAS)[jnp.argmin(vals)]
            return (phi + e * dl, f, g), jnp.stack([vals.min(), e, vals[-1] - vals.min()])
        (phi, f, g), tr = jax.lax.scan(step, (phi, f, g), None, length=J); return phi, f, g, tr

    # start: the capacities' uniform map, then Adam on the deconvolution
    caps = caps_of(th); f, g = jnp.zeros(n, jnp.float32), jnp.zeros((1, S), jnp.float32)
    f, g = sink(jnp.zeros((n, S), jnp.float32), jnp.log(caps), f, g, IT0); phi = jnp.exp(f[:, None] + g[0]); th, st = adam(th, st, phi, caps, ADAM0)
    dcs = lambda C: float((C - C.mean(1, keepdims=True) - C.mean(0, keepdims=True) + C.mean()).std())   # spread of a double-centred cost
    # the structural term: pair weights U and target distances Tg, its strength set against the likelihood's spread
    if tree_d is not None:
        U = np.ones((n, n)); dd = np.asarray(tree_d, float); iu = np.triu_indices(n, 1)
        Tg = float(D[np.triu_indices(S, 1)].mean() / max(dd[iu].mean(), 1e-12)) * dd             # tree distances rescaled to the spots' mean distance
    else:
        U = (clone[:, None] == clone[None, :]).astype(float); Tg = np.zeros((n, n))
    np.fill_diagonal(U, 0.0); U = U / max(U.sum(), 1e-30) * n; W2u = U * Tg; rru = (W2u * Tg).sum(1)
    Cl = -np.asarray(jax.grad(lambda p_: loglik(th, p_, caps))(phi), float)
    Ltu = np.asarray(2.0 * f32(rru)[:, None] + 2.0 * (f32(U) @ pD2(phi)) - 4.0 * (f32(W2u) @ (phi @ Dj)), float)
    lam = dcs(Cl) / max(dcs(Ltu), 1e-30) * float(alpha) / max(1.0 - float(alpha), 1e-9)
    if omega is not None:                                                                       # clade weights, after the calibration
        om = np.array([float(omega.get(int(k), 1.0)) for k in clone]); U = U * np.outer(om, om); W2u = U * Tg; rru = (W2u * Tg).sum(1)
    W1, W2, rr = f32(lam * U), f32(lam * W2u), f32(lam * rru)
    # the expression cost: cosine distance between each cell and each spot's cancer part, in log-normalised expression
    Xn = np.asarray(X_cells, float); Lc = f32(np.log1p(Xn / np.maximum(Xn.sum(1, keepdims=True), 1.0) * 1e4)); Lcn = Lc / jnp.maximum(jnp.sqrt(jnp.sum(Lc ** 2, 1)), 1e-12)[:, None]

    def M_of(t):
        we_, _ = parts(t); rest = (jnp.exp(t["logm"])[None, :] * (we_ @ Gj) + jnp.exp(t["logs"])[None, :]) * jnp.exp(t["logy"])[:, None]
        Ys = jnp.maximum(Yj - rest, 0.0); Ls = jnp.log1p(Ys / jnp.maximum(Ys.sum(1, keepdims=True), 1.0) * 1e4)
        return 1.0 - Lcn @ (Ls / jnp.maximum(jnp.linalg.norm(Ls, axis=1, keepdims=True), 1e-12)).T

    M0 = np.asarray(M_of(th), float); beff = dcs(Cl) / max(dcs(M0), 1e-30) * float(beta) / max(1.0 - float(beta), 1e-9) if beta > 0 else 0.0
    trace = []
    for it in range(N_OUTER):
        caps = caps_of(th); lphi = jnp.log(jnp.maximum(phi, 1e-30)); f, g = sink(-lphi, jnp.log(jnp.maximum(caps, 1e-30)), f, g, IT0); phi = jnp.exp(f[:, None] + g[0] + lphi)
        Mb = f32(beff * np.asarray(M_of(th), float))
        phi, f, g, tr = gcg(phi, f, g, th, caps, W1, W2, rr, Mb); th, st = adam(th, st, phi, caps, ADAM1)
        tr = np.asarray(tr, float); trace.append(dict(loglik=float(loglik(th, phi, caps)), steps=tr[:, 1].tolist()))
        if verbose:
            print(f"round {it}: log-likelihood {trace[-1]['loglik']:.6g}", flush=True)
    costs = dict(expression=float(jnp.sum(M_of(th) * phi)),
                 structure=float(0.5 * jnp.sum(phi * (2.0 * f32(rru)[:, None] + 2.0 * (f32(U) @ pD2(phi)) - 4.0 * (f32(W2u) @ (phi @ Dj))))))
    return dict(phi=np.asarray(phi, float), th=th, trace=trace, costs=costs)


def abund(th):
    """each spot's abundances: the reference types, the untraced cancer cells (when fitted), the mapped cancer cells"""
    z = np.asarray(th["z"], float); e = np.exp(z - z.max(1, keepdims=True)); return (np.exp(np.asarray(th["logN"], float))[:, None] * e / e.sum(1, keepdims=True)).astype(np.float32)
