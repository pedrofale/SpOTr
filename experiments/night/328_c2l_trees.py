"""Our method with three trees -- clone labels only, the clone tree, the lineage tree -- and cell2location as published (no GW), on the
simulations and PEtracer.  2026-10-29.
Pedro, 2026-10-29: "From now on we also need to include the basic cell2location without our GW part in the comparisons. And let's always
run our own method with 1) clone labels only, 2) clone tree, and 3) lineage tree."
The tree prior generalises 321_'s clone clouds to the clades of a tree: each clade c (a set of traced cells) is a cloud whose centre is
integrated out and whose spread sigma_c^2 is learned -- the masked GW term sum_c 1 / (2 n_c sigma_c^2) sum_{i<j in c} E|x_i - x_j|^2, one
mask per clade (the clade-aware Omega), with 321_'s closed-form spread step per clade (a product of clade clouds, each normalised on its
own), and 321_'s within_free otherwise unchanged (cell2location likelihood, within-clone profiles, free spreads, the no-tree map's level).
  clone labels   the clades are the clones, a star (321_ / 323_'s within_free, reused)
  clone tree     the clones and every internal node of the tree among them below the root (average linkage on the clone-to-clone mean tree
                 distance, equal heights merged): the true clone tree in the simulations, where the clones are clades of the lineage; the
                 authors' clades are not exactly monophyletic in PEtracer's tree, so there it is the tree of the clades
  lineage tree   every node of the lineage tree (ultrametric; equal heights merged) with >= 10 traced cells, below the root; the traced
                 cells without a clone enter here and only here
cell2location as published, without the GW part: a transcription of its spatial model (LocationModelLinearDependentWMultiExperiment-
LocationBackgroundNormLevelGeneAlpha) -- N_s ~ Gamma(N, 1), B_s ~ Gamma(7, 1), z_sr ~ Gamma(B_s / R, B_s / N_s) with R = 50,
k_r ~ Gamma(1, 1), x_rf ~ Gamma(k_r / F, k_r), w_sf ~ Gamma(5 mu_sf, 5) with mu = z x; gene sensitivity m_g ~ Gamma(1 / e^2, 1 / (e^2 mu_m)),
mu_m ~ Gamma(1, 1), e ~ Exp(h), h ~ Gamma(3, 1); detection y_s ~ Gamma(20, 20 / y_e), y_e ~ Gamma(10, 20); background s_g likewise with
mean ~ Gamma(1, 100) and h ~ Gamma(9, 3); NB overdispersion 1 / a_g^2, a_g ~ Exp(h), h ~ Gamma(9, 3); mu_sg = (w_s g_g m_g + s_g) y_s --
fitted as the package does (mean-field log-normal VI, Adam lr 0.002 clipped at norm 10, 30,000 full-data steps); signatures = each cell
type's mean counts, every clone its own type; N = the mean cells per spot (the user-set prior); a clone's abundance = w_sf's posterior mean.
The package is not installed anywhere here and downloaded code is not run, so this is a reimplementation.
Problems: simulations W1-W7 x seeds 46, 47 (323_'s grid); PEtracer half M3-4, M3-1, M3-2 and serial, seed 0 (where v1 and the tools exist).
FIXED BEFORE RUNNING:
  E165  simulations, oriented worlds (12 world-seeds): per-spot subclone error of the lineage-tree arm <= the clone-tree arm + 0.02 <= the
        clone-label arm + 0.04 in >= 9 of 12, and the lineage-tree arm within 0.05 of v1 (which has the same true lineage tree) in >= 8 of 12
  E166  cell2location alone is worse than our method with clone labels: higher per-spot subclone error in >= 10 of 12 oriented world-seeds,
        lower per-clade PCC in 4 of 4 PEtracer designs
  E167  PEtracer: per-clade PCC of the clone-tree arm >= the clone-label arm - 0.01 in >= 3 of 4 designs
  read  the lineage arm on PEtracer; the maps; the learned clade spreads
Usage: python 328_c2l_trees.py    (writes spotr_out/spotr_c2l_trees/<problem>.json)
"""
import os, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE / "spotr_out"; OUT = O / "spotr_c2l_trees"
PROBLEMS = ([("sim", w, 46) for w in ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]] + [("petracer", p, 0) for p in ["half:M3-4", "half:M3-1", "half:M3-2", "serial:M3-1>M3-2"]]
            + [("sim", w, 47) for w in ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]])
MIN_CLADE = 10


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def stem(src, name, seed):
    return f"{src}_{name.replace(':', '_').replace('>', '_to_')}_{seed}" if src == "petracer" else f"{name}_{seed}"


def node_sets(d):
    """leaf sets of the internal nodes of an ultrametric tree given by its distances, equal heights merged, root excluded"""
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    if len(d) < 3:
        return []
    root = to_tree(linkage(squareform(d, checks=False), method="average")); tol = 1e-9 * max(root.dist, 1.0); out, stack = [], [(root, None)]
    while stack:
        nd, ph = stack.pop()
        if nd.is_leaf():
            continue
        if ph is not None and abs(ph - nd.dist) > tol:
            out.append(np.array(nd.pre_order()))
        stack += [(nd.left, nd.dist), (nd.right, nd.dist)]
    return out


def clade_matrix(kind, d, clone, Kc):
    """membership (traced cells x clades); the clones first in 'clones' and 'clone_tree'"""
    n = len(clone); cols = [np.where(clone == k)[0] for k in range(Kc)]
    if kind == "clone_tree":
        Dk = np.array([[d[np.ix_(cols[a], cols[b])].mean() if a != b else 0.0 for b in range(Kc)] for a in range(Kc)])
        cols = cols + [np.concatenate([cols[k] for k in s]) for s in node_sets(Dk)]
    elif kind == "lineage":
        cols = [s for s in node_sets(d) if len(s) >= MIN_CLADE]
    B = np.zeros((n, len(cols)))
    for j, c in enumerate(cols):
        B[c, j] = 1.0
    return B


def nesting(B):
    """each clade's parent (the smallest clade strictly containing it, -1 for none) and its number of children (child clades + cells directly in it)"""
    n_c = B.sum(0); ov = B.T @ B; Q = B.shape[1]; par = np.full(Q, -1)
    for c in range(Q):
        cand = np.where((ov[c] == n_c[c]) & (n_c > n_c[c]))[0]
        if len(cand):
            par[c] = cand[np.argmin(n_c[cand])]
    kids = np.bincount(par[par >= 0], minlength=Q); inner = np.bincount(par[par >= 0], weights=n_c[par >= 0], minlength=Q)
    return par, kids + (n_c - inner)


def exact_rho(Wc, B, b0, a0, rho, n_top):
    """the clade precisions rho_c = 1 / s_c^2 under the exact normaliser (tried 2026-10-29 and set aside -- it learns one weight per clade,
    i.e. the tree's branch lengths; Pedro: "we definitely shouldn't be learning the matrix"; 331_ fixes the matrix instead): maximise -1/2 sum_c rho_c W_c + log det+ Lambda(rho)
    + sum_c [(a0 - 1) log rho_c - b0 rho_c], Lambda = sum_c rho_c H_c on the cells in some clade (2D: 2 x 1/2 log det+), W_c the expected
    within-clade sum of squares; concave in rho, and for a star it is 321_'s closed form (a0 + n_c - 2) / (b0 + W_c / 2)"""
    from scipy.optimize import minimize
    Bk = B[B.sum(1) > 0]; nc = np.maximum(Bk.sum(0), 1.0)
    M, mg = np.unique(Bk, axis=0, return_counts=True); sq = np.sqrt(mg)                 # groups of cells with the same clades: Lambda acts on
    def f(u):                                                                          # each group's contrasts as r_g, so it reduces to groups
        r = np.exp(u); rg = M @ r; L = np.diag(rg) - (sq[:, None] * M * (r / nc)) @ (M * sq[:, None]).T
        ev, U = np.linalg.eigh(L); ev, U = ev[n_top:], U[:, n_top:]; Lp = (U / ev) @ U.T
        tr = M.T @ ((mg - 1.0) / rg + np.diag(Lp)) - (M * (Lp @ (M * sq[:, None])) * sq[:, None]).sum(0) / nc   # tr(Lambda+ H_c)
        J = -0.5 * (r * Wc).sum() + ((mg - 1.0) * np.log(rg)).sum() + np.log(ev).sum() + ((a0 - 1.0) * u - b0 * r).sum()
        return -J, -(r * (-0.5 * Wc + tr - b0) + (a0 - 1.0))
    res = minimize(f, np.log(rho), jac=True, method="L-BFGS-B", bounds=[(np.log(1e-14), np.log(1e2))] * len(rho), options=dict(maxiter=500))
    return np.exp(res.x)


def fit(Y, Gme, gu, Xh, fT, B, D, level, nested=False, norm="separable", rho0=None):
    """321_'s within_free (319_'s fit, arm free) with the clade membership B in place of the clone one-hot; the tree cost is the gradient of
    the clade sum for overlapping clades.  nested (330_): each clade's cloud is of its children's centres about its own (the within-clade sum
    of squares minus its child clades'), so a cell is not counted again at every level -- weight 1 / s_c^2 - 1 / s_parent^2 on clade c's
    masked term, and the spread step on the between-children sum of squares with (children - 1) degrees of freedom; a star is unchanged.
    norm 'exact' (set aside, see exact_rho): the clade precisions by exact_rho (the Gaussian's own normaliser, log det+), started at rho0"""
    import jax, jax.numpy as jnp, optax
    jax.config.update("jax_enable_x64", True); from jax.scipy.special import logsumexp, gammaln
    from functools import partial
    m319 = mod("m319", "319_spotr_c2l.py"); N_OUTER, J, IT0, IT1, ADAM0, ADAM1, ETAS = m319.N_OUTER, m319.J, m319.IT0, m319.IT1, m319.ADAM0, m319.ADAM1, m319.ETAS
    S, G = Y.shape; E = Gme.shape[0]; n = Xh.shape[0]; Yj, Gj, guj, Xj, D2 = map(jnp.asarray, (Y, Gme, gu, Xh, D * D)); Q = B.shape[1]
    nk = B.sum(0); Bj = jnp.asarray(B); nkj = jnp.asarray(np.maximum(nk, 1.0)); mc = float(Y.mean())
    th = dict(logN=jnp.full(S, np.log(10.0)), z=jnp.zeros((S, E + 1)), logm=jnp.zeros(G), logy=jnp.zeros(S), logs=jnp.full(G, np.log(1e-3 * mc)), logr=jnp.full(G, np.log(10.0)))
    def parts(th):
        N = jnp.exp(th["logN"]); pi = jax.nn.softmax(th["z"], axis=1); w = N[:, None] * pi; return w, w[:, E]
    def caps_of(th):
        wt = parts(th)[1]; return n * wt / wt.sum()
    def mu_of(th, phi):
        w, wt = parts(th); cs = jnp.maximum(phi.sum(0), 1e-12); h = (phi.T @ Xj) / cs[:, None]
        expr = w[:, :E] @ Gj + wt[:, None] * ((1.0 - fT) * guj[None, :] + fT * h)
        return (jnp.exp(th["logm"])[None, :] * expr + jnp.exp(th["logs"])[None, :]) * jnp.exp(th["logy"])[:, None]
    def loglik(th, phi):
        mu = mu_of(th, phi); r = jnp.exp(th["logr"])[None, :]
        return jnp.sum(gammaln(Yj + r) - gammaln(r) - gammaln(Yj + 1.0) + r * jnp.log(r / (r + mu)) + Yj * jnp.log(mu / (r + mu) + 1e-300))
    def logprior(th):
        N = jnp.exp(th["logN"]); y = jnp.exp(th["logy"])
        return (jnp.sum(9.0 * th["logN"] - N) + jnp.sum(19.0 * th["logy"] - 20.0 * y) - 0.5 * jnp.sum(th["logm"] ** 2)
                - jnp.sum((th["logs"] - np.log(1e-3 * mc)) ** 2) / 8.0 - jnp.sum((th["logr"] - np.log(10.0)) ** 2) / 8.0)
    opt = optax.adam(0.02); st = opt.init(th)
    @partial(jax.jit, static_argnums=3)
    def adam(th, st, phi, steps):
        def body(c, _):
            th, st = c; g = jax.grad(lambda t: -(loglik(t, phi) + logprior(t)))(th); up, st = opt.update(g, st, th); return (optax.apply_updates(th, up), st), None
        (th, st), _ = jax.lax.scan(body, (th, st), None, length=steps); return th, st
    def sink(Cst, logc, f, g, iters):
        def body(_, s):
            f, g = s; f = -logsumexp(g[None, :] - Cst, axis=1); g = logc - logsumexp(f[:, None] - Cst, axis=0); return (f, g)
        return jax.lax.fori_loop(0, iters, body, (f, g))
    def quad(phi, wk):
        C = Bj.T @ phi; self_ = ((phi @ D2) * phi).sum(1); Sk = 0.5 * ((C @ D2) * C).sum(1) - 0.5 * (Bj.T @ self_)
        return jnp.sum(wk / (2.0 * nkj) * Sk), Sk
    def tree_cost(phi, wk):
        C = Bj.T @ phi; a = wk / (2.0 * nkj); return Bj @ (a[:, None] * (C @ D2)) - (Bj @ a)[:, None] * (phi @ D2)
    def F(phi, th, wk):
        return -loglik(th, phi) + quad(phi, wk)[0] + jnp.sum(phi * jnp.log(jnp.maximum(phi, 1e-300)))
    @jax.jit
    def gcg(phi, f, g, th, wk, logc):
        def step(c, _):
            phi, f, g = c; Cst = -jax.grad(lambda p: loglik(th, p))(phi) + tree_cost(phi, wk); f, g = sink(Cst, logc, f, g, IT1); ph = jnp.exp(f[:, None] + g[None, :] - Cst)
            vals = jax.lax.map(lambda e: F(phi + e * (ph - phi), th, wk), jnp.asarray(ETAS)); e = jnp.asarray(ETAS)[jnp.argmin(vals)]
            return (phi + e * (ph - phi), f, g), vals.min()
        (phi, f, g), tr = jax.lax.scan(step, (phi, f, g), None, length=J); return phi, f, g, tr
    a0, b0 = np.full(Q, 1.0), np.full(Q, 1e-3 * float(np.median(D[D > 0])) ** 2); s2 = np.full(Q, level) if rho0 is None else 1.0 / rho0
    par, m_c = nesting(B) if nested else (np.full(Q, -1), nk); top = par < 0
    weff = lambda s2: 1.0 / s2 - np.where(top, 0.0, 1.0 / s2[np.maximum(par, 0)])
    logc = jnp.log(caps_of(th)); f, g = jnp.zeros(n), jnp.zeros(S); f, g = sink(jnp.zeros((n, S)), logc, f, g, IT0); phi = jnp.exp(f[:, None] + g[None, :])
    th, st = adam(th, st, phi, ADAM0); trace = []
    for it in range(N_OUTER):
        logc = jnp.log(caps_of(th)); f, g = sink(-jnp.log(jnp.maximum(phi, 1e-300)), logc, f, g, IT0); phi = jnp.exp(f[:, None] + g[None, :] + jnp.log(jnp.maximum(phi, 1e-300)))
        phi, f, g, tr = gcg(phi, f, g, th, jnp.asarray(weff(s2)), logc); th, st = adam(th, st, phi, ADAM1); Sk = np.asarray(quad(phi, jnp.ones(Q))[1])
        Wc = Sk / np.maximum(nk, 1.0); Tc = np.maximum(Wc - np.bincount(par[~top], weights=Wc[~top], minlength=Q), 0.0) if nested else Wc
        if norm == "exact":
            s2 = 1.0 / exact_rho(Wc, B, b0, a0, 1.0 / s2, int((nesting(B)[0] < 0).sum()))
        else:
            s2 = (b0 + Tc / 2.0) / np.maximum(a0 + np.maximum(m_c - 1, 0) - 1.0, 1e-9)
        trace.append(dict(loglik=float(loglik(th, phi)), map_obj=float(tr[-1]), s2=[float(x) for x in s2]))
    return dict(phi=np.asarray(phi), s2=s2, trace=trace)


def c2l_basic(Y, sig, Nhat, steps=30000, lr=0.002, seed=0, x64=False, init_ls=0.1):
    """cell2location's spatial model (see the docstring), mean-field log-normal VI; returns w_sf's posterior mean (spots x types).  x64 and
    a smaller init_ls (335_, HTAN's Visium counts, where float32 with the default noise diverged) keep the defaults' behaviour otherwise"""
    import jax, jax.numpy as jnp, optax
    from jax.scipy.special import gammaln
    if x64:
        jax.config.update("jax_enable_x64", True)
    S, G = Y.shape; Fn = sig.shape[0]; R = 50; dt = jnp.float64 if x64 else jnp.float32; f32 = lambda a: jnp.asarray(np.asarray(a, dt)); Yj, Sg = f32(Y), f32(sig)
    init = dict(N=(S, Nhat), Bs=(S, 7.0), z=((S, R), Nhat / R), k=(R, 1.0), x=((R, Fn), 1.0 / Fn), w=((S, Fn), Nhat / Fn), mmu=(1, 1.0), mh=(1, 3.0), me=(G, 1 / 3),
                m=(G, 1.0), ye=(1, 0.5), y=(S, 0.5), sh=(1, 3.0), smu=(1, 0.01), se=(1, 1 / 3), s=(G, 0.01), ah=(1, 3.0), ae=(G, 1 / 3))
    p = {k: dict(loc=jnp.full(sh, np.log(v), dt), ls=jnp.full(sh, np.log(init_ls), dt)) for k, (sh, v) in init.items()}
    lg = lambda x, lx, a, b: a * jnp.log(b) - gammaln(a) + (a - 1.0) * lx - b * x
    lexp = lambda x, rate: jnp.log(rate) - rate * x
    def logjoint(u):
        e = {k: jnp.exp(v) for k, v in u.items()}; lp = lg(e["N"], u["N"], Nhat, 1.0).sum() + lg(e["Bs"], u["Bs"], 7.0, 1.0).sum()
        lp += lg(e["z"], u["z"], e["Bs"][:, None] / R, (e["Bs"] / e["N"])[:, None]).sum() + lg(e["k"], u["k"], 1.0, 1.0).sum()
        lp += lg(e["x"], u["x"], e["k"][:, None] / Fn, e["k"][:, None]).sum() + lg(e["w"], u["w"], 5.0 * (e["z"] @ e["x"]), 5.0).sum()
        lp += lg(e["mmu"], u["mmu"], 1.0, 1.0).sum() + lg(e["mh"], u["mh"], 3.0, 1.0).sum() + lexp(e["me"], e["mh"]).sum()
        am = 1.0 / e["me"] ** 2; lp += lg(e["m"], u["m"], am, am / e["mmu"]).sum()
        lp += lg(e["ye"], u["ye"], 10.0, 20.0).sum() + lg(e["y"], u["y"], 20.0, 20.0 / e["ye"]).sum()
        lp += lg(e["sh"], u["sh"], 9.0, 3.0).sum() + lg(e["smu"], u["smu"], 1.0, 100.0).sum() + lexp(e["se"], e["sh"]).sum()
        a_s = 1.0 / e["se"] ** 2; lp += lg(e["s"], u["s"], a_s, a_s / e["smu"]).sum()
        lp += lg(e["ah"], u["ah"], 9.0, 3.0).sum() + lexp(e["ae"], e["ah"]).sum()
        r = (1.0 / e["ae"] ** 2)[None, :]; mu = ((e["w"] @ Sg) * e["m"][None, :] + e["s"][None, :]) * e["y"][:, None]
        ll = jnp.sum(gammaln(Yj + r) - gammaln(r) - gammaln(Yj + 1.0) + r * jnp.log(r / (r + mu)) + Yj * jnp.log(mu / (r + mu) + 1e-30))
        return ll + lp + sum(v.sum() for v in u.values())                                         # + log |dx / du| of x = exp(u)
    def neg_elbo(p, key):
        ks = jax.random.split(key, len(p)); u = {k: p[k]["loc"] + jnp.exp(p[k]["ls"]) * jax.random.normal(kk, p[k]["loc"].shape, dt) for k, kk in zip(p, ks)}
        return -(logjoint(u) + sum(p[k]["ls"].sum() for k in p))
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr)); st = opt.init(p)
    @jax.jit
    def run(p, st, key):
        def body(c, k):
            p, st = c; l, g = jax.value_and_grad(neg_elbo)(p, k); up, st = opt.update(g, st, p); return (optax.apply_updates(p, up), st), l
        (p, st), ls = jax.lax.scan(body, (p, st), jax.random.split(key, steps)); return p, ls
    p, ls = run(p, st, jax.random.PRNGKey(seed)); ls = np.asarray(ls)
    w = np.asarray(jnp.exp(p["w"]["loc"] + 0.5 * jnp.exp(2.0 * p["w"]["ls"])), float)
    return w, dict(elbo=[float(-ls[i]) for i in range(0, steps, 1000)] + [float(-ls[-1])])


def clone_start(B, clone, Kc, level):
    """the per-clade exact version's start (set aside, see exact_rho): the clone-label prior -- each clone's clade (the clade with the clone's cells, best Jaccard >= 0.5) at the no-tree
    level, every other clade at 1e-3 of it -- so the tree's other levels are switched on by the data"""
    rho = np.full(B.shape[1], 1e-3 / level)
    for k in range(Kc):
        ink = (clone == k).astype(float); inter = ink @ B; jac = inter / (ink.sum() + B.sum(0) - inter)
        if jac.max() >= 0.5:
            rho[int(np.argmax(jac))] = 1.0 / level
    return rho


def run_problem(src, name, seed, f, arms=("c2l", "clone_tree", "lineage"), nested=False, norm="separable"):
    import warnings; warnings.filterwarnings("ignore")
    m321, m316, f289 = mod("m321", "321_c2l_profiles.py"), mod("m316", "316_spotr_vb.py"), mod("f289", "289_field_metrics.py")
    if src == "sim":
        m268, m239, m189 = mod("m268", "268_lineage_cv.py"), mod("m239", "239_deconv_fgw.py"), mod("m189", "189_me_sim.py")
        NB, X, Xc, t_f, _ = m268.build(seed, **mod("m270", "270_lineage_cv_concordance.py").WORLDS[name]); Y = X["Sc"]; Nb = m239.window(X); d = m239.tree_dists(X, NB, t_f, X["q"])[1]["lineage"]
        prev = O / "spotr_c2l_sims" / f"{name}_{seed}.json"
    else:
        P = mod("m276", "276_petracer.py").build(name, seed); X, Xc, Y = P["X"], P["Xc"], P["Y"]; Nb = m239 = m189 = None; d = np.asarray(P["d_raw"], float)
        prev = O / "spotr_c2l" / f"{stem(src, name, seed)}.json"
    Xc, Y = np.asarray(Xc, float), np.asarray(Y, float); tum = X["kind"] == 0; tr = tum & X["traced"]; K = X["K"]; cl = X["cl"][tr]; clone = np.where(cl < K - 1, cl, -1)
    D = np.asarray(X["D"], float); Ncl = X["Ncl"]; me = np.asarray(X["me_type"]); fT = float(tr.sum() / max(tum.sum(), 1))
    types = sorted(set(me[~tum].tolist())); Gme = np.stack([Xc[(~tum) & (me == e)].mean(0) for e in types]); gu = Xc[tum & ~tr].mean(0) if (tum & ~tr).any() else Xc[tr].mean(0)
    prof = m321.profiles(Xc[tr], clone)["within"]; onehot = np.eye(K)[cl]; has = Ncl.sum(1) > 0; nrm = lambda A: A / np.maximum(A.sum(1, keepdims=True), 1e-300)
    def score(Pm):
        r = f289.scores(Pm, Ncl); r["dominant_share"] = float(nrm(Pm[has]).max(1).mean())
        if src == "sim":
            r.update({k: v for k, v in m239.readout(None, X, Nb, m189, P_clone=Pm).items() if k in ("clone_tv", "clone_tv_win")})
        return r
    e = np.asarray(json.load(open(prev))["arms"]["notree"]["emp"], float); level = float(np.nanmedian(e[np.isfinite(e)]))
    out = json.load(open(f)) if f.exists() else dict(meta=dict(problem=name, src=src, seed=seed, traced=int(tr.sum()), spots=len(Y), level=level), arms={}, masses={})
    t0 = time.time()
    for arm in arms:
        if arm in out["arms"]:
            continue
        if arm == "c2l":
            labs = np.array([f"me{e}" for e in me]); labs[tum & ~tr] = "untraced"; labs[tr] = [f"c{c}" for c in cl]; ul = sorted(set(labs.tolist()))
            sig = np.stack([Xc[labs == u].mean(0) for u in ul]); w, info = c2l_basic(Y, sig, float(np.mean(X["ncell"])))
            Pm = np.zeros((len(Y), K)); rec = dict(info=info, types=ul)
            for k in range(K):
                if f"c{k}" in ul:
                    Pm[:, k] = w[:, ul.index(f"c{k}")]
        else:
            B = clade_matrix(arm, d, clone, K - 1); rho0 = clone_start(B, clone, K - 1, level) if norm == "exact" else None
            res = fit(Y, Gme, gu, prof, fT, B, D, level, nested=nested, norm=norm, rho0=rho0); Pm = res["phi"].T @ onehot
            rec = dict(clades=int(B.shape[1]), clade_sizes=B.sum(0).astype(int).tolist(), s2=[float(x) for x in res["s2"]], trace=res["trace"])
        s = score(Pm); out["arms"][arm] = dict(scores=s, **rec); out["masses"][arm] = Pm.tolist(); json.dump(out, open(f, "w"), default=float)
        print(f"  {name} s{seed} {arm:10s}: PCC {s['PCC']:+.3f}" + (f", subclone error {s['clone_tv']:.3f}" if "clone_tv" in s else "")
              + (f", {rec['clades']} clades" if "clades" in rec else "") + f"  [{time.time() - t0:.0f}s]", flush=True)
    out["done"] = True; json.dump(out, open(f, "w"), default=float)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for src, name, seed in PROBLEMS:
        f = OUT / f"{stem(src, name, seed)}.json"
        if f.exists() and json.load(open(f)).get("done"):
            continue
        print(f"{name} seed {seed}: start", flush=True); run_problem(src, name, seed, f); print(f"{name} seed {seed}: done ({time.time() - t0:.0f}s)", flush=True)
