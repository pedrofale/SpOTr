"""cell2location with the traced cell types' abundances from a map of the traced cells under the clade-aware GW prior.  2026-10-29.
FIXED BEFORE RUNNING.
Pedro, 2026-10-29: "I want to extend cell2location, but with my GW OT to compute the abundances for the traced cell types ... and the
complete deconvolution uses that internally"; "I quite like it"; "I agree with your suggestion of using each cell's own profile".
Model (exp-unmatched-oct12 section 117; method-spotr-correction, the cell2location form):
  spots      y_sg ~ NB(mu_sg, r_g),  mu_sg = ( m_g [ sum_e w_se g_eg + w_s,tum ( (1 - f_T) g_ug + f_T h_sg(pi) ) ] + s_g ) y_s
             w_s = N_s pi_s (cells per spot x composition over the microenvironment types e and the tumour);  g_eg, g_ug: mean counts
             per cell of each type and of the untraced tumour cells in the dissociated sample (a first approximation to cell2location's
             step-1 signatures);  f_T: the dissociated sample's traced fraction of the tumour (random tracing, as v1)
  traced     h_sg(pi) = mean of the profiles xhat_n of the traced cells mapped to spot s, with sum_n 1[pi_n = s] = n_T w_s,tum / sum w_tum;
             xhat_n = each traced cell's own profile, its 15 nearest traced neighbours' mean counts (PCA of log-normalised counts)
  map prior  the clade-aware GW: clone clouds, centres integrated out, -sum_k [S_k / (2 n_k s2_k) + (n_k - 1) log s2_k], s2_k ~ InvGamma
  priors     N_s ~ Gamma(10, 1); pi_s flat (MAP); log m_g ~ N(0, 1); y_s ~ Gamma(20, 20); log s_g ~ N(log(1e-3 mean count), 2); log r_g ~ N(log 10, 2)
Inference: cell2location's continuous parameters by MAP (Adam), the map by variational q(pi) = prod Cat(phi_n) (Sinkhorn under the
tumour capacities, entropy weight 1, the GW linearised: conditional gradient), q(s2_k) inverse-gamma closed form; alternated
(1,500 Adam steps to start, then 10 outer rounds of 20 map steps and 300 Adam steps).
Arms: notree; prior (spread prior mean at the notree map's level, a0 = 10); free (a0 = 1, b0 tiny); clonesig (the prior arm with each
traced cell's profile replaced by its clone's mean -- the control for orientation).  The method arm: free if its spreads pass the
collapse check on M3-4 (median learned / true in [0.5, 2]), else prior (the rule of section 116).
Problems: as 316_ (simulation W1, W2 seed 46; PEtracer half M3-4, M3-1, M3-2, serial).
Expectations stated before running:
  E154  PEtracer: the method's per-clade PCC >= max(v1 at its CV lambda, 316_'s prior arm) - 0.01 in at least 3 of 4 designs
  E155  sharpness (Pedro: "on the PEtracer figure it looks so bad"): on M3-4 the method's mean dominant clade share per spot lies within
        [0.45, 0.70] (the truth: 0.69 per spot, 0.56 over 3 x 3 windows; 316_ 0.42, v1 0.65)
  E156  simulations: the method's per-spot subclone error <= v1's at its CV lambda + 0.02 in both worlds
  E157  orientation: cell profiles beat clone signatures (method arm against clonesig) on per-clade PCC in at least 3 of 4 PEtracer designs
  read  spreads against the truth; SSIM, JS, windowed errors, centroid errors; the per-clade maps (318_'s layout)
Usage: python 319_spotr_c2l.py    (writes spotr_out/spotr_c2l/<problem>.json; resumable per problem)
"""
import os, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / "spotr_out" / "spotr_c2l"
PROBLEMS = [("sim", "W1", 46), ("sim", "W2", 46), ("petracer", "half:M3-4", 0), ("petracer", "half:M3-1", 0), ("petracer", "half:M3-2", 0), ("petracer", "serial:M3-1>M3-2", 0)]
ARMS, N_OUTER, J, IT0, IT1, ADAM0, ADAM1, KNN = ["notree", "prior", "free", "clonesig"], 10, 20, 500, 150, 1500, 300, 15
ETAS = [2.0 ** -k for k in range(11)] + [0.0]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def knn_profiles(Xt, k=KNN):
    """each traced cell's own profile: the mean counts of its k nearest traced cells (itself included) in a 30-PC space of log CP10k"""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    L = np.log1p(Xt / np.maximum(Xt.sum(1, keepdims=True), 1.0) * 1e4); Z = PCA(n_components=min(30, L.shape[1] - 1), random_state=0).fit_transform(L - L.mean(0))
    idx = NearestNeighbors(n_neighbors=min(k, len(Xt))).fit(Z).kneighbors(Z, return_distance=False); return Xt[idx].mean(1)


def fit(Y, Gme, gu, Xh, fT, clone, Kc, D, arm, level=None, s2_truth=None):
    import jax, jax.numpy as jnp, optax
    jax.config.update("jax_enable_x64", True); from jax.scipy.special import logsumexp, gammaln
    S, G = Y.shape; E = Gme.shape[0]; n = Xh.shape[0]; Yj, Gj, guj, Xj, D2 = map(jnp.asarray, (Y, Gme, gu, Xh, D * D))
    on = clone >= 0; B = np.zeros((n, max(Kc, 1))); B[np.where(on)[0], clone[on]] = 1.0; nk = B.sum(0); Bj = jnp.asarray(B); nkj = jnp.asarray(np.maximum(nk, 1.0))
    mc = float(Y.mean())
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
    from functools import partial
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
        C = Bj.T @ phi; coef = Bj @ (wk / (2.0 * nkj)); return coef[:, None] * (Bj @ (C @ D2) - phi @ D2)
    def F(phi, th, wk):
        return -loglik(th, phi) + quad(phi, wk)[0] + jnp.sum(phi * jnp.log(jnp.maximum(phi, 1e-300)))
    @jax.jit
    def gcg(phi, f, g, th, wk, logc):
        def step(c, _):
            phi, f, g = c; Cst = -jax.grad(lambda p: loglik(th, p))(phi) + tree_cost(phi, wk); f, g = sink(Cst, logc, f, g, IT1); ph = jnp.exp(f[:, None] + g[None, :] - Cst)
            vals = jax.lax.map(lambda e: F(phi + e * (ph - phi), th, wk), jnp.asarray(ETAS)); e = jnp.asarray(ETAS)[jnp.argmin(vals)]
            return (phi + e * (ph - phi), f, g), vals.min()
        (phi, f, g), tr = jax.lax.scan(step, (phi, f, g), None, length=J); return phi, f, g, tr
    use_tree = arm != "notree" and Kc > 0
    if use_tree:
        a0, b0 = (np.full(Kc, 1.0), np.full(Kc, 1e-3 * float(np.median(D[D > 0])) ** 2)) if arm == "free" else (np.full(Kc, 10.0), np.full(Kc, 9.0 * level))
        s2 = np.full(Kc, level)
    else:
        s2 = np.full(max(Kc, 1), np.inf)
    logc = jnp.log(caps_of(th)); f, g = jnp.zeros(n), jnp.zeros(S); f, g = sink(jnp.zeros((n, S)), logc, f, g, IT0); phi = jnp.exp(f[:, None] + g[None, :])
    th, st = adam(th, st, phi, ADAM0); trace = []
    for it in range(N_OUTER):
        logc = jnp.log(caps_of(th)); f, g = sink(-jnp.log(jnp.maximum(phi, 1e-300)), logc, f, g, IT0); phi = jnp.exp(f[:, None] + g[None, :] + jnp.log(jnp.maximum(phi, 1e-300)))
        wk = jnp.asarray(1.0 / s2 if use_tree else np.zeros(max(Kc, 1))); phi, f, g, tr = gcg(phi, f, g, th, wk, logc)
        th, st = adam(th, st, phi, ADAM1); _, Sk = quad(phi, jnp.ones(max(Kc, 1))); Sk = np.asarray(Sk)[:Kc]
        if use_tree and arm in ("free", "prior", "clonesig"):
            s2 = (b0 + Sk / (2.0 * np.maximum(nk[:Kc], 1.0))) / np.maximum(a0 + np.maximum(nk[:Kc] - 1, 0) - 1.0, 1e-9)
        emp = Sk / np.maximum(nk[:Kc] * (nk[:Kc] - 1) / 2.0, 1.0) / 4.0
        trace.append(dict(loglik=float(loglik(th, phi)), map_obj=float(tr[-1]), s2=[float(x) for x in (s2[:Kc] if use_tree else emp)]))
    return dict(phi=np.asarray(phi), s2=s2[:Kc] if use_tree else np.full(Kc, np.inf), emp=emp, trace=trace)


def run_problem(src, name, seed, f):
    import warnings; warnings.filterwarnings("ignore")
    from scipy.stats import spearmanr
    f289, m316 = mod("f289", "289_field_metrics.py"), mod("m316", "316_spotr_vb.py")
    if src == "sim":
        m268, m239, m189 = mod("m268", "268_lineage_cv.py"), mod("m239", "239_deconv_fgw.py"), mod("m189", "189_me_sim.py")
        W = mod("m270", "270_lineage_cv_concordance.py").WORLDS[name]; NB, X, Xc, t_f, _ = m268.build(seed, **W); Y = X["Sc"]; Nb = m239.window(X)
    else:
        m276 = mod("m276", "276_petracer.py"); P = m276.build(name, seed); X, Xc, Y = P["X"], P["Xc"], P["Y"]; Nb = m239 = m189 = None
    Xc, Y = np.asarray(Xc, float), np.asarray(Y, float); tum = X["kind"] == 0; tr = tum & X["traced"]; K = X["K"]; cl = X["cl"][tr]; clone = np.where(cl < K - 1, cl, -1)
    D = np.asarray(X["D"], float); Ncl = X["Ncl"]; me = np.asarray(X["me_type"]); fT = float(tr.sum() / max(tum.sum(), 1))
    types = sorted(set(me[~tum].tolist())); Gme = np.stack([Xc[(~tum) & (me == e)].mean(0) for e in types]); gu = Xc[tum & ~tr].mean(0) if (tum & ~tr).any() else Xc[tr].mean(0)
    Xt = Xc[tr]; Xh = knn_profiles(Xt); Xsig = Xh.copy()
    for k in np.unique(clone[clone >= 0]):
        Xsig[clone == k] = Xt[clone == k].mean(0)
    s2_truth = m316.truth_spread(Ncl, D, K)[: K - 1]; onehot = np.eye(K)[cl]
    out = json.load(open(f)) if f.exists() else dict(meta=dict(problem=name, src=src, seed=seed, traced=int(tr.sum()), spots=len(Y), genes=Y.shape[1], fT=fT, s2_truth=s2_truth.tolist()), arms={}, masses={})
    def score(Pm):
        r = f289.scores(Pm, Ncl); nrm = lambda A: A / np.maximum(A.sum(1, keepdims=True), 1e-300); has = Ncl.sum(1) > 0; r["dominant_share"] = float(nrm(Pm[has]).max(1).mean())
        if src == "sim":
            r.update({k: v for k, v in m239.readout(None, X, Nb, m189, P_clone=Pm).items() if k in ("clone_tv", "clone_tv_win")})
        return r
    t0 = time.time(); level = None
    if "notree" in out["arms"]:
        e = np.asarray(out["arms"]["notree"]["emp"], float); level = float(np.nanmedian(e[np.isfinite(e)]))
    for arm in ARMS:
        if arm in out["arms"]:
            continue
        res = fit(Y, Gme, gu, Xsig if arm == "clonesig" else Xh, fT, clone, K - 1, D, "prior" if arm == "clonesig" else arm, level=level, s2_truth=s2_truth)
        if arm == "notree":
            e = res["emp"]; level = float(np.nanmedian(e[np.isfinite(e)]))
        Pm = res["phi"].T @ onehot; s = score(Pm); ok = np.isfinite(s2_truth) & np.isfinite(res["s2"])
        s.update(spread_spearman=float(spearmanr(res["s2"][ok], s2_truth[ok])[0]) if ok.sum() > 2 else None, spread_ratio_median=float(np.median(res["s2"][ok] / s2_truth[ok])) if ok.sum() else None, level=level)
        out["arms"][arm] = dict(scores=s, s2=[float(x) for x in res["s2"]], emp=[float(x) for x in res["emp"]], trace=res["trace"]); out["masses"][arm] = Pm.tolist(); json.dump(out, open(f, "w"), default=float)
        print(f"  {name} {arm:8s}: PCC {s['PCC']:+.3f}, dominant share {s['dominant_share']:.2f}" + (f", subclone error {s['clone_tv']:.3f}" if "clone_tv" in s else "")
              + (f", spreads Spearman {s['spread_spearman']:+.2f}, ratio {s['spread_ratio_median']:.2f}" if s.get("spread_spearman") is not None else "") + f"  [{time.time() - t0:.0f}s]", flush=True)
    out["done"] = True; json.dump(out, open(f, "w"), default=float)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for src, name, seed in PROBLEMS:
        f = OUT / f"{src}_{name.replace(':', '_').replace('>', '_to_')}_{seed}.json"
        if f.exists() and json.load(open(f)).get("done"):
            continue
        print(f"{name}: start", flush=True); run_problem(src, name, seed, f); print(f"{name}: done ({time.time() - t0:.0f}s total)", flush=True)
