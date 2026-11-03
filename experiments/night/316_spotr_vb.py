"""SpOTr as a correction of a deconvolution method's output: a posterior over the traced cells' map with a clone-cloud tree prior.  2026-10-29.
FIXED BEFORE RUNNING.
Pedro, 2026-10-29: SpOTr "should take the results of a method like that, and use the tree term to correct it, accounting for possibly
different clone-space concordance across clones ... given as a prior from which a posterior would be learned on top of the map"; then:
"Do you think you can run a few quick tests with this approach so that we can check if this is our final method?"
The model (as agreed in the conversation of 2026-10-29; star of clones, no Brownian motion, expression once, no temperature):
  inputs (trusted)    the method's traced capacity per spot c_m (DestVI: its 'traced' proportion x library; Tangram: its traced mass)
  inputs (corrected)  a score of each traced cell against each spot: DestVI (run subclone-unaware, one 'traced' label): its per-spot
                      traced state gamma_m against the cell's CondSCVI state z_i; Tangram: log of its own map
  latent              pi_i, the spot of traced cell i;  sum_i 1[pi_i = m] = n_T c_m  (posterior regularization)
  likelihood          DestVI: z_i | pi_i = m ~ N(gamma_m, tau^2 I_d), tau^2 ~ InvGamma(1, small);  Tangram: p(pi_i = m) ~ exp(b log M_im), b learned
  prior on the map    clone clouds, centres integrated out (Oskar's clade-aware GW): -sum_k [S_k / (2 n_k s2_k) + (n_k - 1) log s2_k],
                      S_k = sum_{i<j in k} D^2(pi_i, pi_j);  s2_k ~ InvGamma(a0, b0) -- the clone-space concordance, prior -> posterior
  variational         q(pi) = prod Cat(phi_i) under the capacities; q(tau^2), q(s2_k) inverse-gamma (closed-form updates)
  map step            generalised conditional gradient: Sinkhorn (entropy weight 1) on DestVI's/Tangram's cost + the clade-aware GW cost
                      (s2_k^-1 / 2 n_k) [D^2 (C_k - phi_i)], exact line search on a grid; 15 outer updates x 20 map steps
Cells without a clone (PEtracer's traced cells without a clade; the simulations' minor-clone column) are mapped by the method term only.
Arms (per problem and method): notree (no tree term); free (a0 = 1, b0 tiny: the clone spreads learned freely); prior (prior mean at the
expression-only level -- each clone's spread under the notree map, pooled -- with a0 = 10); oracle (s2_k fixed at the truth: the ceiling).
Truth spread per clone: from each spot's true clone counts, mean squared distance between two of the clone's cells / 4 (per-dimension variance).
Problems: simulation worlds W1 (clone-specific expression tree_sd 0.3) and W2 (states varying across the section, no clone expression),
seed 46 (268_.build, as v1's headline); PEtracer half M3-4, M3-1, M3-2 and serial M3-1 -> M3-2 (276_.build, seed 0).
Comparators read from saved runs: v1 at its cross-validated lambda (simulations: lineage_cv2, the true lineage tree; PEtracer: the clone
tree), the tool alone, exact OT.
Expectations stated before running:
  E151  simulations (both worlds, DestVI): the method's per-spot subclone error is at most v1's at its CV lambda + 0.02, and below DestVI
        alone, Tangram alone and its own notree arm
  E152  PEtracer (4 designs, DestVI): the method's per-clade PCC >= v1's (clone tree, CV) - 0.01 in at least 3 of 4, and above DestVI
        alone in 4 of 4
  E153  clone weights: Spearman of the learned spreads with the truth >= 0.5, and median learned / true in [0.5, 2] (no collapse), on
        M3-4 and on each simulation world
  The method arm is 'free' if it passes E153's collapse check on M3-4; otherwise 'prior'.  Read: Tangram; the oracle ceiling; tau; the maps.
Usage: python 316_spotr_vb.py [SUMMARY]    (writes spotr_out/spotr_vb/<problem>.json; resumable per problem)
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / "spotr_out" / "spotr_vb"
PROBLEMS = [("sim", "W1", 46), ("sim", "W2", 46), ("petracer", "half:M3-4", 0), ("petracer", "half:M3-1", 0), ("petracer", "half:M3-2", 0), ("petracer", "serial:M3-1>M3-2", 0)]
ARMS, N_OUTER, J, IT0, IT1 = ["notree", "free", "prior", "oracle"], 15, 20, 500, 150
ETAS = [2.0 ** -k for k in range(11)] + [0.0]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def fit_destvi_states(Xc, labels, Y, seed=0, epochs_sc=300, epochs_st=1500):
    """250_'s fit_destvi, also returning the cells' CondSCVI states z and DestVI's per-spot states gamma per label (same latent space)"""
    import anndata, scvi, pandas as pd
    scvi.settings.seed = seed
    sc = anndata.AnnData(np.asarray(Xc, np.float32)); sc.obs["label"] = pd.Categorical(labels)
    scvi.model.CondSCVI.setup_anndata(sc, labels_key="label"); msc = scvi.model.CondSCVI(sc, weight_obs=False); msc.train(max_epochs=epochs_sc, enable_progress_bar=False)
    st = anndata.AnnData(np.asarray(Y, np.float32)); scvi.model.DestVI.setup_anndata(st); mst = scvi.model.DestVI.from_rna_model(st, msc); mst.train(max_epochs=epochs_st, enable_progress_bar=False)
    names = list(msc.adata_manager.get_state_registry("labels").categorical_mapping)
    own = mst.get_proportions(); own = own[names].values if hasattr(own, "values") else own
    gam = mst.get_gamma(return_numpy=False)
    return dict(names=names, own=np.asarray(own, float), z=np.asarray(msc.get_latent_representation(), float),
                gamma={k: np.asarray(v, float) for k, v in gam.items()}, library=np.asarray(Y, float).sum(1))


def truth_spread(Ncl, D, K):
    """per clone: mean squared distance between two of its cells (from spot counts) / 4"""
    D2 = D * D; out = np.full(K, np.nan)
    for k in range(K):
        n = Ncl[:, k]; tot = n.sum()
        if tot >= 2:
            out[k] = float(n @ D2 @ n / (tot * (tot - 1)) / 4.0)
    return out


def vb(kind, base, caps, clone, Kc, D, arm, s2_truth=None, level=None, phi0=None):
    """kind 'gauss' (base = squared state distances, d = base dims recorded separately) or 'logp' (base = log map); caps sum to 1"""
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True); from jax.scipy.special import logsumexp
    n, M = base.shape; K = Kc
    on = clone >= 0; B = np.zeros((n, max(K, 1))); B[np.where(on)[0], clone[on]] = 1.0; nk = B.sum(0)
    D2 = jnp.asarray(D * D); Bj = jnp.asarray(B); base_j = jnp.asarray(base); logr = jnp.zeros(n); logc = jnp.log(jnp.asarray(caps) * n)
    def sink(Cst, f, g, iters):
        def body(_, s):
            f, g = s; f = logr - logsumexp(g[None, :] - Cst, axis=1); g = logc - logsumexp(f[:, None] - Cst, axis=0); return (f, g)
        return jax.lax.fori_loop(0, iters, body, (f, g))
    def quad(phi, wk):                                  # sum_k wk/(2 n_k) S_k(phi), S_k = 1/2 (C_k D2 C_k - sum_i phi_i D2 phi_i)
        C = Bj.T @ phi; PD = phi @ D2; self_ = (PD * phi).sum(1); Sk = 0.5 * ((C @ D2) * C).sum(1) - 0.5 * (Bj.T @ self_)
        return jnp.sum(wk / (2.0 * jnp.maximum(jnp.asarray(nk), 1.0)) * Sk), Sk
    def tree_cost(phi, wk):
        C = Bj.T @ phi; coef = Bj @ (wk / (2.0 * jnp.maximum(jnp.asarray(nk), 1.0))); return coef[:, None] * (Bj @ (C @ D2) - phi @ D2)
    def obj(phi, cb, wk):
        return jnp.sum(phi * cb) + quad(phi, wk)[0] + jnp.sum(phi * jnp.log(jnp.maximum(phi, 1e-300)))
    @jax.jit
    def gcg(phi, f, g, cb, wk):
        def step(c, _):
            phi, f, g = c; Cst = cb + tree_cost(phi, wk); f, g = sink(Cst, f, g, IT1); ph = jnp.exp(f[:, None] + g[None, :] - Cst)
            vals = jax.lax.map(lambda e: obj(phi + e * (ph - phi), cb, wk), jnp.asarray(ETAS)); e = jnp.asarray(ETAS)[jnp.argmin(vals)]
            return (phi + e * (ph - phi), f, g), vals.min()
        (phi, f, g), tr = jax.lax.scan(step, (phi, f, g), None, length=J); return phi, f, g, tr
    use_tree = arm != "notree" and K > 0
    if use_tree:
        if arm == "free":
            a0, b0 = np.full(K, 1.0), np.full(K, 1e-3 * float(np.median(D[D > 0])) ** 2)
        elif arm == "prior":
            a0 = np.full(K, 10.0); b0 = level * (a0 - 1.0)              # prior mean b0 / (a0 - 1) = the expression-only level
        s2 = np.where(np.isfinite(s2_truth), s2_truth, np.nanmedian(s2_truth)) if arm == "oracle" else np.full(K, level)
    else:
        s2 = np.full(max(K, 1), np.inf)
    f, g = jnp.zeros(n), jnp.zeros(M); trace = []
    w_tau = 1.0 / max(float(np.median(base)) / max(GAUSS_D[0], 1), 1e-300) if kind == "gauss" else None; beta = 1.0
    phi = jnp.asarray(phi0) if phi0 is not None else None
    for it in range(N_OUTER):
        cb = (0.5 * w_tau * base_j) if kind == "gauss" else (-beta * base_j)
        wk = jnp.asarray(1.0 / s2 if use_tree else np.zeros(max(K, 1)))
        if it == 0:                                          # potentials warmed on the first cost; start from phi0 (the notree map) if given
            f, g = sink(cb, f, g, IT0); phi = phi if phi is not None else jnp.exp(f[:, None] + g[None, :] - cb)
        phi, f, g, tr = gcg(phi, f, g, cb, wk); P = np.asarray(phi)
        if kind == "gauss":                                  # q(tau^2) = InvGamma(1 + n d / 2, small + 1/2 sum phi R2)
            a_t = 1.0 + n * GAUSS_D[0] / 2.0; b_t = 1e-6 + 0.5 * float((P * base).sum()); w_tau = a_t / b_t
        else:                                                # beta: 1-D maximisation of sum phi beta s - sum_i lse(beta s_i) (concave)
            grid = np.exp(np.linspace(np.log(0.05), np.log(50.0), 120)); ll = [float((P * base).sum() * b - np.sum(np.logaddexp.reduce(b * base, axis=1))) for b in grid]
            beta = float(grid[int(np.argmax(ll))])
        _, Sk = quad(phi, jnp.ones(max(K, 1))); Sk = np.asarray(Sk)
        if use_tree and arm in ("free", "prior"):
            a_k = a0 + np.maximum(nk[:K] - 1, 0); b_k = b0 + Sk[:K] / (2.0 * np.maximum(nk[:K], 1.0)); s2 = b_k / np.maximum(a_k - 1.0, 1e-9)   # posterior mean
        emp = Sk[:K] / np.maximum(nk[:K] * (nk[:K] - 1) / 2.0, 1.0) / 4.0 if K else np.zeros(0)   # the map's own clone spread (as the truth's)
        trace.append(dict(obj=float(tr[-1]), s2=[float(x) for x in (s2[:K] if use_tree else emp)], emp=[float(x) for x in emp], w_tau=w_tau, beta=beta))
    return dict(phi=np.asarray(phi), s2=s2[:K] if use_tree else np.full(K, np.inf), emp=emp, trace=trace, w_tau=w_tau, beta=beta)


GAUSS_D = [10]


def run_problem(src, name, seed, f):
    import warnings; warnings.filterwarnings("ignore")
    core, f289 = mod("core", "250_spotr_core.py"), mod("f289", "289_field_metrics.py")
    if src == "sim":
        m268, m239, m189 = mod("m268", "268_lineage_cv.py"), mod("m239", "239_deconv_fgw.py"), mod("m189", "189_me_sim.py")
        W = mod("m270", "270_lineage_cv_concordance.py").WORLDS[name]; NB, X, Xc, t_f, _ = m268.build(seed, **W); Y = X["Sc"]; Nb = m239.window(X)
    else:
        m276 = mod("m276", "276_petracer.py"); P = m276.build(name, seed); X, Xc, Y = P["X"], P["Xc"], P["Y"]; Nb = m239 = m189 = None
    tum = X["kind"] == 0; tr = tum & X["traced"]; K = X["K"]; cl = X["cl"][tr]; clone = np.where(cl < K - 1, cl, -1); D = np.asarray(X["D"], float); Ncl = X["Ncl"]
    labels = np.array(["traced" if t else ("untraced" if m else f"me{e}") for t, m, e in zip(tr, tum, X["me_type"])])
    out = json.load(open(f)) if f.exists() else dict(meta=dict(problem=name, src=src, seed=seed, traced=int(tr.sum()), spots=len(Y), K=int(K)), arms={}, masses={})
    t0 = time.time(); fit = fit_destvi_states(Xc, labels, Y, seed=0); q = fit["library"] / fit["library"].sum(); Tt = core.tangram_own(Xc, Y, q, seed=0)
    s2_truth = truth_spread(Ncl, D, K)[: K - 1]; out["meta"]["s2_truth"] = s2_truth.tolist(); out["meta"]["fit_seconds"] = time.time() - t0
    onehot = np.eye(K)[cl]; ti = np.where(tr)[0]
    z = fit["z"][ti]; gm = fit["gamma"]["traced"]; GAUSS_D[0] = z.shape[1]; R2 = np.maximum((z * z).sum(1)[:, None] + (gm * gm).sum(1)[None, :] - 2.0 * z @ gm.T, 0.0)
    own_t = fit["own"][:, fit["names"].index("traced")]; caps_d = own_t * fit["library"]; caps_d = caps_d / caps_d.sum()
    LS = np.log(np.maximum(Tt[ti], 1e-12)); caps_t = Tt[ti].sum(0); caps_t = caps_t / caps_t.sum()
    def score(Pm):
        r = f289.scores(Pm, Ncl)
        if src == "sim":
            r.update({k: v for k, v in m239.readout(None, X, Nb, m189, P_clone=Pm).items() if k in ("clone_tv", "clone_tv_win")})
        return r
    for tool, kind, base, caps in [("destvi", "gauss", R2, caps_d), ("tangram", "logp", LS, caps_t)]:
        level = None; phi0 = None
        if f"{tool}|notree" in out["arms"]:                    # resumed: the expression-only level from the saved notree arm
            e = np.asarray(out["arms"][f"{tool}|notree"]["emp"], float); level = float(np.nanmedian(e[np.isfinite(e)]))
        for arm in ARMS:
            key = f"{tool}|{arm}"
            if key in out["arms"]:
                continue
            res = vb(kind, base, caps, clone, K - 1, D, arm, s2_truth=s2_truth, level=level, phi0=phi0)
            if arm == "notree":
                emp = res["emp"]; level = float(np.nanmedian(emp[np.isfinite(emp)])); phi0 = res["phi"]
            Pm = res["phi"].T @ onehot; s = score(Pm); ok = np.isfinite(s2_truth) & np.isfinite(res["s2"])
            from scipy.stats import spearmanr
            s.update(spread_spearman=float(spearmanr(res["s2"][ok], s2_truth[ok])[0]) if ok.sum() > 2 else None,
                     spread_ratio_median=float(np.median(res["s2"][ok] / s2_truth[ok])) if ok.sum() else None, level=level)
            out["arms"][key] = dict(scores=s, s2=[float(x) for x in res["s2"]], emp=[float(x) for x in res["emp"]], trace=res["trace"], w_tau=res["w_tau"], beta=res["beta"])
            out["masses"][key] = Pm.tolist(); json.dump(out, open(f, "w"), default=float)
            print(f"  {name} {tool:8s} {arm:7s}: PCC {s['PCC']:+.3f}" + (f", subclone error {s['clone_tv']:.3f}" if "clone_tv" in s else "")
                  + (f", spreads vs truth Spearman {s['spread_spearman']:+.2f}, median ratio {s['spread_ratio_median']:.2f}" if s["spread_spearman"] is not None else "")
                  + f"  [{time.time() - t0:.0f}s]", flush=True)
    out["done"] = True; json.dump(out, open(f, "w"), default=float)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for src, name, seed in PROBLEMS:
        f = OUT / f"{src}_{name.replace(':', '_').replace('>', '_to_')}_{seed}.json"
        if f.exists() and json.load(open(f)).get("done"):
            continue
        print(f"{name}: start", flush=True); run_problem(src, name, seed, f); print(f"{name}: done ({time.time() - t0:.0f}s total)", flush=True)
