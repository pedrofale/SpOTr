"""The new method on the data without spatial ground truth: B189 (core) and the embryo (seqFISH E8.5, MOSTA E9.5).  2026-10-29.
FIXED BEFORE RUNNING.
Method as selected on 2026-10-29 (319_/321_: cell2location + the GW-regularised traced map, within-clone profiles, free clone spreads),
generalised for this data:
  several traced kinds  each cell type that holds traced cells is a traced kind with its own capacity: its traced cells are mapped within
                        it (sum_{n in c} pi_n = s ~ n_c w_sc), its untraced cells enter as the kind's mean profile, split by the kind's
                        traced fraction f_c -- the embryo's clones span many cell types; B189 has one traced kind (tumour)
  scale                 float32, the 500 fitted genes most variable in the dissociated cells, no self-pair correction in the GW term
                        (an O(1 / n_k) term), 8 outer rounds of 12 map steps, a 6-point line search
Arms: notree, within_free.  Problems: b189_core_CvAB, b189_core_ABC (304_), seqfish, e95 (the embryo npz of 279_).
Readouts (no spatial truth): held-out genes (Spearman across spots of the spots' measured expression per total against the map's
prediction, as 279_; EGFR for B189, Meox1 and Ankrd1 for the embryo), the clone spreads, clone maps.
Expectations stated before running:
  E162  the tree costs no held-out expression: the method's held-out Spearman >= its notree arm's - 0.01 in all 4 problems
  E163  the method's held-out Spearman >= Tangram alone's (279_ / 305_) in >= 3 of 4
  read  B189: C's share against the spots' chromosome 2 loss (310_'s residual), and whether C is still sent to the section's edge
        (section 116 of the vault note's B189 entry); the embryo: clone maps and spreads (early clones expected to disperse)
Usage: python 324_c2l_general.py    (writes spotr_out/spotr_c2l_real/<problem>.json)
"""
import os, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / "spotr_out" / "spotr_c2l_real"; DATA = HERE.parents[1] / "data"
PROBLEMS = [("b189_core_CvAB", DATA / "b189" / "b189_core_CvAB.npz"), ("b189_core_ABC", DATA / "b189" / "b189_core_ABC.npz"),
            ("seqfish", DATA / "embryo" / "embryo2a_seqfish.npz"), ("e95", DATA / "embryo" / "embryo2a_e95.npz")]
N_OUTER, J, IT0, IT1, ADAM0, ADAM1, G_MAX, MIN_TYPE = 8, 12, 300, 100, 1000, 200, 500, 30
ETAS = [1.0, 0.5, 0.25, 0.1, 0.03, 0.0]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def fit(Y, Gme, gu, fT, Xh, kc, clone, Kc, D, arm, level=None):
    import jax, jax.numpy as jnp, optax
    from functools import partial
    from jax.scipy.special import logsumexp, gammaln
    S, G = Y.shape; E = Gme.shape[0]; T = gu.shape[0]; n = Xh.shape[0]; rows = [np.where(kc == c)[0] for c in range(T)]; nc = np.array([len(r) for r in rows], float)
    f32 = lambda a: jnp.asarray(np.asarray(a, np.float32))
    Yj, Gj, guj, fTj, Xj, D2, kcj = f32(Y), f32(Gme if E else np.zeros((1, G))), f32(gu), f32(fT), f32(Xh), f32(D * D), jnp.asarray(kc)
    on = clone >= 0; B = np.zeros((n, max(Kc, 1)), np.float32); B[np.where(on)[0], clone[on]] = 1.0; nk = B.sum(0); Bj = jnp.asarray(B); nkj = f32(np.maximum(nk, 1.0)); mc = float(Y.mean())
    th = dict(logN=f32(np.full(S, np.log(10.0))), z=f32(np.zeros((S, E + T))), logm=f32(np.zeros(G)), logy=f32(np.zeros(S)), logs=f32(np.full(G, np.log(1e-3 * mc))), logr=f32(np.full(G, np.log(10.0))))
    def parts(th):
        w = jnp.exp(th["logN"])[:, None] * jax.nn.softmax(th["z"], axis=1); return w[:, :E], w[:, E:]
    def caps_of(th):
        wt = parts(th)[1]; return (wt / wt.sum(0, keepdims=True) * f32(nc)[None, :]).T              # T x S
    def mu_of(th, phi, caps):
        we, wt = parts(th); A = (wt * fTj[None, :] / jnp.maximum(caps.T, 1e-12))[:, kcj].T            # n x S: each cell's weight in its spot
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
    def sink(Cst, logcap, f, g, iters):                                                           # per-kind column constraints
        def body(_, s):
            f, g = s; f = -logsumexp(g[kcj] - Cst, axis=1)
            g = jnp.stack([logcap[c] - logsumexp(f[rows[c], None] - Cst[rows[c]], axis=0) for c in range(T)]); return (f, g)
        return jax.lax.fori_loop(0, iters, body, (f, g))
    def quad(phi, wk):
        C = Bj.T @ phi; Sk = 0.5 * ((C @ D2) * C).sum(1); return jnp.sum(wk / (2.0 * nkj) * Sk), Sk
    def tree_cost(phi, wk):
        C = Bj.T @ phi; return (Bj @ (wk / (2.0 * nkj)))[:, None] * (Bj @ (C @ D2))
    def F(phi, th, wk, caps):
        return -loglik(th, phi, caps) + quad(phi, wk)[0] + jnp.sum(phi * jnp.log(jnp.maximum(phi, 1e-30)))
    @jax.jit
    def gcg(phi, f, g, th, wk, caps):
        logcap = jnp.log(jnp.maximum(caps, 1e-30))
        def step(c, _):
            phi, f, g = c; Cst = -jax.grad(lambda p: loglik(th, p, caps))(phi) + tree_cost(phi, wk); f, g = sink(Cst, logcap, f, g, IT1); ph = jnp.exp(f[:, None] + g[kcj] - Cst)
            vals = jnp.stack([F(phi + e * (ph - phi), th, wk, caps) for e in ETAS]); e = jnp.asarray(ETAS)[jnp.argmin(vals)]
            return (phi + e * (ph - phi), f, g), vals.min()
        (phi, f, g), tr = jax.lax.scan(step, (phi, f, g), None, length=J); return phi, f, g, tr
    use_tree = arm != "notree" and Kc > 0
    a0, b0 = np.full(Kc, 1.0), np.full(Kc, 1e-3 * float(np.median(D[D > 0])) ** 2)
    s2 = np.full(Kc, level if level else float(np.median(D[D > 0])) ** 2 / 16.0) if use_tree else np.full(max(Kc, 1), np.inf)
    caps = caps_of(th); f, g = jnp.zeros(n, jnp.float32), jnp.zeros((T, S), jnp.float32)
    f, g = sink(jnp.zeros((n, S), jnp.float32), jnp.log(caps), f, g, IT0); phi = jnp.exp(f[:, None] + g[kcj]); th, st = adam(th, st, phi, caps, ADAM0); trace = []
    for it in range(N_OUTER):
        caps = caps_of(th); lphi = jnp.log(jnp.maximum(phi, 1e-30)); f, g = sink(-lphi, jnp.log(caps), f, g, IT0); phi = jnp.exp(f[:, None] + g[kcj] + lphi)
        wk = f32(1.0 / s2 if use_tree else np.zeros(max(Kc, 1))); phi, f, g, tr = gcg(phi, f, g, th, wk, caps); th, st = adam(th, st, phi, caps, ADAM1)
        Sk = np.asarray(quad(phi, f32(np.ones(max(Kc, 1))))[1], float)[:Kc]
        if use_tree and not arm.startswith("fixed"):                                                # 337_: 'fixed*' arms keep the spreads they start at
            s2 = (b0 + Sk / (2.0 * np.maximum(nk[:Kc], 1.0))) / np.maximum(a0 + np.maximum(nk[:Kc] - 1, 0) - 1.0, 1e-9)
        emp = Sk / np.maximum(nk[:Kc] ** 2 / 2.0, 1.0) / 4.0
        trace.append(dict(loglik=float(loglik(th, phi, caps)), s2=[float(x) for x in (s2[:Kc] if use_tree else emp)]))
        print(f"      outer {it}: loglik {trace[-1]['loglik']:.4g}", flush=True)
    we, wt = parts(th); obj = None
    if use_tree:                                                                                  # the joint objective (327_): comparable across fixed points of one problem
        ll, lp, ent = float(loglik(th, phi, caps)), float(logprior(th)), float(-jnp.sum(phi * jnp.log(jnp.maximum(phi, 1e-30))))
        qd = float(quad(phi, f32(1.0 / s2))[0]); ls2 = np.log(s2[:Kc]); sp = float(-np.sum(np.maximum(nk[:Kc] - 1, 0) * ls2) - np.sum((a0 + 1.0) * ls2 + b0 / s2[:Kc]))
        obj = dict(total=ll + lp - qd + sp + ent, loglik=ll, logprior=lp, tree=-qd, spreads=sp, entropy=ent)
    return dict(phi=np.asarray(phi, float), th={k: np.asarray(v, float) for k, v in th.items()}, we=np.asarray(we, float), wt=np.asarray(wt, float), caps=np.asarray(caps, float),
                s2=s2[:Kc] if use_tree else np.full(Kc, np.inf), emp=emp, trace=trace, objective=obj)


def run(name, path, f, arms=("notree", "within_free"), levels=None):
    import warnings; warnings.filterwarnings("ignore")
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import rankdata
    m321 = mod("m321", "321_c2l_profiles.py"); z = dict(np.load(path, allow_pickle=True))
    Xc, Y = z["Xc_fit"].astype(float), z["Ys_fit"].astype(float); ct = np.asarray(z["cell_type"]).astype(str); u, cnt = np.unique(ct, return_counts=True); ct = np.where(np.isin(ct, u[cnt < MIN_TYPE]), "other", ct)
    inform = np.asarray(z["clone_informative"], bool); clone_all = z["clone"].astype(int); traced = inform[clone_all]; remap = -np.ones(len(inform), int); remap[inform] = np.arange(inform.sum()); Kc = int(inform.sum())
    L = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1.0) * 1e4); gsel = np.argsort(-L.var(0))[: min(G_MAX, Xc.shape[1])]; Xg, Yg = Xc[:, gsel], Y[:, gsel]
    kinds = sorted(set(ct.tolist())); tk = [k for k in kinds if (traced & (ct == k)).any()]; ek = [k for k in kinds if k not in tk]
    Gme = np.stack([Xg[ct == k].mean(0) for k in ek]) if ek else np.zeros((0, Xg.shape[1]))
    gu = np.stack([Xg[(ct == k) & ~traced].mean(0) if ((ct == k) & ~traced).any() else Xg[(ct == k) & traced].mean(0) for k in tk]); fT = np.array([((ct == k) & traced).sum() / (ct == k).sum() for k in tk])
    ti = np.where(traced)[0]; kc = np.array([tk.index(k) for k in ct[ti]]); clone = remap[clone_all[ti]]
    prof = m321.profiles(Xg[ti], clone)["within"]; D = squareform(pdist(np.asarray(z["xy"], float)))
    lib = z["lib"].astype(float); Cev = z["Xc_ev"].astype(float) / lib[:, None]; Ev = z["Ys_ev"].astype(float) / np.maximum(z["spot_total"].astype(float), 1.0)[:, None]; ev = [str(g) for g in z["ev_genes"]]
    Pev = m321.profiles(z["Xc_ev"].astype(float)[ti], clone)["within"] / lib[ti][:, None]                  # held-out genes, the same within-clone smoothing
    gev_e = np.stack([Cev[ct == k].mean(0) for k in ek]) if ek else np.zeros((0, len(ev))); gev_u = np.stack([Cev[(ct == k) & ~traced].mean(0) if ((ct == k) & ~traced).any() else Cev[(ct == k) & traced].mean(0) for k in tk])
    def heldout(res):
        we, wt, caps, phi = res["we"], res["wt"], res["caps"], res["phi"]; A = (wt * fT[None, :] / np.maximum(caps.T, 1e-12))[:, kc].T
        pred = (we @ gev_e if len(ek) else 0.0) + (wt * (1.0 - fT)[None, :]) @ gev_u + (phi * A).T @ Pev
        ra, rb = np.apply_along_axis(rankdata, 0, pred), np.apply_along_axis(rankdata, 0, Ev); ra -= ra.mean(0); rb -= rb.mean(0)
        r = (ra * rb).sum(0) / np.maximum(np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0)), 1e-300)
        return dict(heldout_spearman=float(np.nanmean(r)), **{f"spearman_{g}": float(r[ev.index(g)]) for g in ["EGFR", "Meox1", "Ankrd1"] if g in ev})
    out = json.load(open(f)) if f.exists() else dict(meta=dict(problem=name, cells=len(ct), traced=len(ti), spots=len(Y), genes=len(gsel), traced_kinds=tk, other_kinds=ek, clones=Kc), arms={}, masses={})
    level = None; t0 = time.time()
    for arm in arms:
        if arm in out["arms"]:
            continue
        print(f"  {name} {arm}: start", flush=True); res = fit(Yg, Gme, gu, fT, prof, kc, clone, Kc, D, arm, level=(levels or {}).get(arm, level))
        if arm == "notree":
            e = res["emp"]; level = float(np.nanmedian(e[np.isfinite(e)]))
        Pm = res["phi"].T @ np.eye(Kc)[clone]; h = heldout(res)
        out["arms"][arm] = dict(scores=h, s2=[float(x) for x in res["s2"]], emp=[float(x) for x in res["emp"]], trace=res["trace"], objective=res["objective"]); out["masses"][arm] = Pm.tolist()
        out["masses"][f"{arm}|types"] = np.concatenate([res["we"], res["wt"]], 1).tolist(); json.dump(out, open(f, "w"), default=float)
        print(f"  {name} {arm}: held-out {h['heldout_spearman']:.3f} " + " ".join(f"{k} {v:.3f}" for k, v in h.items() if k != "heldout_spearman") + f"  [{time.time() - t0:.0f}s]", flush=True)
    out["done"] = True; json.dump(out, open(f, "w"), default=float)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for name, path in PROBLEMS:
        f = OUT / f"{name}.json"
        if f.exists() and json.load(open(f)).get("done"):
            continue
        print(f"{name}: start", flush=True); run(name, path, f); print(f"{name}: done ({time.time() - t0:.0f}s total)", flush=True)
