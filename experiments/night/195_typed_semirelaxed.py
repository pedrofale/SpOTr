"""All cells mapped, with per-kind spot capacities and relaxed cell masses: an (alpha, tau) sweep.  2026-10-21.
FIXED BEFORE RUNNING.  Pedro, 2026-10-21: "Let's only focus on the 'all cells mapped' case. That's the target"; on per-kind
capacities with relaxed cell masses, under a simulated dissociation bias: "Yes please".
The map (every dissociated cell, one coupling T, cells x spots):
  spot capacity  hard and per kind: for each kind k (tumour; each of the 3 microenvironment types), sum_{i in k} T_is =
                 q_s * w_sk, with q_s proportional to the cells in spot s (the true count in simulation) and w_sk the spot's
                 share of kind k from a non-negative least-squares deconvolution of its normalised counts into the mean
                 profiles of the dissociated cells of each kind (cell types as annotated)
  cell mass      relaxed: KL(sum_s T_is || a_i) with weight tau, a_i = kind k's total capacity / its number of dissociated
                 cells (balanced within kind); tau = infinity is balanced within kind
  costs          the expression cost (1 - alpha) M for every cell; the tree term alpha * c * sum w_ij (A_ij - D_st)^2 T_is T_jt
                 only between pairs of traced tumour cells (w_ij = 1), tree distances in spatial units (as 193_), c calibrated
                 on the traced rows (189_.tree_scale); with relaxed masses the term sum w_ij A_ij^2 r_i r_j, which has no
                 spatial content and would only shape the masses, is left out
  solution       Sinkhorn throughout, so that the map is differentiable (Pedro, 2026-10-21: "Please use Sinkhorn so that we have
                 a differentiable map"; "Why torch and not the jax based approach we had?"): entropic FGW by damped fixed-point
                 iteration, T <- (T + Sinkhorn(linearised cost)) / 2, each step a semi-relaxed log-domain Sinkhorn in JAX
                 (jit; fixed iteration counts: 1000 for the first step, 150 per later step, 40 outer steps; entropy eps = 0.02 x
                 the doubly-centred spread of M).  Not spotr.py's sinkhorn_unrolled_safe: its kernel-domain exp_smooth_clipped
                 saturates the exponent (audit defect D1), so it does not compute exp(-C / eps) at small eps.
                 Kinds do not interact, so T splits into blocks: the tumour block (traced and untraced tumour cells) by
                 entropic FGW, each microenvironment type by semi-relaxed entropic OT on M.  The sweep runs without gradients.
Reference arms (same solver): 'all-single' -- one capacity per spot for all cells, balanced (193_'s 'all' in entropic form);
'typed tau=inf' -- typed capacities, balanced within kind.
Revised before running, after a smoke test (seed 43): tau as first planned ({0.03, 0.1, 0.3, 1}) let cell masses degenerate
(0 to 280 x the balanced mass: hub cells); at alpha 0 masses stay between 0.2 and 2.5 x at tau 1 and within 0.9-1.1 x at tau 10,
covering the simulated capture factors, so tau in {1, 3, 10, 30, inf}.  The same check found that at alpha 0 relaxing does not
correct the dissociated clone mix (map-vs-tissue clone tv 0.242 -> 0.224 at tau 1): expression alone cannot tell which clone
is over-captured, so any correction must come through the tree.
Simulator 189_ (tree distances in spatial units), realistic expression; microenvironment interspersed / clone-dependent;
dissociation bias off (half the dissociated cells kept uniformly) or on (capture rates: tumour 1, microenvironment types
0.3 / 1 / 2, a per-clone tumour factor exp(N(0, 0.6))); seeds 43, 44, 45 (fresh): 12 tumours; single-cell tree and clonal tree
of 20 clones; alpha in {0, 0.05, 0.1, 0.2, 0.3, 0.5}; tau (relative to the doubly-centred spread of M) in {1, 3, 10, 30, inf}.
Truth only to evaluate: traced tumour cells' median placement error and clone composition error; untraced tumour and
microenvironment cells' placement errors; microenvironment-type composition error; and the bias readout -- total variation
between the map's clone proportions (traced cells' mass) and the tissue's, and the same for the microenvironment types.
Expectations stated before running (reads):
  E1  typed capacities remove the subset problem: 'typed tau=inf' places the traced cells better than 'all-single' at each
      arm's best alpha, with and without bias;
  E2  with dissociation bias, the best tau < infinity beats 'typed tau=inf' on the traced cells' placement and clone
      composition, and brings the map's clone proportions closer to the tissue's; without bias, relaxing costs little;
  E3  read: the (alpha, tau) surface, and whether the best alpha moves with tau.
Corrected after the run (2026-10-21): the microenvironment blocks used the cost M instead of (1 - alpha) M, i.e. an entropy of
eps / (1 - alpha) for those blocks; the blocks are independent, so every traced and untraced tumour-cell result in
spotr_out/typed_semirelaxed.json is unaffected, and only the microenvironment cells' own maps at alpha > 0 differ slightly.
Usage: python 195_typed_semirelaxed.py [procs] [SUMMARY]    (writes spotr_out/typed_semirelaxed.json)
"""
import os, sys, json, time, math, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np
HERE = Path(__file__).resolve().parent
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
TAUS = [1.0, 3.0, 10.0, 30.0, None]                     # None = infinity (balanced within kind)
EPS_REL = 0.02
PATTERNS, BIASES, SEEDS = ["interspersed", "clone"], ["off", "on"], [43, 44, 45]
CAPTURE = {"off": dict(keep=0.5), "on": dict(keep=0.5, tumour=1.0, me=[0.3, 1.0, 2.0], clone_sd=0.6)}


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def kind_shares(S, X):
    """NNLS of each spot's normalised counts on the mean profiles of the dissociated tumour cells and of each ME type present"""
    from scipy.optimize import nnls
    use = X["use"]; C = S["counts"][use]; kind = S["kind"][use]; mt = S["me_type"][use]
    groups = [kind == 0] + [(kind == 1) & (mt == k) for k in range(X["nT"])]
    present = [g.any() for g in groups]; Pm = np.stack([C[g].sum(0) / C[g].sum() for g, pr in zip(groups, present) if pr], 1)
    B = X["Sc"] / np.maximum(X["Sc"].sum(1, keepdims=True), 1.0)
    w = np.array([nnls(Pm, b)[0] for b in B]); w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    out = np.zeros((len(B), 1 + X["nT"])); out[:, np.where(present)[0]] = w; return out


def _jax():
    """JAX with 64-bit floats, without preallocating the GPU (several worker processes share it)"""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax


_SOLVER = {}


def _build_solver():
    """jit-compiled entropic FGW (damped fixed point of semi-relaxed log-domain Sinkhorn steps), fixed iteration counts so that
    the whole map is reverse-differentiable in alpha, c, lam = tau / (tau + eps) and the costs (lax.fori_loop / scan)"""
    jax = _jax(); import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    def sinkhorn(G, la, lq, eps, lam, f, g, iters):
        def body(_, fg):
            f, g = fg
            f = lam * (eps * la - eps * logsumexp((g[None, :] - G) / eps, axis=1))
            g = eps * lq - eps * logsumexp((f[:, None] - G) / eps, axis=0)
            return (f, g)
        return jax.lax.fori_loop(0, iters, body, (f, g))

    def solve(M, Wc, WPc, WQc, idx, D, la, lq, alpha, eps, lam, T_init, J, it0, it1, eta, warm):
        f0, g0 = sinkhorn(M, la, lq, eps, lam, jnp.zeros(M.shape[0]), jnp.zeros(M.shape[1]), it0)
        T0 = T_init if warm else jnp.exp((f0[:, None] + g0[None, :] - M) / eps); D2 = D * D     # warm: continue from a given map (2026-10-22)
        def step(carry, _):
            T, f, g = carry
            # the tree term's half-gradient: sum_kl W_ik (Delta_ik - D_jl)^2 T_kl, Delta = WP / W.  Its first part, sum_k W_ik Delta_ik^2
            # r_k (r = row masses), is a constant per row when the rows are balanced and Sinkhorn absorbs it; with relaxed rows it is
            # not, and leaving it out (as 195_-202_ did, a2=False) rewards piling mass onto the traced rows (exp-unmatched-oct12 section 67)
            # computed on the rows that carry a tree term only (idx; Wc, WPc, WQc are their block) and zero elsewhere -- the same
            # numbers as over all rows, at a fraction of the cost when most cells are untraced (2026-10-22)
            Ts = T[idx]
            A = jnp.zeros_like(T).at[idx].set(-2.0 * (WPc @ Ts @ D) + Wc @ Ts @ D2 + (WQc @ Ts.sum(1))[:, None])
            G = (1 - alpha) * M + alpha * 2.0 * A
            f, g = sinkhorn(G, la, lq, eps, lam, f, g, it1)
            Tn = (1 - eta) * T + eta * jnp.exp((f[:, None] + g[None, :] - G) / eps)
            return (Tn, f, g), jnp.abs(Tn - T).sum()
        (T, _, _), trace = jax.lax.scan(step, (T0, f0, g0), None, length=J)
        return T, trace

    _SOLVER["solve"] = jax.jit(solve, static_argnums=(12, 13, 14, 16))
    _SOLVER["eta"] = 0.5


def fgw_sinkhorn(M, W, WP, D, a, q, alpha, c, eps, tau, J=40, it0=1000, it1=150, grad=False, a2=True, all_rows=False, T_init=None):
    """entropic FGW in JAX: T <- (T + Sinkhorn(linearised cost)) / 2, each step a semi-relaxed log-domain Sinkhorn
    (tau None: balanced); pair weights W and target distances WP / W for the tree term; returns a numpy coupling and the
    change per outer iteration.  With grad=True the caller differentiates _SOLVER['solve'] directly (this wrapper converts).
    a2: include the tree term's sum_k W_ik Delta_ik^2 r_k part (added 2026-10-21; a2=False reproduces 195_-202_ as run, which is exact
    for balanced rows only).  all_rows: compute the tree term over every row, as before 2026-10-22 (for checks; same result).
    T_init: start the fixed point from this map instead of the expression-only map (continuation along alpha; 2026-10-22)."""
    if "solve" not in _SOLVER:
        _build_solver()
    jax = _jax(); import jax.numpy as jnp
    lam = 1.0 if tau is None else tau / (tau + eps)
    Jx = J if (alpha > 0 and W.sum() > 0) else 0
    idx = np.arange(len(W)) if all_rows else np.where((W > 0).any(1))[0]
    if len(idx) == 0:
        idx = np.zeros(1, int)
    Ws, WPs = W[np.ix_(idx, idx)], WP[np.ix_(idx, idx)]
    WQ = np.where(Ws > 0, WPs * WPs / np.maximum(Ws, 1e-300), 0.0) if a2 else np.zeros_like(Ws)          # W Delta^2, on the block
    T, trace = _SOLVER["solve"](jnp.asarray(M), jnp.asarray(Ws * c), jnp.asarray(WPs * c), jnp.asarray(WQ * c), jnp.asarray(idx), jnp.asarray(D), jnp.log(jnp.asarray(a)),
                                jnp.log(jnp.asarray(np.maximum(q, 1e-300))), float(alpha), float(eps), float(lam),
                                jnp.asarray(T_init if T_init is not None else np.zeros((len(M), len(q)))), Jx, it0, it1, _SOLVER["eta"], T_init is not None)
    return np.asarray(T), np.asarray(trace)


def solve_typed(X, sh, alpha, tau_rel, typed=True):
    """the whole coupling.  typed: the tumour block (fgw_sinkhorn on capacities q * share_tumour) and each ME type (Sinkhorn on
    q * share_type); not typed: one block of all cells on capacities q (the single-capacity reference), balanced"""
    m70, m189 = mod("m70", "70_spotr.py"), mod("m189", "189_me_sim.py")
    n, m = len(X["cl"]), len(X["q"]); T = np.zeros((n, m)); scaleM = float(m70.dc_std(X["M"])); eps = EPS_REL * scaleM
    tau = None if tau_rel is None else tau_rel * scaleM
    if not typed:
        c = m189.tree_scale(X)
        return fgw_sinkhorn(X["M"], X["W"], X["WP"], X["D"], np.full(n, 1.0 / n), X["q"], alpha, c, eps, None)[0]
    kinds = [np.where(X["kind"] == 0)[0]] + [np.where((X["kind"] == 1) & (X["me_type"] == k))[0] for k in range(X["nT"])]
    for k, rows in enumerate(kinds):
        if len(rows) == 0 or sh[:, k].sum() <= 0:
            continue
        qk = X["q"] * sh[:, k]; ak = np.full(len(rows), qk.sum() / len(rows)); Mk = X["M"][rows]
        if k == 0:
            Xk = dict(X); Xk.update(M=Mk, W=X["W"][np.ix_(rows, rows)], WP=X["WP"][np.ix_(rows, rows)], q=qk / qk.sum(), cl=X["cl"][rows],
                                    traced=X["traced"][rows], kind=X["kind"][rows])
            c = m189.tree_scale(Xk) if Xk["W"].sum() > 0 else 1.0
            T[rows] = fgw_sinkhorn(Mk, Xk["W"], Xk["WP"], X["D"], ak, qk, alpha, c, eps, tau)[0]
        else:
            T[rows] = fgw_sinkhorn((1 - alpha) * Mk, np.zeros((len(rows), len(rows))), np.zeros((len(rows), len(rows))), X["D"], ak, qk, 0.0, 1.0, eps, tau)[0]
    return T


def bias_readout(T, X, S):
    """total variation between the map's clone (traced cells) and ME-type proportions and the tissue's (the spatial half)"""
    tr = (X["kind"] == 0) & X["traced"]; K = X["K"]
    mc = np.bincount(X["cl"][tr], weights=T[tr].sum(1), minlength=K); tc = X["Ncl"].sum(0)
    me = X["kind"] == 1; mm = np.bincount(X["me_type"][me], weights=T[me].sum(1), minlength=X["nT"]); tm = X["Nme"].sum(0)
    tv = lambda a, b: float(0.5 * np.abs(a / max(a.sum(), 1e-300) - b / max(b.sum(), 1e-300)).sum())
    return dict(clone_prop_tv=tv(mc, tc), me_prop_tv=tv(mm, tm) if tm.sum() > 0 else None)


def job(args):
    pattern, bias, seed = args
    import warnings; warnings.filterwarnings("ignore")
    m189 = mod("m189", "189_me_sim.py")
    S = m189.simulate_me(seed, pattern=pattern); out = dict(pattern=pattern, bias=bias, seed=seed, arms={})
    for res in ["single", "clonal"]:
        X = m189.problem_me(S, "all", res, tree_units="space", capture=CAPTURE[bias]); sh = kind_shares(S, X)
        true_sh = np.stack([X["Ncl"].sum(1)] + [X["Nme"][:, k] for k in range(3)], 1) / np.maximum(X["ncell"][:, None], 1)
        out[f"share_err {res}"] = float(np.abs(sh - true_sh).sum(1).mean() / 2); out[f"n {res}"] = len(X["cl"])
        out[f"dissociated mix {res}"] = dict(tumour=float((X["kind"] == 0).mean()), clone_prop_tv=bias_readout(np.ones((len(X["cl"]), 1)) / len(X["cl"]), X, S)["clone_prop_tv"])
        out["arms"][f"all-single {res}"] = {}
        for a in ALPHAS:
            T = solve_typed(X, sh, a, None, typed=False); out["arms"][f"all-single {res}"][str(a)] = {**m189.metrics(T, X), **bias_readout(T, X, S)}
        for tau in TAUS:
            key = f"typed tau={'inf' if tau is None else tau} {res}"
            out["arms"][key] = {}
            for a in ALPHAS:
                T = solve_typed(X, sh, a, tau); out["arms"][key][str(a)] = {**m189.metrics(T, X), **bias_readout(T, X, S)}
    return out


def summary(res):
    A = [str(a) for a in ALPHAS]; tags = ["all-single"] + [f"typed tau={'inf' if t is None else t}" for t in TAUS]
    bestv = lambda d, key="traced_err": min(d[a][key] for a in A)
    for res_ in ["single", "clonal"]:
        print(f"######## {res_} tree.  traced cells' placement error at each arm's best alpha (mean of 3 tumours) | clone tv | map-vs-tissue clone proportion tv | best alpha")
        for pat in PATTERNS:
            for b in BIASES:
                rr = [r for r in res if (r["pattern"], r["bias"]) == (pat, b)]
                print(f"== {pat}, dissociation bias {b}: share error {np.mean([r[f'share_err {res_}'] for r in rr]):.3f}; dissociated clone mix vs tissue tv "
                      f"{np.mean([r[f'dissociated mix {res_}']['clone_prop_tv'] for r in rr]):.3f}; tumour fraction of dissociated cells {np.mean([r[f'dissociated mix {res_}']['tumour'] for r in rr]):.2f}")
                for t in tags:
                    d = [r["arms"][f"{t} {res_}"] for r in rr]; ba = [min(A, key=lambda a: q[a]["traced_err"]) for q in d]
                    print(f"   {t:18s}: {np.mean([bestv(q) for q in d]):5.0f} um | clone tv {np.mean([q[x]['clone_tv'] for q, x in zip(d, ba)]):.3f} | "
                          f"clone prop tv {np.mean([q[x]['clone_prop_tv'] for q, x in zip(d, ba)]):.3f} | ME err {np.mean([q[x]['me_err'] for q, x in zip(d, ba)]):.0f} | "
                          f"untraced {np.mean([q[x]['untraced_err'] for q, x in zip(d, ba)]):.0f} | alpha {ba} | alpha 0: {np.mean([q['0.0']['traced_err'] for q in d]):.0f}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    t0 = time.time(); procs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    fn = HERE / "spotr_out" / "typed_semirelaxed.json"
    if len(sys.argv) > 2 and sys.argv[2] == "SUMMARY":
        res = json.load(open(fn))
    else:
        with get_context("spawn").Pool(procs) as pool:
            res = pool.map(job, [(p, b, s) for p in PATTERNS for b in BIASES for s in SEEDS])
        json.dump(res, open(fn, "w"), indent=1, default=float)
    summary(res)
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
