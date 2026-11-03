"""All cells mapped with ONE capacity per spot and relaxed cell masses: an (alpha, tau) sweep.  2026-10-21.  FIXED BEFORE RUNNING.
Pedro, 2026-10-21: "Let's not use the per-cell type capacities, please" (after 195_).
The map (every dissociated cell, one coupling T, cells x spots; no deconvolution, no cell-type information):
  spot capacity  hard, one per spot: sum_i T_is = q_s, q_s proportional to the cells in spot s (the true count in simulation)
  cell mass      relaxed: tau KL(sum_s T_is || 1/n); tau = infinity is balanced (195_'s 'all-single')
  costs          (1 - alpha) M for every cell; alpha c sum_{traced tumour pairs} (A_ij - D_st)^2 T_is T_jt (tree distances in spatial
                 units, c calibrated on the traced rows; the A^2 term left out with relaxed masses); entropy eps = 0.02 x the
                 doubly-centred spread of M
  solver         195_.fgw_sinkhorn: entropic FGW by damped fixed point of semi-relaxed log-domain Sinkhorn steps in JAX
                 (differentiable)
Same 12 tumours as 195_ (interspersed / clone-dependent microenvironment x dissociation bias off / on x seeds 43-45), single-cell
tree and clonal tree of 20 clones; alpha in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8} (extended past 195_'s edge at 0.5);
tau (relative to the doubly-centred spread of M) in {1, 3, 10, 30, inf}.
Truth only to evaluate: as 195_ (traced cells' placement and clone composition; untraced and microenvironment cells' placement;
the map's clone mix against the tissue's), plus each spot's mapped tumour share against the truth -- free again here.
195_'s typed arms as references (from spotr_out/typed_semirelaxed.json).
Expectations stated before running (reads): E5 with dissociation bias, relaxing the masses (best tau < inf) improves the traced
cells' placement over the balanced single capacity; E6 read: whether relaxed masses change the mapped tumour share per spot
and the subset problem (traced cells crowding into part of the slide) seen with one capacity in 190_ / 193_.
Usage: python 197_single_relaxed.py [procs] [SUMMARY]    (writes spotr_out/single_relaxed.json)
"""
import os, sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np
HERE = Path(__file__).resolve().parent
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8]
TAUS = [1.0, 3.0, 10.0, 30.0, None]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def solve_single(X, alpha, tau_rel, a2=True):
    """one capacity per spot, relaxed cell masses (tau None: balanced), the tree only between traced tumour cells.
    a2=False: the tree term without its sum_k W Delta^2 r_k part, as 197_, 199_, 200_ and 202_ were run (195_.fgw_sinkhorn)"""
    m70, m189, m195 = mod("m70", "70_spotr.py"), mod("m189", "189_me_sim.py"), mod("m195", "195_typed_semirelaxed.py")
    n = len(X["cl"]); scaleM = float(m70.dc_std(X["M"])); eps = m195.EPS_REL * scaleM; tau = None if tau_rel is None else tau_rel * scaleM
    c = m189.tree_scale(X)
    return m195.fgw_sinkhorn(X["M"], X["W"], X["WP"], X["D"], np.full(n, 1.0 / n), X["q"], alpha, c, eps, tau, a2=a2)[0]


def job(args):
    pattern, bias, seed = args
    import warnings; warnings.filterwarnings("ignore")
    m189, m195 = mod("m189", "189_me_sim.py"), mod("m195", "195_typed_semirelaxed.py")
    S = m189.simulate_me(seed, pattern=pattern); out = dict(pattern=pattern, bias=bias, seed=seed, arms={})
    for res in ["single", "clonal"]:
        X = m189.problem_me(S, "all", res, tree_units="space", capture=m195.CAPTURE[bias])
        for tau in TAUS:
            key = f"single tau={'inf' if tau is None else tau} {res}"; out["arms"][key] = {}
            for a in ALPHAS:
                T = solve_single(X, a, tau); out["arms"][key][str(a)] = {**m189.metrics(T, X), **m195.bias_readout(T, X, S)}
    return out


def summary(res):
    A = [str(a) for a in ALPHAS]; ref = {(r["pattern"], r["bias"], r["seed"]): r for r in json.load(open(HERE / "spotr_out" / "typed_semirelaxed.json"))}
    tags = [f"single tau={'inf' if t is None else t}" for t in TAUS]
    for res_ in ["single", "clonal"]:
        print(f"######## {res_} tree.  traced cells' placement error at each arm's best alpha (mean of 3) | clone tv | map-vs-tissue clone mix tv | tumour-share error | best alpha")
        for pat in ["interspersed", "clone"]:
            for b in ["off", "on"]:
                rr = [r for r in res if (r["pattern"], r["bias"]) == (pat, b)]; print(f"== {pat}, dissociation bias {b}")
                for t in tags:
                    d = [r["arms"][f"{t} {res_}"] for r in rr]; ba = [min(A, key=lambda a: q[a]["traced_err"]) for q in d]
                    print(f"   {t:16s}: {np.mean([q[x]['traced_err'] for q, x in zip(d, ba)]):5.0f} um | clone tv {np.mean([q[x]['clone_tv'] for q, x in zip(d, ba)]):.3f} | "
                          f"mix tv {np.mean([q[x]['clone_prop_tv'] for q, x in zip(d, ba)]):.3f} | share err {np.mean([q[x]['tumour_share_err'] for q, x in zip(d, ba)]):.3f} | "
                          f"ME {np.mean([q[x]['me_err'] for q, x in zip(d, ba)]):.0f} | alpha {ba}")
                for t in ["typed tau=inf", "typed tau=1.0"]:
                    d = [ref[(r["pattern"], r["bias"], r["seed"])]["arms"][f"{t} {res_}"] for r in rr]; A5 = [a for a in A if a in d[0]]
                    print(f"   (195_ {t}: {np.mean([min(q[a]['traced_err'] for a in A5) for q in d]):.0f} um)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    t0 = time.time(); procs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    fn = HERE / "spotr_out" / "single_relaxed.json"
    if len(sys.argv) > 2 and sys.argv[2] == "SUMMARY":
        res = json.load(open(fn))
    else:
        with get_context("spawn").Pool(procs) as pool:
            res = pool.map(job, [(p, b, s) for p in ["interspersed", "clone"] for b in ["off", "on"] for s in [43, 44, 45]])
        json.dump(res, open(fn, "w"), indent=1, default=float)
    summary(res)
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
