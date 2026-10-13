"""SpOTr against Tangram on the paper's simulated data (2026-10-13), for the poster's simulation panel.

Simulator: src/spotr/utils.simulate_data, the paper's framework (Methods, "Simulated data"): a complete binary tree,
cell positions from Brownian motion on the tree in four spatial modes ('main_clades', 'main_and_subclades',
'subclades', 'unstructured'), expression mode 'mix' (tree signal and a tree-independent spatial program equally
weighted), Poisson counts, Visium-like spots on a 0.05-unit grid (scaled here to 50 um).
Metric: the paper's -- Pearson correlation between true and estimated subclone (level-2 clade) proportions across
spots, averaged over subclones -- on the spots outside SpOTr's anchor (also recorded over all spots).
Methods: Tangram (73_tangram.py settings; expression only); SpOTr as on the poster: growth-law tree term with L and
c fitted on the anchor and the tree alone (70_spotr.anchor_law), the weight and one mixing length per subclone
learned through the mapping against the anchor (20% of spots, half fit, half validation); expression-only exact OT is
recorded as a reference.
Usage: python 81_sim_benchmark.py [reps] [procs]      (writes spotr_out/sim_benchmark.json)
"""
import os, sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np

HERE = Path(__file__).resolve().parent
MODES = ["main_clades", "main_and_subclades", "subclades", "unstructured"]
UM = 1000.0                                                   # simulator units -> um (0.05-unit spots = 50 um)


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def job(args):
    mode, rep = args
    import warnings; warnings.filterwarnings("ignore")
    import ot, scanpy as sc
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist, squareform
    utils = mod("spotr_utils", HERE.parents[1] / "src" / "spotr" / "utils.py")
    m70 = mod("m70", HERE / "70_spotr.py"); m55 = mod("m55", HERE / "55_decomposition_check.py")
    m73 = mod("m73", HERE / "73_tangram.py")
    ss, sp, tdm, sdm, true_c, tree = utils.simulate_data(mode, "mix", seed=rep, n_cells=512, n_genes=100, n_traits=5,
                                                          obs_model="poisson")
    cells, spots = list(ss.obs_names), list(sp.obs_names)
    n, m = len(cells), len(spots)
    lognorm = lambda A: np.log1p(A / np.maximum(A.sum(1, keepdims=True), 1.0) * 1e4)
    Xc = lognorm(np.asarray(ss.X, float)); Xs = lognorm(np.asarray(sp.X, float))
    M = ot.dist(Xc, Xs); M /= M.max()
    xy_c = np.asarray(ss.obsm["spatial"], float) * UM
    xy_s = (np.asarray(sp.obsm["spatial"], float) + 0.025) * UM          # grid corner -> spot centre
    spot_of = np.array([spots.index(s) for s in ss.obs["spot"]])
    clades = sorted(ss.obs["clade_level2"].astype(str).unique()); K = len(clades)
    cl = np.array([clades.index(c) for c in ss.obs["clade_level2"].astype(str)])
    Nspot = np.zeros((m, K)); np.add.at(Nspot, (spot_of, cl), 1.0)
    q = Nspot.sum(1) / n
    D = np.asarray(tdm.loc[cells, cells].values, float)                  # patristic, the tree is ultrametric
    AGE = D / 2.0
    J = np.random.default_rng(1000 * rep + 20).choice(m, max(6, int(round(0.2 * m))), replace=False)
    X = dict(M=M, D=squareform(pdist(xy_s)), q=q, NC=m55.clone_size(AGE).astype(float),
             rho=n / ConvexHull(xy_c).volume, cl=cl, K=K, Nspot=Nspot, spots=xy_s, xy_true=xy_c, AGE=AGE, J=J,
             L0=300.0, c0=1.0, inB=None, tag=f"sim-{mode}-{rep}")
    X["L0"], X["c0"] = m70.anchor_law(X, fit_c=True)                     # law from the anchor and the tree alone
    T = {"expression": ot.emd(np.ones(n) / n, q, M, numItermax=10 ** 7),
         "tangram": m73.tangram(Xc, Xs, q, rep)}
    init = os.environ.get("INIT", "product")                         # "expression": start the solver at the expression map
    log, best = m70.learn(X, cl.copy(), learn_a=True, a0=0.197 / 1.197,
                          T_init=T["expression"] if init == "expression" else None)
    T["spotr"] = best["T"]
    # the anchor's validation spots choose between the learned map and the expression-only map (alpha = 0)
    Jv = J[len(J) // 2:]
    nll_learned, nll_expr = best["nll_val"], m70.anchor_nll(T["expression"], X, Jv)
    T["spotr_select"] = T["spotr"] if nll_learned <= nll_expr else T["expression"]
    held = np.setdiff1d(np.arange(m), J)
    true_frac = Nspot / Nspot.sum(1, keepdims=True)

    def score(TT, idx):
        inf = TT.T @ np.eye(K)[cl]; inf = inf / np.maximum(inf.sum(1, keepdims=True), 1e-300)
        rs = [np.corrcoef(true_frac[idx, k], inf[idx, k])[0, 1] for k in range(K)
              if true_frac[idx, k].std() > 0 and inf[idx, k].std() > 0]
        return float(np.mean(rs)) if rs else float("nan")
    out = dict(mode=mode, rep=rep, cells=n, spots=m, L0=X["L0"], c0=X["c0"], alpha=best["a"], L=best["L"],
               nll_learned=nll_learned, nll_expression=nll_expr, picked="learned" if nll_learned <= nll_expr else "expression")
    for k, TT in T.items():
        out[k] = dict(pearson_heldout=score(TT, held), pearson_all=score(TT, np.arange(m)), err=m70.err(TT, X))
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    t0 = time.time()
    jobs = [(mode, r) for mode in MODES for r in range(reps)]
    if os.environ.get("ONLY"):                                   # e.g. ONLY=subclades:8,unstructured:4
        jobs = [(o.split(":")[0], int(o.split(":")[1])) for o in os.environ["ONLY"].split(",")]
    with get_context("spawn").Pool(procs) as pool:
        res = pool.map(job, jobs)
    json.dump(res, open(HERE / "spotr_out" / os.environ.get("OUTJSON", "sim_benchmark.json"), "w"), indent=1, default=float)
    for mode in MODES:
        rr = [r for r in res if r["mode"] == mode]
        if not rr:
            continue
        print(f"{mode:20s} " + "  ".join(f"{k} {np.nanmean([r[k]['pearson_heldout'] for r in rr]):.3f}"
                                          for k in ["spotr", "spotr_select", "tangram", "expression"]))
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
