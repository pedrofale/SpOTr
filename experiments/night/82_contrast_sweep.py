"""Does the per-clade decomposition change the map, and when?  A contrast sweep on the planted control (2026-10-13).

Setting: 66_planted_control.build -- PEtracer M3-4, seed-0 split; half of the clades (group draw g) get extra
dispersal N(0, sigma^2) on the dissociated and the spot side, spots rebuilt from the moved cells, so expression and
tree stay real and only the mixing is planted.  sigma = 0, 150, 300, 500, 800 um; group draws 0-3; anchor draws
0-1 (20% of spots, half fit, half validation): 8 replicates per sigma.  Fixed before running.

Arms, all with the same learner (70_spotr.learn: exact map, Sinkhorn best response, anchor NLL, Adam; the tree
weight learned from w = 0.197; law L0 = 305 um, c = 1.36 as in 66_):
  shared      one within-clade mixing length for every clade
  per-clade   one mixing length per clade (18)
  random      random clade-sized groups of cells, the "just more parameters" control
  oracle      each clade's L held at its measured value on the moved true positions; only the weight learned
  expression  expression-only exact OT, for reference
Readouts: placement error (median distance from each dissociated cell's true position to its argmax spot) and,
for per-clade, recovery of the planted groups (AUC dispersed > undisturbed) and Spearman with the measured L.
Usage: python 82_contrast_sweep.py [procs]     (writes spotr_out/contrast_sweep.json)
"""
import os, sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np

HERE = Path(__file__).resolve().parent
SIGMAS, GDRAWS, ADRAWS = [0.0, 150.0, 300.0, 500.0, 800.0], [0, 1, 2, 3], [0, 1]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def job(args):
    sigma, g, r = args
    import warnings; warnings.filterwarnings("ignore")
    import ot
    from scipy.stats import spearmanr, mannwhitneyu
    m70 = mod("m70", "70_spotr.py"); m66 = mod("m66", "66_planted_control.py")
    B = m66.build(sigma, g)
    z = np.load(HERE.parent / "petracer" / "petracerFAIR_M3-4.npz", allow_pickle=True)
    n_all = len(z["cell_names"]); use = np.random.default_rng(0).permutation(n_all)[n_all // 2:]
    AGE = np.asarray(z["C_tree"], float)[np.ix_(use, use)] / 2.0
    X = dict(M=B["M"], D=B["D"], q=B["q"], NC=B["NC"], rho=B["rho"], cl=B["cl"], K=B["K"], Nspot=B["Nspot"],
             spots=B["centres"], xy_true=B["xy_true"], AGE=AGE, J=B["anchors"][0.2][r], L0=305.0, c0=1.36,
             inB=B["inB"], tag=f"planted_s{int(sigma)}_g{g}_a{r}")
    n, K, cl = len(X["cl"]), X["K"], X["cl"]
    out = dict(sigma=sigma, gdraw=g, adraw=r, arms={})
    Te = ot.emd(np.ones(n) / n, X["q"], X["M"], numItermax=10 ** 7)
    out["arms"]["expression"] = dict(err=m70.err(Te, X))
    meas = m70.measured_L(X, cl, X["c0"])
    Lmeas = np.array([meas.get(k, np.nan) for k in range(K)])
    a0 = 0.197 / 1.197
    runs = {"shared": dict(groups=np.zeros(K, int)[cl]),
            "per-clade": dict(groups=cl.copy()),
            "random": dict(groups=np.random.default_rng(300 + 10 * g + r).permutation(cl), random=True),
            "oracle": dict(groups=cl.copy(), L_init=np.where(np.isfinite(Lmeas), Lmeas, X["L0"]))}
    for arm, o in runs.items():
        kw = dict(learn_a=True, a0=a0)
        if o.get("random"):
            kw["pair_groups"] = o["groups"]
        if "L_init" in o:
            kw["L_init"] = o["L_init"]; kw["learn_L"] = False
        log, best = m70.learn(X, o["groups"], **kw)
        rec = dict(err=best["err"], alpha=best["a"], step=best["step"], nll_val=best["nll_val"])
        if arm == "per-clade":
            L = np.array(best["L"]); ok = np.isfinite(Lmeas) & (np.bincount(cl, minlength=K) >= 30)
            inB = X["inB"].astype(bool)
            if sigma > 0 and (inB & ok).any() and (~inB & ok).any():
                rec["auc"] = float(mannwhitneyu(np.log(L[inB & ok]), np.log(L[~inB & ok])).statistic /
                                   ((inB & ok).sum() * (~inB & ok).sum()))
            rec["rho_meas"] = float(spearmanr(L[ok], Lmeas[ok]).statistic)
            rec["L"] = L.tolist(); rec["L_meas"] = Lmeas.tolist(); rec["inB"] = inB.tolist()
        out["arms"][arm] = rec
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    t0 = time.time()
    jobs = [(s, g, r) for s in SIGMAS for g in GDRAWS for r in ADRAWS]
    with get_context("spawn").Pool(procs) as pool:
        res = pool.map(job, jobs)
    json.dump(res, open(HERE / "spotr_out" / "contrast_sweep.json", "w"), indent=1, default=float)
    print("sigma  " + "  ".join(f"{a:>10s}" for a in ["expression", "shared", "per-clade", "random", "oracle"]) +
          "   per-clade-vs-shared (wins)   AUC   rho")
    for s in SIGMAS:
        rr = [x for x in res if x["sigma"] == s]
        e = {a: np.array([x["arms"][a]["err"] for x in rr]) for a in ["expression", "shared", "per-clade", "random", "oracle"]}
        auc = [x["arms"]["per-clade"].get("auc", np.nan) for x in rr]; rho = [x["arms"]["per-clade"]["rho_meas"] for x in rr]
        d = e["shared"] - e["per-clade"]
        print(f"{s:5.0f}  " + "  ".join(f"{e[a].mean():10.1f}" for a in e) +
              f"   {d.mean():+6.1f} um ({(d > 0).sum()}/{len(d)})        {np.nanmean(auc):.2f}  {np.mean(rho):+.2f}")
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
