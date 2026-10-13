"""Two questions on the contrast sweep's planted data (82_contrast_sweep.py, same replicates), 2026-10-13:

1. Without anchors: is the per-clade decomposition still learnable, and does it still change the map, when the
   outer loss is anchor-free?  The loss (70_spotr.deconv_np) predicts each spot's expression from the clade mix of
   the cells mapped to it and the clades' mean expression from the dissociated cells, scored by squared error on
   half the spots (the other half picks the step).  No clade label of any spatial cell is used.
2. Does our learning scheme matter?  The paper's scheme (70_spotr.learn_unrolled: an entropic map carried across
   steps, 3 rounds of linearise + 50 Sinkhorn iterations per step, autodiff through all of it) with the same law,
   the same per-clade L and the same anchor loss as ours.
Arms per replicate: deconv-shared, deconv-per-clade (our scheme, anchor-free); unrolled-shared, unrolled-per-clade
(paper's scheme, anchor).  The anchor arms of our scheme are read from spotr_out/contrast_sweep.json.
Usage: python 83_outer_and_scheme.py [procs]     (writes spotr_out/outer_and_scheme.json)
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


def recovery(L, X, Lmeas):
    from scipy.stats import spearmanr, mannwhitneyu
    L = np.asarray(L); K, cl = X["K"], X["cl"]
    ok = np.isfinite(Lmeas) & (np.bincount(cl, minlength=K) >= 30); inB = np.asarray(X["inB"], bool)
    out = dict(rho_meas=float(spearmanr(L[ok], Lmeas[ok]).statistic))
    if (inB & ok).any() and (~inB & ok).any():
        out["auc"] = float(mannwhitneyu(np.log(L[inB & ok]), np.log(L[~inB & ok])).statistic / ((inB & ok).sum() * (~inB & ok).sum()))
    return out


def job(args):
    sigma, g, r = args
    import warnings; warnings.filterwarnings("ignore")
    import ot
    m70 = mod("m70", "70_spotr.py"); m66 = mod("m66", "66_planted_control.py")
    B = m66.build(sigma, g)
    z = np.load(HERE.parent / "petracer" / "petracerFAIR_M3-4.npz", allow_pickle=True)
    n_all = len(z["cell_names"]); use = np.random.default_rng(0).permutation(n_all)[n_all // 2:]
    AGE = np.asarray(z["C_tree"], float)[np.ix_(use, use)] / 2.0
    X = dict(M=B["M"], D=B["D"], q=B["q"], NC=B["NC"], rho=B["rho"], cl=B["cl"], K=B["K"], Nspot=B["Nspot"],
             spots=B["centres"], xy_true=B["xy_true"], AGE=AGE, J=B["anchors"][0.2][r], L0=305.0, c0=1.36,
             inB=B["inB"], tag=f"planted_s{int(sigma)}_g{g}_a{r}")
    n, m, K, cl = len(B["cl"]), len(B["centres"]), B["K"], B["cl"]
    # anchor-free loss: signatures and spot expression only
    X["W_sig"] = np.vstack([B["Xsc"][cl == k].mean(0) if (cl == k).any() else np.zeros(B["Xsc"].shape[1]) for k in range(K)])
    X["Y_spot"] = B["Xst"]
    perm = np.random.default_rng(5000 + 10 * g + r).permutation(m)
    X["deconv_fit"], X["deconv_val"] = perm[: m // 2], perm[m // 2:]
    Te = ot.emd(np.ones(n) / n, X["q"], X["M"], numItermax=10 ** 7)
    X["sigma2"] = 1.0
    X["sigma2"] = 2 * m70.deconv_np(Te, X, np.arange(m))     # residual variance of the expression-only map, per gene
    meas = m70.measured_L(X, cl, X["c0"]); Lmeas = np.array([meas.get(k, np.nan) for k in range(K)])
    a0 = 0.197 / 1.197
    out = dict(sigma=sigma, gdraw=g, adraw=r, arms={})
    for arm, groups in [("deconv-shared", np.zeros(K, int)[cl]), ("deconv-per-clade", cl.copy())]:
        log, best = m70.learn(X, groups, learn_a=True, a0=a0, outer="deconv")
        rec = dict(err=best["err"], alpha=best["a"], step=best["step"], alpha_last=log[-1]["a"],
                   deconv_val_start=log[0]["nll_val"], deconv_val_best=best["nll_val"])
        if arm.endswith("per-clade"):
            rec.update(recovery(best["L"], X, Lmeas))
        out["arms"][arm] = rec
    for arm, groups in [("unrolled-shared", np.zeros(K, int)[cl]), ("unrolled-per-clade", cl.copy())]:
        log, best, last = m70.learn_unrolled(X, groups, learn_a=True, a0=a0)
        rec = dict(err=best["err"], err_last=last["err"], alpha=best["a"], alpha_last=last["a"], step=best["step"])
        if arm.endswith("per-clade"):
            rec.update(recovery(best["L"], X, Lmeas)); rec["last"] = recovery(last["L"], X, Lmeas)
        out["arms"][arm] = rec
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    t0 = time.time()
    jobs = [(s, g, r) for s in SIGMAS for g in GDRAWS for r in ADRAWS]
    if os.environ.get("ONE"):
        jobs = jobs[int(os.environ["ONE"]): int(os.environ["ONE"]) + 1]
    with get_context("spawn").Pool(procs) as pool:
        res = pool.map(job, jobs)
    json.dump(res, open(HERE / "spotr_out" / os.environ.get("OUTJSON", "outer_and_scheme.json"), "w"), indent=1, default=float)
    print(json.dumps(res[:1], indent=1, default=float)[:2500])
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
