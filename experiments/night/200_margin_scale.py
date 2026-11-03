"""The margin worlds again, with the tree distances' scale corrected, and the scale from a tumour segmentation.  2026-10-21.
FIXED BEFORE RUNNING.  After a factor-2 error was found in 189_'s tree_units='space' (target tree distances twice the stated
calibration; affects 193_, 195_, 197_ and 199_), and Pedro, 2026-10-21: "I suppose then we could rely on an optional
preprocessing step that decides the scaling factor for the distances based on some tumor-margin segmentation".
The map, as Pedro set it: every dissociated cell mapped in one coupling; one hard capacity per spot (q_s proportional to its
cells); no deconvolution; cell masses relaxed (tau KL to 1/n; tau = inf balanced); the tree only between traced tumour cells;
entropic FGW by Sinkhorn in JAX (195_.fgw_sinkhorn).
Tree scale (one scalar, set before mapping):
  section     the tree distances scaled so their mean over traced pairs equals the mean distance between all spots (189_
              tree_units='space2', the corrected calibration)
  segment     the same, but over the spots inside the tumour region given by a segmentation (tree_units='segment'); here the
              true tumour region (spots with any tumour cell) -- an oracle segmentation, the best case for this option
Worlds as 199_ (margin 1.0; interspersed / clone-dependent; bias off / on; seeds 46-48): 12 tumours; single-cell tree and
clonal tree of 20 clones; alpha in {0, 0.1, 0.2, 0.3, 0.5, 0.8}; tau in {1, 3, 10, inf}.
Truth only to evaluate: as 199_ (traced placement, clone composition, traced mass in tumour-free spots, tumour-share error,
the map's clone mix against the tissue's).
Expectations stated before running (reads): E9 with the corrected scale, balanced maps keep the traced cells out of the margin
(as an exploratory check on seed 46 found) and the tree helps more than in 199_; E10 the segmentation scale improves the traced
cells' placement over the section scale; E11 read: whether relaxed masses still crowd the margin under either scale.
Usage: python 200_margin_scale.py [procs] [SUMMARY]    (writes spotr_out/margin_scale.json)
"""
import os, sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np
HERE = Path(__file__).resolve().parent
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
TAUS = [1.0, 3.0, 10.0, None]
SCALES = {"section": "space2", "segment": "segment"}
CAPTURE = {"off": dict(keep=0.5), "on": dict(keep=0.5, tumour=1.0, me=[0.3, 1.0, 2.0, 0.5], clone_sd=0.6)}


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def job(args):
    pattern, bias, seed = args
    import warnings; warnings.filterwarnings("ignore")
    m189, m195, m197 = mod("m189", "189_me_sim.py"), mod("m195", "195_typed_semirelaxed.py"), mod("m197", "197_single_relaxed.py")
    S = m189.simulate_me(seed, pattern=pattern, margin=1.0)
    out = dict(pattern=pattern, bias=bias, seed=seed, n_margin=S["n_margin"], arms={})
    for res in ["single", "clonal"]:
        for sname, units in SCALES.items():
            X = m189.problem_me(S, "all", res, tree_units=units, capture=CAPTURE[bias])
            for tau in TAUS:
                key = f"{sname} tau={'inf' if tau is None else tau} {res}"; out["arms"][key] = {}
                for a in ALPHAS:
                    T = m197.solve_single(X, a, tau); out["arms"][key][str(a)] = {**m189.metrics(T, X), **m195.bias_readout(T, X, S)}
    return out


def summary(res):
    A = [str(a) for a in ALPHAS]; tags = [f"{s} tau={'inf' if t is None else t}" for s in SCALES for t in TAUS]
    old = {(r["pattern"], r["bias"], r["seed"]): r for r in json.load(open(HERE / "spotr_out" / "margin_sweep.json"))}
    for res_ in ["single", "clonal"]:
        print(f"######## {res_} tree.  at each arm's best alpha (mean of 3): traced placement | clone tv | traced mass in tumour-free spots | tumour-share error | mix tv | best alpha")
        for pat in ["interspersed", "clone"]:
            for b in ["off", "on"]:
                rr = [r for r in res if (r["pattern"], r["bias"]) == (pat, b)]; print(f"== {pat}, bias {b}")
                for t in tags:
                    d = [r["arms"][f"{t} {res_}"] for r in rr]; ba = [min(A, key=lambda a: q[a]["traced_err"]) for q in d]
                    g = lambda k: np.mean([q[x][k] for q, x in zip(d, ba)])
                    print(f"   {t:18s}: {g('traced_err'):5.0f} um | clone tv {g('clone_tv'):.3f} | in tumour-free {g('traced_mass_in_tumour_free'):.3f} (alpha 0.3: "
                          f"{np.mean([q['0.3']['traced_mass_in_tumour_free'] for q in d]):.3f}) | share err {g('tumour_share_err'):.3f} | mix tv {g('clone_prop_tv'):.3f} | alpha {ba} | alpha 0: {np.mean([q['0.0']['traced_err'] for q in d]):.0f}")
                d = [old[(r["pattern"], r["bias"], r["seed"])]["arms"][f"single tau=inf {res_}"] for r in rr]
                print(f"   (199_, factor-2 error, balanced: {np.mean([min(q[a]['traced_err'] for a in A) for q in d]):.0f} um)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    t0 = time.time(); procs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    fn = HERE / "spotr_out" / "margin_scale.json"
    if len(sys.argv) > 2 and sys.argv[2] == "SUMMARY":
        res = json.load(open(fn))
    else:
        with get_context("spawn").Pool(procs) as pool:
            res = pool.map(job, [(p, b, s) for p in ["interspersed", "clone"] for b in ["off", "on"] for s in [46, 47, 48]])
        json.dump(res, open(fn, "w"), indent=1, default=float)
    summary(res)
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
