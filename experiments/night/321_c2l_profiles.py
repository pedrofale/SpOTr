"""319_'s model with three ways to give each traced cell its profile, and a confirmation on data no variant has seen.  2026-10-29.
FIXED BEFORE RUNNING.
Pedro, 2026-10-29, after 319_ (the clone-mean control beat the pre-registered cell-profile method, whose profiles were smoothed across
clones): "Go ahead and show me the results of this" -- smoothing each cell only within its own clone.
Profiles (the only change; 319_'s fit() unchanged):  cell = 319_'s (mean of the 15 nearest traced cells, any clone);  within = the mean of
the cell's nearest cells within its own clone (15, or the clone's size), in the same 30-PC space;  clone = the clone's mean (319_'s control).
Spread settings: free (a0 = 1) and prior (a0 = 10 at the notree map's level), as 319_.
Part 1 (319_'s six problems, seed 0 / 46): within_free, within_prior, clone_free (the rest is in spotr_out/spotr_c2l).
Part 2 (confirmation, unseen data): PEtracer half M3-4, M3-1, M3-2 with seed 1 (a new random half-split), serial with seed 1 (a new
dissociated sample), simulation worlds W1, W2 with seed 47; arms notree, cell/within/clone x free/prior.
Expectations stated before running:
  E158  part 1: within_free >= clone_prior (319_'s control) on per-clade PCC in >= 3 of 4 PEtracer designs, and its mean dominant clade
        share on M3-4 in [0.45, 0.70]
  E159  part 2: the variant E158 selects (within_free if E158 passes, else clone_prior) beats notree and exact OT in all 4 PEtracer designs
        and is within 0.02 of the best of the six profile x spread arms in >= 3 of 4
  E160  part 2: that variant's learned spreads -- if it is a free arm -- against the truth: Spearman >= 0.3 and median ratio in [0.5, 2]
        in >= 3 of 4 PEtracer designs
  read  the simulation worlds; the maps
Usage: python 321_c2l_profiles.py    (writes spotr_out/spotr_c2l_v2/<problem>.json)
"""
import os, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / "spotr_out" / "spotr_c2l_v2"; OLD = HERE / "spotr_out" / "spotr_c2l"
PART1 = [("sim", "W1", 46), ("sim", "W2", 46), ("petracer", "half:M3-4", 0), ("petracer", "half:M3-1", 0), ("petracer", "half:M3-2", 0), ("petracer", "serial:M3-1>M3-2", 0)]
PART2 = [("petracer", "half:M3-4", 1), ("petracer", "half:M3-1", 1), ("petracer", "half:M3-2", 1), ("petracer", "serial:M3-1>M3-2", 1), ("sim", "W1", 47), ("sim", "W2", 47)]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def profiles(Xt, clone, k=15):
    """cell: 319_'s across-clone kNN means; within: kNN means within each clone (cells without a clone: across); clone: clone means"""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    L = np.log1p(Xt / np.maximum(Xt.sum(1, keepdims=True), 1.0) * 1e4); Z = PCA(n_components=min(30, L.shape[1] - 1), random_state=0).fit_transform(L - L.mean(0))
    idx = NearestNeighbors(n_neighbors=min(k, len(Xt))).fit(Z).kneighbors(Z, return_distance=False); cell = Xt[idx].mean(1)
    within, clm = cell.copy(), cell.copy()
    for c in np.unique(clone[clone >= 0]):
        r = np.where(clone == c)[0]; kk = min(k, len(r)); j = NearestNeighbors(n_neighbors=kk).fit(Z[r]).kneighbors(Z[r], return_distance=False)
        within[r] = Xt[r][j].mean(1); clm[r] = Xt[r].mean(0)
    return dict(cell=cell, within=within, clone=clm)


def run_problem(src, name, seed, arms, f):
    import warnings; warnings.filterwarnings("ignore")
    from scipy.stats import spearmanr
    m319, m316, f289 = mod("m319", "319_spotr_c2l.py"), mod("m316", "316_spotr_vb.py"), mod("f289", "289_field_metrics.py")
    if src == "sim":
        m268, m239, m189 = mod("m268", "268_lineage_cv.py"), mod("m239", "239_deconv_fgw.py"), mod("m189", "189_me_sim.py")
        NB, X, Xc, t_f, _ = m268.build(seed, **mod("m270", "270_lineage_cv_concordance.py").WORLDS[name]); Y = X["Sc"]; Nb = m239.window(X)
    else:
        P = mod("m276", "276_petracer.py").build(name, seed); X, Xc, Y = P["X"], P["Xc"], P["Y"]; Nb = m239 = m189 = None
    Xc, Y = np.asarray(Xc, float), np.asarray(Y, float); tum = X["kind"] == 0; tr = tum & X["traced"]; K = X["K"]; cl = X["cl"][tr]; clone = np.where(cl < K - 1, cl, -1)
    D = np.asarray(X["D"], float); Ncl = X["Ncl"]; me = np.asarray(X["me_type"]); fT = float(tr.sum() / max(tum.sum(), 1))
    types = sorted(set(me[~tum].tolist())); Gme = np.stack([Xc[(~tum) & (me == e)].mean(0) for e in types]); gu = Xc[tum & ~tr].mean(0) if (tum & ~tr).any() else Xc[tr].mean(0)
    prof = profiles(Xc[tr], clone); s2_truth = m316.truth_spread(Ncl, D, K)[: K - 1]; onehot = np.eye(K)[cl]
    out = json.load(open(f)) if f.exists() else dict(meta=dict(problem=name, src=src, seed=seed, traced=int(tr.sum()), spots=len(Y), s2_truth=s2_truth.tolist()), arms={}, masses={})
    has = Ncl.sum(1) > 0; nrm = lambda A: A / np.maximum(A.sum(1, keepdims=True), 1e-300)
    def score(Pm):
        r = f289.scores(Pm, Ncl); r["dominant_share"] = float(nrm(Pm[has]).max(1).mean())
        if src == "sim":
            r.update({k: v for k, v in m239.readout(None, X, Nb, m189, P_clone=Pm).items() if k in ("clone_tv", "clone_tv_win")})
        return r
    if "exact OT" not in out["masses"]:                                          # comparators that need no tool fit
        import ot
        n = len(X["cl"]); To = ot.emd(np.full(n, 1.0 / n), X["q"], X["M"]); out["masses"]["exact OT"] = (To[tr].T @ onehot).tolist(); out["arms"]["exact OT"] = dict(scores=score(To[tr].T @ onehot))
        if np.isfinite(X["xy_true"]).all():
            Tf = mod("m89", "89_growth_example_figure.py").oracle(X); out["masses"]["floor"] = (Tf[tr].T @ onehot).tolist(); out["arms"]["floor"] = dict(scores=score(Tf[tr].T @ onehot))
    old = OLD / f.name; level = None
    for src_json in [out, json.load(open(old)) if old.exists() and seed in (0, 46) else None]:
        if src_json and "notree" in src_json["arms"] and "emp" in src_json["arms"]["notree"]:
            e = np.asarray(src_json["arms"]["notree"]["emp"], float); level = float(np.nanmedian(e[np.isfinite(e)])); break
    t0 = time.time()
    for arm in arms:
        if arm in out["arms"]:
            continue
        pv, sp = arm.split("_") if arm != "notree" else ("cell", "notree")
        res = m319.fit(Y, Gme, gu, prof[pv], fT, clone, K - 1, D, sp, level=level, s2_truth=s2_truth)
        if arm == "notree":
            e = res["emp"]; level = float(np.nanmedian(e[np.isfinite(e)]))
        Pm = res["phi"].T @ onehot; s = score(Pm); ok = np.isfinite(s2_truth) & np.isfinite(res["s2"])
        s.update(spread_spearman=float(spearmanr(res["s2"][ok], s2_truth[ok])[0]) if ok.sum() > 2 else None, spread_ratio_median=float(np.median(res["s2"][ok] / s2_truth[ok])) if ok.sum() else None)
        out["arms"][arm] = dict(scores=s, s2=[float(x) for x in res["s2"]], emp=[float(x) for x in res["emp"]], trace=res["trace"]); out["masses"][arm] = Pm.tolist(); json.dump(out, open(f, "w"), default=float)
        print(f"  {name} s{seed} {arm:13s}: PCC {s['PCC']:+.3f}, dominant {s['dominant_share']:.2f}" + (f", subclone error {s['clone_tv']:.3f}" if "clone_tv" in s else "")
              + (f", spreads Spearman {s['spread_spearman']:+.2f}, ratio {s['spread_ratio_median']:.2f}" if s.get("spread_spearman") is not None else "") + f"  [{time.time() - t0:.0f}s]", flush=True)
    out["done"] = True; json.dump(out, open(f, "w"), default=float)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for part, probs, arms in [(1, PART1, ["within_free", "within_prior", "clone_free"]),
                              (2, PART2, ["notree", "clone_prior", "clone_free", "within_prior", "within_free", "cell_prior", "cell_free"])]:
        for src, name, seed in probs:
            f = OUT / f"{src}_{name.replace(':', '_').replace('>', '_to_')}_{seed}.json"
            if f.exists() and json.load(open(f)).get("done"):
                continue
            print(f"part {part}, {name} seed {seed}: start", flush=True); run_problem(src, name, seed, arms, f); print(f"{name} seed {seed}: done ({time.time() - t0:.0f}s total)", flush=True)
