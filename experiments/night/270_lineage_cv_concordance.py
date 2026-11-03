"""Choosing lambda by cross-validation on the lineage, scored by concordance (268_ revised).  2026-10-27.
FIXED BEFORE RUNNING.
Pedro, 2026-10-26: "I don't see how using a fixed \\lambda makes sense in principle. It's a regularization strength hyperparameter, right?
Those are usually set by cross-validation."  2026-10-27: "Yes go ahead".
Why revised: 268_ (exp-unmatched-oct12 section 100) scored the held-out cells by the squared tree mismatch, sum G_im G_jm' (d_ij - D_mm')^2.
On its first world (W1, seed 46, SpOTr-DestVI, true tree) that score was lowest at lambda 0 (worst at 0.1, improving toward 2.5 but never
back to lambda 0's), while lambda 0.5 had subclone error 0.275 against lambda 0's 0.681.  The held-out cells are placed by expression alone,
weakly; as lambda sharpens the training cells' placement, the predicted distances E[D_ij] spread out in ways the held-out cells' placement
cannot follow, and a squared loss penalises that spread -- it rewards the vaguest map, a property of the loss, not evidence against the tree.
268_ was stopped (section 100).
The score here is scale-free: the concordance (Pearson correlation, over held-out x training pairs) between the tree distances d_ij and the
map's expected spatial distances E[D_ij] = G_i D G_j^T / (G_i. G_j.) -- the Mantel-type statistic used for lineage against space.  A vague map
gets no credit for being vague and no penalty for being confident; a tree that does not match the section should give a concordance near 0
whatever lambda.  Everything else is 268_'s:
  traced cells into F = 4 folds at random (stream 15e6 + seed); per fold the tree term on the training cells only (Omega = 0 on every pair
  with a held-out cell; tree scale calibrated on them); the held-out cells stay in the map; lambda path with continuation; the score per
  fold and lambda, averaged over folds.
  Rules: 'argmax' -- lambda maximising the mean concordance; '1-SE' -- the smallest lambda whose mean is within one standard error of the
  maximum (SE of the per-fold differences from the maximiser).  The map reported is the full-data map at the chosen lambda.
  268_'s squared score is computed alongside, as a read.
Method, trees, worlds as 268_: SpOTr v1, both plug-ins, lineage tree; true and scrambled (stream 14e6 + seed); lambda {0, 0.1, 0.3, 0.5, 1, 2.5};
microenvironment 'structured'; W1 tree_sd 0.3 pos_w 0; W2 tree_sd 0 pos_w 0.3 (seeds 46, 47); W3 tree_sd 0 pos_w 0 (seed 46).
Expectations stated before running (268_'s E116-E118 with the concordance; the first world, W1 seed 46, was seen under the squared score only):
  E119  true tree, W1 and W2: the argmax lambda is above 0 and within 0.05 of the truth-best subclone error, in at least 7 of 8
  E120  scrambled tree, W1 and W2: the 1-SE rule picks lambda 0 in at least 7 of 8
  E121  true tree, W1 and W2: the CV-chosen map (argmax) has a lower subclone error than the tool alone in at least 7 of 8
  read: argmax against 1-SE; the scrambled tree's concordance level (expected near 0); W3; CV's lambda against the fixed 0.5; the squared score.
Deviation (2026-10-27, after W1 seed 46's DestVI arms and its Tangram true-tree arm): Pedro, on the guard: "It's ok if it doesn't guard
against a wrong tree".  The CV only has to choose lambda; the scrambled tree's arms (half the GPU time) are dropped from the remaining worlds;
E120 is not evaluated beyond W1 seed 46 and the Mantel guard (271_) stays a read.
Extension (2026-10-27, after the CV results, exp-unmatched-oct12 section 102): the S1 / S2 headline rerun on v1 with the structured
microenvironment -- W4-W6 S1's other levels (tree_sd 0.15, 0.6, 1.2, pos_w 0), W7 S2's pos_w 1 (tree_sd 0), and W3 seed 47; the same job,
true tree only; lambda chosen afterwards by 272_ (anchored CV) from the saved fold maps; the fixed 0.5 and the truth-best read off the same path.
One job per world, saved after each tool and tree to spotr_out/lineage_cv2/<world>_<seed>.json; the traced cells' fold maps (rows normalised,
float16) to spotr_out/lineage_cv2/maps/, so other scores can be read without the GPU; resumable.  Usage: python 270_lineage_cv_concordance.py [SUMMARY]
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / "spotr_out" / "lineage_cv2"
WORLDS = {"W1": dict(tree_sd=0.3, pos_w=0.0), "W2": dict(tree_sd=0.0, pos_w=0.3), "W3": dict(tree_sd=0.0, pos_w=0.0),
          "W4": dict(tree_sd=0.15, pos_w=0.0), "W5": dict(tree_sd=0.6, pos_w=0.0), "W6": dict(tree_sd=1.2, pos_w=0.0), "W7": dict(tree_sd=0.0, pos_w=1.0)}
RUNS = [("W1", 46), ("W2", 46), ("W3", 46), ("W1", 47), ("W2", 47),
        ("W4", 46), ("W5", 46), ("W6", 46), ("W7", 46), ("W3", 47), ("W4", 47), ("W5", 47), ("W6", 47), ("W7", 47)]
TOOLS, TREES, PATH, N_FOLDS = ["destvi", "tangram"], ["true"], [0.0, 0.1, 0.3, 0.5, 1.0, 2.5], 4


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def scores(G, ti, hid, d, D):
    """held-out x training pairs: (concordance of d_ij with the expected map distance, 268_'s squared mismatch per unit mass)"""
    H, T = ti[hid], ti[~hid]; GH, GT = np.asarray(G[H], float), np.asarray(G[T], float); dd = d[np.ix_(hid, ~hid)]; a, b = GH.sum(1), GT.sum(1)
    ED = (GH @ D) @ GT.T / np.maximum(np.outer(a, b), 1e-300)
    sq = (a @ (dd * dd) @ b - 2.0 * np.sum(dd * ED * np.outer(a, b)) + GH.sum(0) @ (D * D) @ GT.sum(0)) / (a.sum() * b.sum())
    return float(np.corrcoef(dd.ravel(), ED.ravel())[0, 1]), float(sq)


def rules(per_fold):
    """per_fold: folds x lambdas of the concordance -> (argmax lambda, 1-SE lambda)"""
    S = np.asarray(per_fold); i0 = int(np.argmax(S.mean(0))); diff = S[:, [i0]] - S
    se = diff.std(0, ddof=1) / np.sqrt(S.shape[0]); within = [i for i in range(S.shape[1]) if diff.mean(0)[i] <= se[i]]
    return PATH[i0], PATH[min(within)]


def job(wname, seed, f):
    import warnings; warnings.filterwarnings("ignore")
    core, m239, m267, m268 = mod("core", "250_spotr_core.py"), mod("m239", "239_deconv_fgw.py"), mod("m267", "267_v1_check.py"), mod("m268", "268_lineage_cv.py"); t0 = time.time()
    NB, X, Xc, t_f, m189 = m268.build(seed, **WORLDS[wname]); n = len(X["cl"]); tum = X["kind"] == 0; tr = tum & X["traced"]; Y = X["Sc"]; Nb = m239.window(X)
    K, nT = X["K"], X["nT"]; D = np.asarray(X["D"], float)
    labels = np.array([f"c{c}" if t else ("untraced" if k == 0 else f"me{e}") for k, c, t, e in zip(X["kind"], X["cl"], tr, X["me_type"])])
    mass = lambda G: (np.asarray(G[tr], float).T @ np.eye(K)[X["cl"][tr]])
    out = json.load(open(f)) if f.exists() else dict(world=wname, seed=seed, **WORLDS[wname], arms={}, cv={}, masses={}, timing={})
    fit = core.fit_destvi(Xc, labels, Y, seed=0); names, own = fit["names"], fit["own"]; q = fit["library"] / fit["library"].sum()
    t_d = own[:, [names.index(l) for l in names if l.startswith("c") or l == "untraced"]].sum(1)
    Tt = core.tangram_own(Xc, Y, q, seed=0)
    if not out["arms"]:
        Pd = np.zeros((len(Y), K))
        for k in range(K):
            if f"c{k}" in names:
                Pd[:, k] = own[:, names.index(f"c{k}")] * fit["library"]
        out["arms"]["DestVI alone"] = m239.readout(None, X, Nb, m189, P_clone=Pd); out["masses"]["DestVI alone"] = Pd.tolist()
        out["arms"]["Tangram alone"] = m239.readout(Tt / n, X, Nb, m189); out["masses"]["Tangram alone"] = mass(Tt).tolist()
        import ot
        To = ot.emd(np.full(n, 1.0 / n), X["q"], X["M"]); out["arms"]["exact OT"] = m239.readout(To, X, Nb, m189); out["masses"]["exact OT"] = mass(To).tolist()
        Tf = mod("m89", "89_growth_example_figure.py").oracle(X); out["arms"]["floor"] = m239.readout(Tf, X, Nb, m189); out["masses"]["floor"] = mass(Tf).tolist()
        gm = np.bincount(X["cl"][tr], minlength=K).astype(float); out["arms"]["global mix"] = m239.readout(None, X, Nb, m189, P_clone=np.outer(X["Ncl"].sum(1) > 0, gm))
        json.dump(out, open(f, "w"), default=float)
    ti, dt = m239.tree_dists(X, NB, t_f, q * t_d); d_true = dt["lineage"]
    perm = np.random.default_rng(14 * 10 ** 6 + seed).permutation(len(ti)); dtree = {"true": d_true, "scrambled": d_true[np.ix_(perm, perm)]}
    fold = np.random.default_rng(15 * 10 ** 6 + seed).integers(0, N_FOLDS, len(ti)); (OUT / "maps").mkdir(exist_ok=True)
    for tool in TOOLS:
        if tool == "destvi":
            fit["kappa"] = fit["V"][:, :-1].sum(1) / q; R, E = core.expression_term("destvi", fit, Xc, Y); t_hat = t_d
            ck, sh = core.destvi_shares(fit, tr, tum, X["me_type"], nT)
        else:
            R, E = core.expression_term("tangram", None, Xc, Y); t_hat = Tt[tum].sum(0) / np.maximum(Tt.sum(0), 1e-300); ck, sh = m267.map_shares(Tt, tr, tum, X["me_type"], nT)
        kk, caps = core.kinds_from_shares(ck, sh, q); prob = core.Problem(R, E, q, t_hat, tr, tum, D, kind=kk, caps=caps)
        for tree in TREES:
            key = f"{tool}|{tree}"
            if key in out["cv"]:
                continue
            d = dtree[tree]; conc, sqs, maps = [], [], {}
            for k in range(N_FOLDS):
                hid = fold == k; A_fn, c = prob.tree("cell", idx=ti[~hid], d=d[np.ix_(~hid, ~hid)]); path = prob.solve_path(A_fn, c, lambdas=PATH, keep=True); rc, rs = [], []
                for l in PATH:
                    G = path[l]["G"]; cc, ss = scores(G, ti, hid, d, D) if path[l]["finite"] else (np.nan, np.nan); rc.append(cc); rs.append(ss)
                    Gt = np.asarray(G[ti], float); maps[f"fold{k}|{l}"] = (Gt / np.maximum(Gt.sum(1, keepdims=True), 1e-300)).astype(np.float16)
                conc.append(rc); sqs.append(rs)
            A_fn, c = prob.tree("cell", idx=ti, d=d); path = prob.solve_path(A_fn, c, lambdas=PATH, keep=True); full = {}
            for lam, v in path.items():
                full[str(lam)] = dict(fit=v["fit"], tree=v["tree"], finite=v["finite"], **(m239.readout(v["G"], X, Nb, m189) if v["finite"] else {}))
                if v["finite"]:
                    out["masses"][f"{key}|{lam}"] = mass(v["G"]).tolist()
            np.savez_compressed(OUT / "maps" / f"{wname}_{seed}_{tool}_{tree}.npz", ti=ti, fold=fold, d=d.astype(np.float32), **maps)
            am, se1 = rules(conc); best = min(full, key=lambda l: full[l].get("clone_tv", np.inf))
            out["cv"][key] = dict(concordance=conc, squared=sqs, argmax=am, one_se=se1, truth_best=float(best), full=full); out["timing"][key] = time.time() - t0
            json.dump(out, open(f, "w"), default=float)
            mc = np.asarray(conc).mean(0)
            print(f"  {wname} seed {seed} {tool} {tree} tree: concordance " + ", ".join(f"{l:g}: {s:.3f}" for l, s in zip(PATH, mc)) +
                  f" -> argmax {am:g} ({full[str(am)]['clone_tv']:.3f}), 1-SE {se1:g} ({full[str(se1)]['clone_tv']:.3f}); lambda 0 {full['0.0']['clone_tv']:.3f}, "
                  f"0.5 {full['0.5']['clone_tv']:.3f}, truth-best {best} ({full[best]['clone_tv']:.3f})  [{time.time() - t0:.0f}s]", flush=True)
    out["done"] = True; out["timing"]["total"] = time.time() - t0; json.dump(out, open(f, "w"), default=float)


def summary():
    rs = [json.load(open(f)) for f in sorted(OUT.glob("W*_*.json"))]
    for r in sorted(rs, key=lambda r: (r["world"], r["seed"])):
        for key, v in r["cv"].items():
            fu = v["full"]; tool = key.split("|")[0]; alone = r["arms"]["DestVI alone" if tool == "destvi" else "Tangram alone"]["clone_tv"]
            print(f"{r['world']} seed {r['seed']} {key:18s}: concordance " + " ".join(f"{x:.3f}" for x in np.asarray(v["concordance"]).mean(0)) +
                  f" | argmax {v['argmax']:g} -> {fu[str(v['argmax'])]['clone_tv']:.3f}; 1-SE {v['one_se']:g} -> {fu[str(v['one_se'])]['clone_tv']:.3f}; "
                  f"0.5 -> {fu['0.5']['clone_tv']:.3f}; lambda 0 -> {fu['0.0']['clone_tv']:.3f}; truth-best {v['truth_best']:g} -> {fu[str(v['truth_best'])]['clone_tv']:.3f}; "
                  f"tool alone {alone:.3f}; floor {r['arms']['floor']['clone_tv']:.3f}")
    oriented = [r for r in rs if r["world"] in ("W1", "W2")]
    cv = lambda r, t, tree: r["cv"].get(f"{t}|{tree}")
    e119 = [cv(r, t, "true")["argmax"] > 0 and cv(r, t, "true")["full"][str(cv(r, t, "true")["argmax"])]["clone_tv"]
            - cv(r, t, "true")["full"][str(cv(r, t, "true")["truth_best"])]["clone_tv"] <= 0.05 for r in oriented for t in TOOLS if cv(r, t, "true")]
    e120 = [cv(r, t, "scrambled")["one_se"] == 0.0 for r in oriented for t in TOOLS if cv(r, t, "scrambled")]
    e121 = [cv(r, t, "true")["full"][str(cv(r, t, "true")["argmax"])]["clone_tv"] < r["arms"]["DestVI alone" if t == "destvi" else "Tangram alone"]["clone_tv"]
            for r in oriented for t in TOOLS if cv(r, t, "true")]
    print(f"E119: {sum(e119)}/{len(e119)}   E120: {sum(e120)}/{len(e120)}   E121: {sum(e121)}/{len(e121)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "SUMMARY":
        summary(); sys.exit()
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for w, s in RUNS:
        f = OUT / f"{w}_{s}.json"
        if f.exists() and json.load(open(f)).get("done"):
            continue
        job(w, s, f); print(f"{w} seed {s}: done ({time.time() - t0:.0f}s total)", flush=True)
    summary()
