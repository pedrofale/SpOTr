"""Choosing lambda by cross-validation on the lineage: hold out traced cells, predict their tree distances.  2026-10-27.
FIXED BEFORE RUNNING.
Pedro, 2026-10-26: "I don't see how using a fixed \\lambda makes sense in principle. It's a regularization strength hyperparameter, right?
Those are usually set by cross-validation."  2026-10-27, on the order v1 check / lineage cross-validation / rerun S1-S2: "Yes go ahead".
Why the lineage and not the spots: held-out spots barely see which clone goes where (S3, 261_: the guard followed the seed, not the tree,
E108 3/8, E109 2/4); the spots decide how many cells of each state, the tree which ones.  So the data held out here are the lineage's.
The procedure (what a user can run, no truth used):
  traced cells into F = 4 folds at random (own stream 15e6 + seed); for fold f the tree term is the lineage's on the training cells only
  (Omega = 0 on every pair involving a held-out cell; 250_ Problem.tree on the training rows, so the tree scale is calibrated on them);
  the held-out cells stay in the map, placed by expression and the capacities; the map is fitted along the lambda path;
  held-out score at each lambda: the tree mismatch on held-out x training pairs, sum_ij sum_mm' G_im G_jm' (d_ij - D_mm')^2 over
  sum_i G_i. sum_j G_j. (a mean squared error in um^2: how well the map predicts the held-out cells' tree distances to the rest);
  summed over the folds (every traced cell held out once).
  Rules: 'argmin' -- lambda minimising the summed score; '1-SE' -- the smallest lambda whose score is within one standard error of the
  minimum (the SE of the per-fold differences from the minimiser, over the folds), lambda 0 being the simplest model (no tree).
  The map reported is the full-data map (all traced cells in the tree term) at the chosen lambda.
The same score serves as the guard against a tree that does not match the section: with a scrambled tree the held-out cells' expression
placement should not agree with where the tree puts their supposed relatives, so the score should not improve on lambda 0.
Method: SpOTr v1 (decision-capacities-from-tool; 250_ kinds_from_shares; Tangram's shares from its own map, 267_ map_shares), lineage
tree cell-level on the traced block, both plug-ins; lambda over {0, 0.1, 0.3, 0.5, 1, 2.5} with continuation.
Trees: the true lineage; the same tree scrambled across the traced cells (a fixed permutation, stream 14e6 + seed, as 261_).
Worlds (239_'s design, the microenvironment 'structured'; seeds 46, 47; seed 46 first):
  W1 'clone expression'  tree_sd 0.3, pos_w 0   (S1's: radial-only states, clone-specific expression)
  W2 'oriented states'   tree_sd 0,   pos_w 0.3 (S2's: no clone-specific expression, states that vary across the section)
  W3 'no orientation'    tree_sd 0,   pos_w 0   (radial-only states, no clone signal: the tree cannot be oriented; seed 46 only)
Expectations stated before running:
  E116  true tree, W1 and W2: the argmin lambda is above 0 and its subclone error within 0.05 of the truth-best on the grid, in at least
        7 of 8 (2 worlds x 2 seeds x 2 tools)
  E117  scrambled tree, W1 and W2: the 1-SE rule picks lambda 0 in at least 7 of 8
  E118  true tree, W1 and W2: the CV-chosen map (argmin) has a lower subclone error than the tool alone in at least 7 of 8
  read: the argmin against the 1-SE rule; the scrambled tree's argmin; W3; CV's lambda against the fixed 0.5; the shape of the score.
One job per world, saved after each tool and tree to spotr_out/lineage_cv/<world>_<seed>.json with the per-fold scores and the full-data
maps' per-spot clone masses (so the figure needs no GPU); resumable.  Usage: python 268_lineage_cv.py [SUMMARY]
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / "spotr_out" / "lineage_cv"
WORLDS = {"W1": dict(tree_sd=0.3, pos_w=0.0), "W2": dict(tree_sd=0.0, pos_w=0.3), "W3": dict(tree_sd=0.0, pos_w=0.0)}
RUNS = [("W1", 46), ("W2", 46), ("W3", 46), ("W1", 47), ("W2", 47)]
TOOLS, TREES, PATH, N_FOLDS = ["destvi", "tangram"], ["true", "scrambled"], [0.0, 0.1, 0.3, 0.5, 1.0, 2.5], 4


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def build(seed, tree_sd, pos_w):
    m189, m208 = mod("m189", "189_me_sim.py"), mod("m208", "208_neighbour_sim.py")
    NB = m208.simulate_nb(seed, pattern="structured", margin=1.0, tree_sd=tree_sd, pos_w=pos_w)
    X = m208.problem_nb(NB, "all", "single", capture=dict(keep=0.5), dissoc_frac=1.0)
    t_f = float(NB["section"]["sim"]["ntime"][2 * 127 - 1]); Xc = NB["region"]["counts"][X["mapped"]].astype(float)
    return NB, X, Xc, t_f, m189


def heldout_score(G, ti, hid, d, D):
    """the tree mismatch on held-out x training pairs under the map, per unit mass (um^2)"""
    H, T = ti[hid], ti[~hid]; GH, GT = np.asarray(G[H], float), np.asarray(G[T], float); dd = d[np.ix_(hid, ~hid)]; a, b = GH.sum(1), GT.sum(1)
    num = a @ (dd * dd) @ b - 2.0 * np.sum(dd * ((GH @ D) @ GT.T)) + GH.sum(0) @ (D * D) @ GT.sum(0)
    return float(num / (a.sum() * b.sum()))


def rules(per_fold):
    """per_fold: folds x lambdas -> (argmin lambda, 1-SE lambda)"""
    S = np.asarray(per_fold); tot = S.sum(0); i0 = int(np.argmin(tot)); diff = S - S[:, [i0]]
    se = diff.std(0, ddof=1) / np.sqrt(S.shape[0]); within = [i for i in range(S.shape[1]) if diff.mean(0)[i] <= se[i]]
    return PATH[i0], PATH[min(within)]


def job(wname, seed, f):
    import warnings; warnings.filterwarnings("ignore")
    core, m239, m267 = mod("core", "250_spotr_core.py"), mod("m239", "239_deconv_fgw.py"), mod("m267", "267_v1_check.py"); t0 = time.time()
    NB, X, Xc, t_f, m189 = build(seed, **WORLDS[wname]); n = len(X["cl"]); tum = X["kind"] == 0; tr = tum & X["traced"]; Y = X["Sc"]; Nb = m239.window(X)
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
    fold = np.random.default_rng(15 * 10 ** 6 + seed).integers(0, N_FOLDS, len(ti))
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
            d = dtree[tree]; per_fold = []
            for k in range(N_FOLDS):
                hid = fold == k; A_fn, c = prob.tree("cell", idx=ti[~hid], d=d[np.ix_(~hid, ~hid)]); path = prob.solve_path(A_fn, c, lambdas=PATH, keep=True)
                per_fold.append([heldout_score(path[l]["G"], ti, hid, d, D) if path[l]["finite"] else np.nan for l in PATH])
            A_fn, c = prob.tree("cell", idx=ti, d=d); path = prob.solve_path(A_fn, c, lambdas=PATH, keep=True); full = {}
            for lam, v in path.items():
                full[str(lam)] = dict(fit=v["fit"], tree=v["tree"], finite=v["finite"], **(m239.readout(v["G"], X, Nb, m189) if v["finite"] else {}))
                if v["finite"]:
                    out["masses"][f"{key}|{lam}"] = mass(v["G"]).tolist()
            am, se1 = rules(per_fold); best = min(full, key=lambda l: full[l].get("clone_tv", np.inf))
            out["cv"][key] = dict(per_fold=per_fold, argmin=am, one_se=se1, truth_best=float(best), full=full); out["timing"][key] = time.time() - t0
            json.dump(out, open(f, "w"), default=float)
            tot = np.asarray(per_fold).sum(0)
            print(f"  {wname} seed {seed} {tool} {tree} tree: held-out score minus lambda 0's " + ", ".join(f"{l:g}: {s - tot[0]:+.4g}" for l, s in zip(PATH, tot)) +
                  f" -> argmin {am:g} ({full[str(am)]['clone_tv']:.3f}), 1-SE {se1:g} ({full[str(se1)]['clone_tv']:.3f}); lambda 0 {full['0.0']['clone_tv']:.3f}, "
                  f"0.5 {full['0.5']['clone_tv']:.3f}, truth-best {best} ({full[best]['clone_tv']:.3f})  [{time.time() - t0:.0f}s]", flush=True)
    out["done"] = True; out["timing"]["total"] = time.time() - t0; json.dump(out, open(f, "w"), default=float)


def summary():
    rs = [json.load(open(f)) for f in sorted(OUT.glob("*.json"))]
    for r in sorted(rs, key=lambda r: (r["world"], r["seed"])):
        for key, v in r["cv"].items():
            fu = v["full"]; tool = key.split("|")[0]; alone = r["arms"]["DestVI alone" if tool == "destvi" else "Tangram alone"]["clone_tv"]
            print(f"{r['world']} seed {r['seed']} {key:18s}: argmin {v['argmin']:g} -> {fu[str(v['argmin'])]['clone_tv']:.3f}; 1-SE {v['one_se']:g} -> {fu[str(v['one_se'])]['clone_tv']:.3f}; "
                  f"0.5 -> {fu['0.5']['clone_tv']:.3f}; lambda 0 -> {fu['0.0']['clone_tv']:.3f}; truth-best {v['truth_best']:g} -> {fu[str(v['truth_best'])]['clone_tv']:.3f}; "
                  f"tool alone {alone:.3f}; floor {r['arms']['floor']['clone_tv']:.3f}")
    oriented = [r for r in rs if r["world"] in ("W1", "W2")]
    e116 = [r["cv"][f"{t}|true"]["argmin"] > 0 and r["cv"][f"{t}|true"]["full"][str(r["cv"][f"{t}|true"]["argmin"])]["clone_tv"]
            - r["cv"][f"{t}|true"]["full"][str(r["cv"][f"{t}|true"]["truth_best"])]["clone_tv"] <= 0.05 for r in oriented for t in TOOLS if f"{t}|true" in r["cv"]]
    e117 = [r["cv"][f"{t}|scrambled"]["one_se"] == 0.0 for r in oriented for t in TOOLS if f"{t}|scrambled" in r["cv"]]
    e118 = [r["cv"][f"{t}|true"]["full"][str(r["cv"][f"{t}|true"]["argmin"])]["clone_tv"] < r["arms"]["DestVI alone" if t == "destvi" else "Tangram alone"]["clone_tv"]
            for r in oriented for t in TOOLS if f"{t}|true" in r["cv"]]
    print(f"E116: {sum(e116)}/{len(e116)}   E117: {sum(e117)}/{len(e117)}   E118: {sum(e118)}/{len(e118)}")


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
