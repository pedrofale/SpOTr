"""Quick checks of SpOTr as written in spotr.tex's Methods (Pedro, 2026-11-02: "Let's always make sure we follow the writeup - that's the
truth. So given what we have there, it sounds to me like we have a great method that needs to work. Let's run our quick checks on HTAN and
PETracer to make sure it does").

Settings, each from spotr.tex's Methods (Overleaf 71d7239):
  deconvolution    cell2location's likelihood with one signature per clone, from the cell2location package's regression model;
  unassigned       cancer cells without a clone, when there are any, as their own reference type (untraced="free"; the engine's default is
                   still "tied"); HT260C1 v4 has none (every kept tumour nucleus is called), so it has no such type;
  mapping          each cancer cell placed once (totals "fixed"); capacities from the deconvolution's numbers of cancer cells;
  expression cost  the cosine distance between log-normalized profiles, against each spot's cancer part (cell_cost="resid",
                   cell_metric="cosine"), beta 1/2;
  structural term  clone labels: compaction (structure "compact"); a tree: distance matching on the scaled tree distances (structure "dm",
                   tree_target "linear"); no per-clone weights;
  alpha            the elbow: over the grid, the expression cost and the structural cost of the map, normalized between alpha 0 and 0.95, and
                   the alpha minimizing their sum (grid 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.95, a subset of 216_'s, to be quick).
Arms per data set: the tree over the grid, clone labels over the grid, and the tree with the cells shuffled within their clones at the tree's
elbow (the null).
Data: HT260C1 v4 (446_'s problem; the within-clone copy-number tree; purity of B vs A, B1 vs B2, B1b vs B1a, the authors' clones) and PEtracer
half:M3-4 (436_'s problem; the time-calibrated tree; PCC and JS against the in situ truth, sub-clade errors).
Saved per fit: the map, the abundances, the elbow's two costs and the scores; resumable.

Quick mode (`quick` as the first argument; Pedro: "these quick checks are just meant to sanity check that the method seems to do what it's
supposed to do - using the tree to obtain good mappings"): alpha 0 and the tree at alpha 0.2 (the elbow found earlier on both data sets, with
the squared Euclidean cost -- not a fresh elbow, the one departure from the Methods), plus clone labels at 0.2 on HT260C1; same keys and
files as the full sweep.

Usage: python 450_writeup_check.py [quick] [htan] [petracer] [petracer-visium[:<section>]]   (benchmark_env, GPU; writes spotr_out/writeup_check/)
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; O = HERE / "spotr_out"; OUT = O / "writeup_check"; sys.path.insert(0, str(ROOT / "src"))
import spotr.c2l_gw as M
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.95]
METHODS = dict(totals="fixed", cell_cost="resid", cell_metric="cosine", beta=0.5, tree_target="linear", capacity="share")   # share frozen 2026-11-02


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def shuffled(d, clone, seed=0):
    rng = np.random.default_rng(seed); pi = np.arange(len(clone))
    for k in np.unique(clone):
        i = np.where(clone == k)[0]; pi[i] = rng.permutation(i)
    return np.asarray(d, float)[np.ix_(pi, pi)]


def elbow(res, arm):
    """Methods: the alpha minimizing the sum of the expression and structural costs, each normalized between alpha 0 and alpha 0.95"""
    al = [a for a in ALPHAS if f"{arm} {a}" in res]
    if len(al) < len(ALPHAS):
        return None
    E = np.array([res[f"{arm} {a}"]["costs"]["expression"] for a in al]); S_ = np.array([res[f"{arm} {a}"]["costs"]["structure"] for a in al])
    e = (E - E[0]) / max(E[-1] - E[0], 1e-30); s = (S_ - S_[-1]) / max(S_[0] - S_[-1], 1e-30); return float(al[int(np.argmin(e + s))])


def run(case, args, clone, tree_d, Xcells, scorer, untraced, quick=False, labels_quick=True):
    OUT.mkdir(parents=True, exist_ok=True); fn = OUT / f"{case}.json"; res = json.load(open(fn)) if fn.exists() else {}; t0 = time.time()
    def fit(key, kw):
        if key in res:
            return
        r = M.fit(*args, verbose=False, clone=clone, mode="clones", X_cells=Xcells, untraced=untraced, **METHODS, **kw); phi = np.asarray(r["phi"], float)
        np.savez_compressed(OUT / f"{case}_{key.replace(' ', '_')}.npz", phi=phi.astype(np.float16), abund=np.asarray(M.abund(r["th"]), np.float32))
        res[key] = dict(scores=scorer(phi), costs=r["costs"]); json.dump(res, open(fn, "w"), indent=1)
        print(f"{case} {key} [{time.time() - t0:.0f}s]: " + json.dumps({k: round(v, 3) for k, v in res[key]["scores"].items()})
              + "  costs " + json.dumps({k: round(v, 4) for k, v in r["costs"].items()}), flush=True)
    if quick:
        fit("tree 0.0", dict(structure="dm", alpha_tree=0.0, tree_d=tree_d)); fit("tree 0.2", dict(structure="dm", alpha_tree=0.2, tree_d=tree_d))
        if labels_quick:
            fit("labels 0.2", dict(structure="compact", alpha_tree=0.2))
        return
    for a in ALPHAS:
        fit(f"tree {a}", dict(structure="dm", alpha_tree=a, tree_d=tree_d))
    for a in ALPHAS:
        fit(f"labels {a}", dict(structure="compact", alpha_tree=a))
    et, el = elbow(res, "tree"), elbow(res, "labels"); res["elbow"] = dict(tree=et, labels=el); json.dump(res, open(fn, "w"), indent=1)
    print(f"{case} elbows: tree {et}, labels {el}", flush=True)
    fit(f"shuffled {et}", dict(structure="dm", alpha_tree=et, tree_d=shuffled(tree_d, clone)))


QUICK = False


def htan():
    m446 = mod("m446", "446_htan_v4_run.py"); p = m446.problem(); sc = m446.scorer(p["z"], p["ti"], p["lab3"])
    run("HT260C1_v4", p["args"], p["c3"], p["dW"], p["Xc"][p["ti"]], sc, untraced="tied", quick=QUICK)      # no unassigned cancer cells: f = 1, no such type


def petracer(name="half:M3-4", case="PEtracer_M3-4"):
    """name half-visium:M3-4 (case PEtracer_M3-4_visium): the section measured with Visium spots (276_)"""
    m354, m276, m379 = mod("m354", "354_c2l_cgw_anova.py"), mod("m276", "276_petracer.py"), mod("m379", "379_c2l_reference.py"); m379.OUT = O / "petracer_comparison"
    st = name.replace(":", "_"); p = m354.problem("petracer", name); B = m276.build(name, 0); X = B["X"]; Xc = np.asarray(B["Xc"], float); tum = X["kind"] == 0; tr = tum & X["traced"]
    K = X["K"]; cl = X["cl"]; me = np.asarray(X["me_type"]); labs = np.array([f"me{e}" for e in me], dtype=object); labs[tum & ~X["traced"]] = "untraced"
    labs[tr] = [f"c{c}" if c < K - 1 else "pooled" for c in cl[tr]]; S = m379.signatures(Xc, labs, f"{st}_c2l"); types = sorted(set(me[~tum].tolist()))
    args = (p["Y"], np.stack([S[f"me{e}"] for e in types]), S["untraced"][None, :], p["fT"], np.stack([S[l] for l in labs[tr]]), p["kc"], p["D"], p["xy"])
    lev = M.nested_levels(p["d"], p["clone"], p["Kc"]); xyt = m354.truth_xy("petracer", name)
    def sc(P):
        s = p["score"](p["to_mass"](P)); out = {k: float(v) for k, v in s.items() if np.isscalar(v)}; out.update(M.subclade_errors(P, lev, xyt, p["xy"])); return out
    print(f"PEtracer {name}: {int(tr.sum())} traced cells, {int((tum & ~X['traced']).sum())} unassigned, {len(p['Y'])} spots, f = {float(np.ravel(p['fT'])[0]):.3f}", flush=True)
    run(case, args, p["clone"], np.asarray(p["d"], float), Xc[tr], sc, untraced="free", quick=QUICK, labels_quick=False)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    av = sys.argv[1:]; QUICK = bool(av) and av[0] == "quick"; av = av[1:] if QUICK else av
    for w in (av or ["htan", "petracer"]):
        if w.startswith("petracer-visium"):                                                      # petracer-visium[:<section>], default M3-4
            sec_ = w.split(":")[1] if ":" in w else "M3-4"                                     # the share capacity's fits under their own name
            petracer(f"half-visium:{sec_}", f"PEtracer_{sec_}_visium" + ("_share" if METHODS.get("capacity") == "share" else "")); continue
        {"htan": htan, "petracer": petracer}[w]()
    print("ALL DONE 450", flush=True)
