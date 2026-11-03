"""The minimal check of the new SpOTr variant on the simulations, HTAN and PEtracer (Pedro, 2026-11-01: "what is the minimal set of runs we
need to check that SpOTr is working on these data sets?" -> "Ok go ahead with that").

The variant, frozen before running (exp-unmatched-oct12 sections 160-163):
  - per-cell profiles: each traced cell's 15-neighbour mean within its label (spotr.c2l_gw.profiles "within"; the simulations' and PEtracer's
    cell-level inputs already are these);
  - spread="switch": spreads from expected distances for the first 8 of 16 outer rounds, then from the cells' expected positions;
  - the tree at the highest resolution available: PEtracer the lineage tree; HT260C1 the nuclei's copy-number tree (Ward within each of
    A / B1 / B2 on the nuclei's copy-number distances d_cn, cophenetic, scaled under the label tree); K3 a copy-number tree from the multiome
    RNA and ATAC arm scores (413_; both log-depth-regressed and standardised, autosomes, Ward within A and B), with no between-clade term (two
    main clones: no relative distances between them, Pedro's rule);
  - spread_prior=1 (compact finest groups) only on HT260C1 (colorectal glands); off elsewhere (Pedro did not ask for it).
Reference "current SpOTr" (the method of record, actual counts): HTAN subclone-only profiles (HT260C1 A / B1 / B2 clone labels from 390_ at
p = 1, lineage 386_; K3 clone labels 387_); PEtracer subclone-only clone labels (400_, section 155); simulations 361_'s one-phase fits (refitted
today on W1 / W2: identical).
Scores: simulations per-spot subclone error (clone_tv) and per-spot sub-clade error at tree levels 1-3; HTAN balanced purity against the
held-out-gene references (415_'s contrasts: B vs A, B1 vs B2, B1b vs B1a with the sub-split hidden; K3 B vs A, A2 vs A1) and against the
authors' clones (406_); PEtracer per-clade PCC (289_), per-spot sub-clade error, and in the half designs each traced cell's expected distance to
its own position (the half designs split one section's cells at random, so this is the floor 276_ describes).

FIXED BEFORE RUNNING (2026-11-01):
R1. Simulations W1-W7 (seed 46): the variant's clone labels and lineage form each no worse than the method's clone_tv by more than 0.01 in
    every world.
R2. HT260C1 (labels A / B1 / B2 only): (a) the lineage form's B1b vs B1a balanced purity >= 0.80; (b) the lineage form >= the variant's clone
    labels + 0.03 on B1b vs B1a; (c) B vs A and B1 vs B2 each no lower than the current SpOTr's clone labels by more than 0.02.
R3. K3: B vs A no lower than the current SpOTr's clone labels by more than 0.02, both arms.
R4. PEtracer: (a) the variant's clone labels and lineage form each >= section 155's subclone-only clone labels - 0.01 on mean PCC in every
    design; (b) lineage form >= the variant's clone labels on mean PCC in every design; (c) half designs: the lineage form's mean per-cell
    expected distance below exact OT's and Tangram's.
Reported without a prediction: sub-clade errors, the prior's share on HT260C1 (lineage with and without it), K3 A2 vs A1, the authors' clones.

Usage: python 419_check_new_variant.py [sims ht260c1 k3 petracer]   (benchmark_env, GPU; writes spotr_out/new_variant/)
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; O = HERE / "spotr_out"; OUT = O / "new_variant"; sys.path.insert(0, str(ROOT / "src"))
import spotr.c2l_gw as M
SW = dict(spread="switch")


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def jload(fn):
    return json.load(open(fn)) if fn.exists() else {}


def save_fit(d, key, r):
    d.mkdir(parents=True, exist_ok=True); np.save(d / f"{key}_phi.npy", np.asarray(r["phi"], np.float16)); np.save(d / f"{key}_abund.npy", np.asarray(M.abund(r["th"]), np.float32))
    json.dump(dict(trace=r["trace"], weights=r.get("weights", {})), open(d / f"{key}.json", "w"), default=float)


def within_tree(dist_within, labels, Dk, ward_on="dist"):
    """Ward within each label (on a distance matrix, or on observations), cophenetic, scaled so the deepest within-label distance equals the
    shallowest between-label one; between labels the label tree"""
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import squareform
    d = Dk[np.ix_(labels, labels)].astype(float).copy(); off = Dk[~np.eye(len(Dk), dtype=bool)].min()
    for k in range(len(Dk)):
        i = np.where(labels == k)[0]
        Z = linkage(squareform(dist_within[np.ix_(i, i)], checks=False), "ward") if ward_on == "dist" else linkage(dist_within[i], "ward")
        co = squareform(cophenet(Z)); d[np.ix_(i, i)] = co / max(co.max(), 1e-12) * off
    return d


def sims():
    m354 = mod("m354", "354_c2l_cgw_anova.py"); fn = OUT / "sims.json"; res = jload(fn); t0 = time.time()
    for w in ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]:
        if res.get(w, {}).get("done"):
            continue
        p = m354.problem("sim", (w, 46)); args = (p["Y"], p["Gme"], p["gu"], p["fT"], p["Xh"], p["kc"], p["D"], p["xy"]); target = M.ct_target(p["D"])
        lev = M.nested_levels(p["d"], p["clone"], p["Kc"]); xyt = m354.truth_xy("sim", (w, 46)); base = json.load(open(O / "spotr_c2l_cgw_single" / f"{w}_46.json"))["arms"]; r_ = {}
        for arm, kw in [("clones", dict(clone=p["clone"], mode="clones")), ("lineage", dict(clone=p["clone"], mode="lineage", lev=lev, CT=M.tree_ct("lineage", p["clone"], p["Kc"], p["d"], p["Dk"], target)))]:
            r = M.fit(*args, verbose=False, **kw, **SW); s = p["score"](p["to_mass"](r["phi"])); s.update(M.subclade_errors(r["phi"], lev, xyt, p["xy"])); save_fit(OUT / "sims", f"{w}_{arm}", r)
            keys = ["clone_tv", "PCC", "subclade_tv_l1", "subclade_tv_l2", "subclade_tv_l3"]
            r_[arm] = dict(variant={k: float(s[k]) for k in keys}, method={k: float(base[arm]["scores"][k]) for k in keys})
            print(f"{w} {arm}: clone_tv {s['clone_tv']:.3f} (method {base[arm]['scores']['clone_tv']:.3f}); sub-clade l1-l3 {s['subclade_tv_l1']:.3f} {s['subclade_tv_l2']:.3f} {s['subclade_tv_l3']:.3f} "
                  f"(method {base[arm]['scores']['subclade_tv_l1']:.3f} {base[arm]['scores']['subclade_tv_l2']:.3f} {base[arm]['scores']['subclade_tv_l3']:.3f}) [{time.time() - t0:.0f}s]", flush=True)
        r_["notree (method)"] = {k: float(base["notree"]["scores"][k]) for k in ["clone_tv", "PCC"]}; r_["done"] = True; res[w] = r_; json.dump(res, open(fn, "w"), indent=1)


def ht260c1():
    m363, m379, m415, m406 = mod("m363", "363_htan_v2_run.py"), mod("m379", "379_c2l_reference.py"), mod("m415", "415_deeper_trees.py"), mod("m406", "406_score_authors_labels.py")
    fn = OUT / "ht260c1.json"; res = jload(fn); t0 = time.time(); d_out = OUT / "HT260C1"
    m379.OUT = m363.OUT; p = m363.problem(); z, ti, lab3 = m363.load(); labs = m363.labels(z, ti, lab3); Xc = z["Xc_fit"].astype(float)
    S = m379.signatures(Xc, labs, "htan_v2_c2l"); ek = p["types"][:-1]; keep = [e for e in ek if (labs == e).sum() > 1]; Gme = np.stack([S[e] for e in keep]); gu = S["untraced"][None, :]
    c3 = p["c3"]; Xh = M.profiles(Xc[ti], c3)["within"]; args = (p["Y"], Gme, gu, p["fT"], Xh, p["kc"], p["D"], p["xy"]); target = M.ct_target(p["D"])
    dW = within_tree(np.asarray(p["d"], float), c3, p["Dk3"]); lev = M.nested_levels(dW, c3, 3); CT = M.tree_ct("lineage", c3, 3, dW, p["Dk3"], target)
    sec = m415.section("HT260C1"); sb = z["spot_barcodes"].astype(str); cl_auth = m406.labels("HT260C1-Th1K1Fc2U1Z1Bs1", "HT260C1", sb)
    oh = np.stack([lab3 == s for s in ("B1", "B2", "A")], 1).astype(float); two = lambda P: np.stack([P[:, 0] + P[:, 1], P[:, 2]], 1)
    def scores(phi):
        P = np.asarray(phi, float); P = P / np.maximum(P.sum(1, keepdims=True), 1e-300); s = {k: v["purity"] for k, v in m415.gscore(m415.groups_from_map(P, sec), sec).items()}
        s["authors' clones"] = m406.score(two(P.T @ oh), cl_auth)["purity"]; return s
    refs = {"current SpOTr, clone labels (390_ p=1)": O / "thinning" / "HT260C1" / "p1.0" / "spotr_clones_phi.npy", "current SpOTr, no tree (390_ p=1)": O / "thinning" / "HT260C1" / "p1.0" / "spotr_notree_phi.npy",
            "current SpOTr, lineage (386_)": O / "htan_v2" / "spotr_lineagesubnb_phi.npy"}
    for k, f in refs.items():
        if k not in res and f.exists():
            res[k] = scores(np.load(f)); print(f"HT260C1 {k}: " + json.dumps({a: round(b, 3) for a, b in res[k].items()}), flush=True)
    for key, kw in [("variant, clone labels", dict(clone=c3, mode="clones", **SW)), ("variant, lineage (copy-number tree, compact prior)", dict(clone=c3, mode="lineage", lev=lev, CT=CT, spread_prior=1.0, **SW)),
                    ("variant, lineage (copy-number tree, no prior)", dict(clone=c3, mode="lineage", lev=lev, CT=CT, **SW))]:
        if key in res:
            continue
        r = M.fit(*args, verbose=False, **kw); save_fit(d_out, key.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""), r); res[key] = scores(r["phi"])
        json.dump(res, open(fn, "w"), indent=1); print(f"HT260C1 {key}: " + json.dumps({a: round(b, 3) for a, b in res[key].items()}) + f" [{time.time() - t0:.0f}s]", flush=True)
    for b in ("exact_ot", "tangram"):                                                                # 415_'s baselines, the same scoring
        k = {"exact_ot": "exact OT", "tangram": "Tangram"}[b]
        if k not in res:
            res[k] = scores(np.load(O / "deeper_trees" / "HT260C1" / "full" / f"{b}.npy"))
    json.dump(res, open(fn, "w"), indent=1)


def k3():
    m387, m413, m415, m406 = mod("m387", "387_htan_ht268b1.py"), mod("m413", "413_subclones_within.py"), mod("m415", "415_deeper_trees.py"), mod("m406", "406_score_authors_labels.py")
    fn = OUT / "k3.json"; res = jload(fn); t0 = time.time(); d_out = OUT / "K3"
    p = m387.problem("K3"); z, ti, lab = m387.load("K3"); Xc = z["Xc_fit"].astype(float); c2 = p["c2"]; Xh = M.profiles(Xc[ti], c2)["within"]
    args = (p["Y"], p["Gme"], p["gu"], p["fT"], Xh, p["kc"], p["D"], p["xy"]); target = M.ct_target(p["D"])
    d = m413.load("HT268B1"); bt = z["barcodes"].astype(str)[ti]; idx = pd.Index(d["b"]).get_indexer(bt); assert (idx >= 0).all()
    arms = [a for a in d["R"].columns if a in d["A"].columns and not a.startswith(("chrX", "chrY"))]
    Zc = np.hstack([m413.resid_std(d["R"].iloc[idx][arms].to_numpy(), d["umi"][idx]), m413.resid_std(d["A"].iloc[idx][arms].to_numpy(), d["tot"][idx])])
    dW = within_tree(Zc, c2, p["Dk2"], ward_on="obs"); lev = M.nested_levels(dW, c2, 2); CT = M.tree_ct("lineage", c2, 2, dW, p["Dk2"], target)
    sec = m415.section("K3"); sb = z["spot_barcodes"].astype(str); cl_auth = m406.labels("HT268B1-Th1K3Fc2U1Z1Bs1", "K3", sb); oh = np.stack([lab == "B", lab == "A"], 1).astype(float)
    def scores(phi):
        P = np.asarray(phi, float); P = P / np.maximum(P.sum(1, keepdims=True), 1e-300); s = {k: v["purity"] for k, v in m415.gscore(m415.groups_from_map(P, sec), sec).items()}
        s["authors' clones"] = m406.score(P.T @ oh, cl_auth)["purity"]; return s
    for k, f in {"current SpOTr, clone labels (387_)": O / "ht268b1" / "K3" / "spotr_clones_phi.npy", "current SpOTr, no tree (387_)": O / "ht268b1" / "K3" / "spotr_notree_phi.npy"}.items():
        if k not in res and f.exists():
            res[k] = scores(np.load(f)); print(f"K3 {k}: " + json.dumps({a: round(b, 3) for a, b in res[k].items()}), flush=True)
    for key, kw in [("variant, clone labels", dict(clone=c2, mode="clones", **SW)), ("variant, lineage (copy-number tree)", dict(clone=c2, mode="lineage", lev=lev, CT=CT, cross="blocks", **SW))]:
        if key in res:
            continue
        r = M.fit(*args, verbose=False, **kw); save_fit(d_out, key.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""), r); res[key] = scores(r["phi"])
        json.dump(res, open(fn, "w"), indent=1); print(f"K3 {key}: " + json.dumps({a: round(b, 3) for a, b in res[key].items()}) + f" [{time.time() - t0:.0f}s]", flush=True)
    for b in ("exact_ot", "tangram"):
        k = {"exact_ot": "exact OT", "tangram": "Tangram"}[b]
        if k not in res:
            res[k] = scores(np.load(O / "deeper_trees" / "K3" / "full" / f"{b}.npy"))
    json.dump(res, open(fn, "w"), indent=1)


def petracer():
    import ot, anndata as ad, tangram as tg
    m354, m276 = mod("m354", "354_c2l_cgw_anova.py"), mod("m276", "276_petracer.py"); fn = OUT / "petracer.json"; res = jload(fn); t0 = time.time()
    s155 = json.load(open(O / "petracer_comparison" / "scores.json"))
    for name in ["half:M3-4", "half:M3-1", "half:M3-2", "serial:M3-1>M3-2"]:
        st = name.replace(":", "_").replace(">", "_to_"); r_ = res.get(name, {}); d_out = OUT / "petracer" / st
        p = m354.problem("petracer", name); args = (p["Y"], p["Gme"], p["gu"], p["fT"], p["Xh"], p["kc"], p["D"], p["xy"]); target = M.ct_target(p["D"])
        lev = M.nested_levels(p["d"], p["clone"], p["Kc"]); xyt = m354.truth_xy("petracer", name); spots = np.asarray(p["xy"], float)
        for k in ("subclone-only clones", "cell-level clones", "cell-level lineage", "exact OT", "Tangram", "cell2location"):
            if k in s155.get(name, {}):
                r_[f"{k} (section 155)"] = {"PCC": s155[name][k]["PCC"]}
        def per_cell(P):
            P = np.asarray(P, float); P = P / np.maximum(P.sum(1, keepdims=True), 1e-300); dd = np.sqrt(((spots[None, :, :] - xyt[:, None, :]) ** 2).sum(-1)) if False else None
            out = np.array([P[i] @ np.sqrt(((spots - xyt[i]) ** 2).sum(1)) for i in range(len(P))]); return float(out.mean()), float(np.median(out))
        def record(key, phi):
            s = p["score"](p["to_mass"](phi)); e = {"PCC": float(s["PCC"]), "JS": float(s["JS"])}
            if xyt is not None:
                e.update({k: float(v) for k, v in M.subclade_errors(phi, lev, xyt, spots).items()}); e["per-cell expected distance (mean, median um)"] = per_cell(phi)
            r_[key] = e; res[name] = r_; json.dump(res, open(fn, "w"), indent=1); print(f"{name} {key}: " + json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in e.items()}) + f" [{time.time() - t0:.0f}s]", flush=True)
        jobs = [("variant, clone labels", dict(clone=p["clone"], mode="clones", **SW)),
                ("variant, lineage", dict(clone=p["clone"], mode="lineage", lev=lev, CT=M.tree_ct("lineage", p["clone"], p["Kc"], p["d"], p["Dk"], target), **SW))]
        if xyt is not None:                                                                         # per-cell references in the half designs
            jobs += [("cell-level no tree (method)", {}), ("cell-level lineage (method)", dict(clone=p["clone"], mode="lineage", lev=lev, CT=M.tree_ct("lineage", p["clone"], p["Kc"], p["d"], p["Dk"], target)))]
        for key, kw in jobs:
            if key in r_:
                continue
            r = M.fit(*args, verbose=False, **kw); save_fit(d_out, key.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""), r); record(key, r["phi"])
        if xyt is not None:
            B = m276.build(name, 0); X = B["X"]; Xc = np.asarray(B["Xc"], float); Y = np.asarray(B["Y"], float); tr = (X["kind"] == 0) & X["traced"]
            if "exact OT, per cell" not in r_:
                lx = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1) * 1e4); ly = np.log1p(Y / np.maximum(Y.sum(1, keepdims=True), 1) * 1e4)
                Pc = ot.emd(np.full(len(Xc), 1.0 / len(Xc)), np.full(len(Y), 1.0 / len(Y)), ot.dist(lx, ly), numItermax=10 ** 7) * len(Xc); np.save(d_out / "exact_ot_phi.npy", Pc[tr].astype(np.float16)); record("exact OT, per cell", Pc[tr])
            if "Tangram, per cell" not in r_:
                genes = [f"g{i}" for i in range(Xc.shape[1])]; a_sc = ad.AnnData(Xc.astype(np.float32)); a_sc.var_names = genes; a_sp = ad.AnnData(Y.astype(np.float32)); a_sp.var_names = genes
                tg.pp_adatas(a_sc, a_sp, genes=None); a_sp.obs["rna_count_based_density"] = np.full(len(Y), 1.0 / len(Y))
                Pc = np.asarray(tg.map_cells_to_space(a_sc, a_sp, mode="cells", density_prior="rna_count_based", lambda_d=1.0, num_epochs=1000, device="cuda", random_state=0, verbose=False).X, float)
                np.save(d_out / "tangram_phi.npy", Pc[tr].astype(np.float16)); record("Tangram, per cell", Pc[tr])


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    for ph in (sys.argv[1:] or ["sims", "ht260c1", "k3", "petracer"]):
        print(f"=== {ph} ===", flush=True); {"sims": sims, "ht260c1": ht260c1, "k3": k3, "petracer": petracer}[ph]()
    print("ALL DONE 419", flush=True)
