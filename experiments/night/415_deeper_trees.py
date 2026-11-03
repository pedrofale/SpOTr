"""Deeper trees on HTAN: do the sub-clones inside the main clones (413_, 414_) give the tree something to do?  (Pedro, 2026-11-01: "Yes go ahead"
to: "build the deeper trees and run SpOTr with clone labels, clone tree and lineage against cell2location".)

Sub-splits that are spatially organised beyond their parent clone (414_, Moran's I with the parent references and purity regressed out, against 50
random splits of the same nuclei): HT260C1 B1 -> B1a (76 nuclei) / B1b (192), I 0.60 against random median 0.29, max 0.60 (B1a: B1's top-right glands,
B1b: its left glands); HT268B1 A on K3 -> A1 (1,407) / A2 (1,528), I 0.26 against 0.09 / 0.25 (weaker; secondary).  HT268B1's B split is not used
(its signature is the B-vs-A axis, cosine 0.81, and has no spatial structure of its own).

Labels: HT260C1 A / B1a / B1b / B2; K3 A1 / A2 / B.  Trees: HT260C1 ((B1a, B1b), B2), A with branch distances the label groups' mean copy-number
distances (d_cn; printed); K3 ((A1, A2), B), topology only (A1-A2 0.5, A-B 1; K3 has no nucleus copy-number distances).  Subclone-only profiles:
cell2location's RegressionModel signatures of the new label set.
Arms: SpOTr no tree; clone labels at the old resolution (HT260C1 A / B1 / B2; K3 A / B); clone labels at the new resolution; the new clone tree
(per-pair cross term: more than two subclones); the lineage form (top clones, nested levels from the new tree).  Baselines: exact OT, Tangram, the
cell2location package (the new labels).  On the sections' actual counts, no thinning (Pedro, 2026-11-01: "Stop doing the depth thinning tests.
The results should be based on the actual data without manipulations.").
References (held-out genes, 414_): the sub-split's copy-number projection within the parent's territory ("B1b vs B1a", "A2 vs A1"), and the same
with the parent references and purity regressed out; the old contrasts ("B vs A", HT260C1 "B1 vs B2") for the record.  Score: balanced purity.

FIXED BEFORE RUNNING (2026-11-01), AMENDED THE SAME DAY when thinning was dropped -- the first version predicted for depths 0.1 and 0.03 and made
no prediction at full depth; two full-depth fits (no tree, old labels) had been written when the run was stopped, and no output of it was read
before this amendment:
P1. HT260C1 "B1b vs B1a": SpOTr with the new clone labels or the new clone tree >= SpOTr without a tree + 0.03.
P2. HT260C1 "B1b vs B1a": SpOTr's better of those two arms >= cell2location (new labels).
P3. HT260C1: the new labels help beyond the old ones -- clone labels A / B1a / B1b / B2 >= A / B1 / B2 + 0.03 on "B1b vs B1a".
K3 is secondary (the weaker reference): the same comparisons reported, no prediction.

Usage: python 415_deeper_trees.py spotr|c2l|score [HT260C1|K3]   (benchmark_env, GPU; c2l via the c2l env; writes spotr_out/deeper_trees/)
"""
import json, sys, os, time, subprocess, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; O = HERE / "spotr_out"; OUT = O / "deeper_trees"; DATA = ROOT / "data" / "htan"; sys.path.insert(0, str(ROOT / "src"))
SECS = {"HT260C1": "HT260C1-Th1K1Fc2U1Z1Bs1", "K3": "HT268B1-Th1K3Fc2U1Z1Bs1"}


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def section(short):
    """inputs, labels (old and new), trees, groups and references"""
    import spotr.c2l_gw as M
    from scipy.spatial.distance import pdist, squareform
    sec = SECS[short]; z = dict(np.load(DATA / f"{sec}_v2.npz", allow_pickle=True)); ti = np.where(z["traced"].astype(bool))[0]; bt = z["barcodes"].astype(str)[ti]
    m379 = mod("m379", "379_c2l_reference.py"); out = OUT / short; out.mkdir(parents=True, exist_ok=True); m379.OUT = out
    if short == "HT260C1":
        cl = z["sn_clone"].astype(str); bsub = z["b_sub"].astype(str); old = np.where(cl[ti] == "A", "A", bsub[ti]); sp = np.load(O / "subclone_spatial" / f"HT260C1_B1_{sec}.npz", allow_pickle=True)
        hb = dict(zip(sp["nuclei"].astype(str), sp["h"])); assert all(b in hb for b in bt[old == "B1"]); new = old.astype(object).copy()
        for i in np.where(old == "B1")[0]:
            new[i] = "B1b" if hb[bt[i]] == 1 else "B1a"
        new = new.astype(str); NEW = ["A", "B1a", "B1b", "B2"]; OLD = ["A", "B1", "B2"]; top = np.where(new == "A", 1, 0)
        d = np.asarray(z["d_cn"], float); cN = np.array([NEW.index(l) for l in new]); Dk = np.array([[d[np.ix_(cN == a, cN == b)].mean() if a != b else 0.0 for b in range(4)] for a in range(4)])
        Dk2 = np.array([[d[np.ix_(top == a, top == b)].mean() if a != b else 0.0 for b in range(2)] for a in range(2)])
    else:
        old = z["sn_clone"].astype(str)[ti]; sp = np.load(O / "subclone_spatial" / f"HT268B1_A_{sec}.npz", allow_pickle=True); hb = dict(zip(sp["nuclei"].astype(str), sp["h"]))
        miss = [b for b in bt[old == "A"] if b not in hb]; print(f"K3: {len(miss)} A nuclei without a sub-split label (kept as their own nearest: dropped from the new labels)", flush=True); assert not miss
        new = np.array(["B" if l == "B" else ("A2" if hb[b] == 1 else "A1") for l, b in zip(old, bt)]); NEW = ["A1", "A2", "B"]; OLD = ["A", "B"]; top = np.where(new == "B", 0, 1)
        cN = np.array([NEW.index(l) for l in new]); Dk = np.array([[0, 0.5, 1], [0.5, 0, 1], [1, 1, 0]], float); Dk2 = 1.0 - np.eye(2)
    ct = np.asarray(z["cell_type"]).astype(str); labs = np.where(ct == "tumour", "untraced", ct).astype(object); labs[ti] = new; labs = labs.astype(str)
    S = m379.signatures(z["Xc_fit"].astype(float), labs, f"{short}_deep_c2l"); ek = sorted(t for t in set(ct.tolist()) if t != "tumour"); n = {e: int((labs == e).sum()) for e in ek}; keep = [e for e in ek if n[e] > 1]
    xy = np.asarray(z["xy"], float); D = squareform(pdist(xy)); target = M.ct_target(D); dN = Dk[np.ix_(cN, cN)]; cO = np.array([OLD.index(l if l in OLD else l[:2] if short == "HT260C1" else "A") for l in new])
    arms = {"notree": dict(), "labels_old": dict(clone=cO, mode="clones"), "labels_new": dict(clone=cN, mode="clones"),
            "clone_tree_new": dict(clone=cN, mode="clone_tree", CT=M.tree_ct("clone_tree", cN, len(NEW), dN, Dk, target)),
            "lineage_new": dict(clone=top, mode="lineage", lev=M.nested_levels(dN, top, 2), CT=M.tree_ct("lineage", top, 2, dN, Dk2, target))}
    sb = z["spot_barcodes"].astype(str); R = pd.read_csv(DATA / f"{sec}_v2_reference.tsv", sep="\t").set_index("barcode").reindex(sb)
    assert (sp["sb"].astype(str) == sb).all(); terr = sp["terr"].astype(bool); ref = np.where(terr, sp["p"], np.nan)
    pur = R["purity"].to_numpy(float); cov = [np.ones(terr.sum()), R["ref_ba"].to_numpy()[terr], np.nan_to_num(pur[terr], nan=np.nanmean(pur[terr]))]
    if short == "HT260C1":
        cov.append(np.nan_to_num(R["ref_wb"].to_numpy()[terr]))
    Dm = np.column_stack(cov); ref_r = np.full(len(sb), np.nan); ref_r[terr] = sp["p"][terr] - Dm @ np.linalg.lstsq(Dm, sp["p"][terr], rcond=None)[0]
    if short == "HT260C1":
        contrasts = {"B1b vs B1a": (["B1b"], ["B1a"], ref), "B1b vs B1a (parent removed)": (["B1b"], ["B1a"], ref_r), "B vs A": (["B1a", "B1b", "B2"], ["A"], R["ref_ba"].to_numpy()),
                     "B1 vs B2": (["B1a", "B1b"], ["B2"], R["ref_wb"].to_numpy())}
    else:
        contrasts = {"A2 vs A1": (["A2"], ["A1"], ref), "A2 vs A1 (parent removed)": (["A2"], ["A1"], ref_r), "B vs A": (["B"], ["A1", "A2"], R["ref_ba"].to_numpy())}
    return dict(z=z, ti=ti, new=new, NEW=NEW, labs=labs, Dk=Dk, args=(z["Ys_fit"].astype(float), np.stack([S[e] for e in keep]), S["untraced"][None, :], np.array([len(ti) / (ct == "tumour").sum()]),
                np.stack([S[l] for l in new]), np.zeros(len(ti), int), D, xy), arms=arms, contrasts=contrasts, xy=xy, out=out)


def gscore(G, sec):
    """balanced purity per contrast from per-group spot masses"""
    out = {}
    for nm, (g1, g0, ref) in sec["contrasts"].items():
        a = sum(G[g] for g in g1); b = sum(G[g] for g in g0); ok = np.isfinite(ref); zr = np.full(len(ref), np.nan); zr[ok] = (ref[ok] - ref[ok].mean()) / ref[ok].std()
        t1, t0 = ok & (zr > 0.5), ok & (zr < -0.5); p1 = a[t1].sum() / max((a + b)[t1].sum(), 1e-300); p0 = b[t0].sum() / max((a + b)[t0].sum(), 1e-300)
        out[nm] = dict(purity=float((p1 + p0) / 2), purity_1=float(p1), purity_0=float(p0))
    return out


def groups_from_map(P, sec):
    P = np.asarray(P, float); P = P / np.maximum(P.sum(1, keepdims=True), 1e-300); return {g: P[sec["new"] == g].sum(0) for g in sec["NEW"]}


def run_spotr(short):
    import warnings; warnings.filterwarnings("ignore")
    import spotr.c2l_gw as M, ot, anndata as ad, tangram as tg
    sec = section(short); print(short, "label groups' copy-number distances\n", np.round(sec["Dk"], 3), "\n", dict(zip(*np.unique(sec["new"], return_counts=True))), flush=True)
    if True:
        d = sec["out"] / "full"; d.mkdir(parents=True, exist_ok=True); Y = sec["args"][0]; args = sec["args"]; t0 = time.time()
        for a, kw in sec["arms"].items():
            f = d / f"spotr_{a}_phi.npy"
            if f.exists():
                continue
            r = M.fit(*args, verbose=False, **kw); np.save(f, np.asarray(r["phi"], np.float32))
            json.dump(dict(weights=r.get("weights", {}), row_err_max=max(t["row_err"] for t in r["trace"])), open(d / f"spotr_{a}.json", "w"), default=float)
            print(f"{short} SpOTr {a} [{time.time() - t0:.0f}s] " + json.dumps({k: round(v['purity'], 3) for k, v in gscore(groups_from_map(r['phi'], sec), sec).items()}), flush=True)
        Xc = sec["z"]["Xc_fit"].astype(float); mass = np.full(len(Y), 1.0 / len(Y)); ti = sec["ti"]
        if not (d / "exact_ot.npy").exists():
            lx = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1) * 1e4); ly = np.log1p(Y / np.maximum(Y.sum(1, keepdims=True), 1) * 1e4)
            np.save(d / "exact_ot.npy", ((ot.emd(np.full(len(Xc), 1.0 / len(Xc)), mass, ot.dist(lx, ly), numItermax=10 ** 7) * len(Xc))[ti]).astype(np.float32))
        if not (d / "tangram.npy").exists():
            genes = [f"g{i}" for i in range(Xc.shape[1])]; a_sc = ad.AnnData(Xc.astype(np.float32)); a_sc.var_names = genes; a_sp = ad.AnnData(Y.astype(np.float32)); a_sp.var_names = genes
            tg.pp_adatas(a_sc, a_sp, genes=None); a_sp.obs["rna_count_based_density"] = mass
            P = np.asarray(tg.map_cells_to_space(a_sc, a_sp, mode="cells", density_prior="rna_count_based", lambda_d=1.0, num_epochs=1000, device="cuda", random_state=0, verbose=False).X, float)[ti]
            np.save(d / "tangram.npy", P.astype(np.float32))
        for b in ("exact_ot", "tangram"):
            print(f"{short} {b} " + json.dumps({k: round(v['purity'], 3) for k, v in gscore(groups_from_map(np.load(d / f'{b}.npy'), sec), sec).items()}), flush=True)
        # cell2location's input for this depth (run by the c2l phase)
        f_in = d / "c2l_in.npz"
        if not f_in.exists():
            np.savez_compressed(f_in, Xc=Xc.astype(np.float32), Y=Y.astype(np.float32), labels=sec["labs"], clones=np.array(sec["NEW"]), nhat=10.0, genes=np.array([f"g{j}" for j in range(Xc.shape[1])]),
                                src="htan_deep", name=json.dumps([short, "full"]))


def run_c2l(short):
    if True:
        f_in = OUT / short / "full" / "c2l_in.npz"
        if f_in.exists() and not Path(str(f_in).replace("_in.npz", "_out.npz")).exists():
            t0 = time.time(); code = f"import importlib.util; s = importlib.util.spec_from_file_location('m355', r'{HERE / '355_c2l_package.py'}'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); m.fit_one(r'{f_in}')"
            subprocess.run([str(Path.home() / "miniconda3/envs/c2l/bin/python"), "-c", code], check=True, env=dict(os.environ, PYTHONNOUSERSITE="1"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"{short} cell2location [{time.time() - t0:.0f}s]", flush=True)


def score(shorts):
    res = {}
    for short in shorts:
        sec = section(short)
        if True:
            d = OUT / short / "full"; r = {}
            for a in list(sec["arms"]) + ["exact_ot", "tangram"]:
                f = d / (f"spotr_{a}_phi.npy" if a in sec["arms"] else f"{a}.npy")
                if f.exists():
                    r[a if a not in sec["arms"] else f"SpOTr {a}"] = gscore(groups_from_map(np.load(f), sec), sec)
            fo = d / "c2l_out.npz"
            if fo.exists():
                o = np.load(fo, allow_pickle=True); ty = [str(t) for t in o["types"]]; W = np.asarray(o["w"], float); r["cell2location"] = gscore({g: W[:, ty.index(g)] for g in sec["NEW"]}, sec)
            res[short] = r
            print(short); [print(f"   {a:28s} " + "  ".join(f"{k}: {v['purity']:.3f}" for k, v in s.items())) for a, s in r.items()]
    json.dump(res, open(OUT / "scores.json", "w"), indent=1)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); phase = sys.argv[1]; shorts = sys.argv[2:] or ["HT260C1", "K3"]
    if phase == "spotr":
        [run_spotr(s) for s in shorts]
    elif phase == "c2l":
        [run_c2l(s) for s in shorts]
    else:
        score(shorts)
