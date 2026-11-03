"""HT260C1 rebuilt (362_): SpOTr in one phase with every resolution of the copy-number tree, the baselines, and their scores.

SpOTr (spotr.c2l_gw, one phase, 16 rounds; 347_'s HTAN inputs: every annotated type its own reference, the untraced tumour nuclei's mean, the
traced nuclei's profiles smoothed within A / B):
  notree       no clone term
  clones2      clone labels A, B (the earlier paper arm)
  clones3      clone labels A, B1, B2
  clone_tree   the clone tree ((B1, B2), A), distances between clones = mean copy-number distance between their nuclei
  lineage      A and B, each with its nested subtree: the copy-number tree (Ward clustering of each clone's nuclei on their standardised
               depth-regressed arm scores; its cophenetic distances), in the lineage form of 354_
Baselines, mapping or deconvolving the same nuclei with labels A, B1, B2: cell2location (the package, 355_'s fit), DestVI, Tangram, exact OT.
Scores, on spots none of them is fitted to (362_'s second-half-gene references): rho_ba, Spearman of the B-minus-A share with the B-minus-A
copy-number signal (tumour spots); rho_wb, Spearman of the B1-minus-B2 share of B with B's within-clone signal (B's territory, spots holding B);
and the median tumour share of the most stromal quarter of spots (stromal marker score from each spot's own counts, as 359_).

FIXED BEFORE RUNNING (2026-10-31):
  E215  rho_wb: clone_tree and lineage above notree and above clones2 (the arms that do not know B1 / B2).
  E216  rho_ba: every SpOTr arm within 0.03 of the others (the B / A split comes from expression, section 126).
  E217  the most stromal quarter of spots: SpOTr's median tumour share lower than with the earlier genes (0.48, clone labels).

Usage: python 363_htan_v2_run.py spotr | baselines | c2l | score     (benchmark_env; c2l runs the package in the c2l env)
"""
import os, sys, json, time, subprocess, importlib.util
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT / "src")); SEC = "HT260C1-Th1K1Fc2U1Z1Bs1"
DATA = ROOT / "data" / "htan"; OUT = HERE / "spotr_out" / "htan_v2"; OUT.mkdir(parents=True, exist_ok=True)
SUBS = ["B1", "B2", "A"]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def load():
    z = dict(np.load(DATA / f"{SEC}_v2.npz", allow_pickle=True)); tr = z["traced"].astype(bool); ti = np.where(tr)[0]; cl = z["sn_clone"].astype(str); bsub = z["b_sub"].astype(str)
    lab3 = np.where(cl[ti] == "A", "A", bsub[ti]); return z, ti, lab3


def problem():
    import spotr.c2l_gw as M
    from scipy.spatial.distance import pdist, squareform
    z, ti, lab3 = load(); Xc, Y = z["Xc_fit"].astype(float), z["Ys_fit"].astype(float); ct = np.asarray(z["cell_type"]).astype(str); traced = z["traced"].astype(bool)
    kinds = sorted(set(ct.tolist())); tk = ["tumour"]; ek = [k for k in kinds if k not in tk]
    Gme = np.stack([Xc[ct == k].mean(0) for k in ek]); gu = Xc[(ct == "tumour") & ~traced].mean(0)[None, :]; fT = np.array([traced.sum() / (ct == "tumour").sum()])
    top = np.where(lab3 == "A", 1, 0); Xh = M.profiles(Xc[ti], top)["within"]; kc = np.zeros(len(ti), int); xy = np.asarray(z["xy"], float); D = squareform(pdist(xy))
    d = np.asarray(z["d_cn"], float); c3 = np.array([SUBS.index(l) for l in lab3]); top_ = np.where(lab3 == "A", 1, 0)
    from scipy.cluster.hierarchy import linkage, cophenet                                           # the copy-number tree: Ward clustering of each clone's
    dt = d.copy()                                                                                   # nuclei, its cophenetic distances within the clone
    for k in (0, 1):
        i = np.where(top_ == k)[0]; dt[np.ix_(i, i)] = squareform(cophenet(linkage(squareform(d[np.ix_(i, i)], checks=False), method="ward")))
    Dk3 = np.array([[d[np.ix_(c3 == a, c3 == b)].mean() if a != b else 0.0 for b in range(3)] for a in range(3)]); Dk2 = np.array([[d[np.ix_(top == a, top == b)].mean() if a != b else 0.0 for b in range(2)] for a in range(2)])
    return dict(Y=Y, Gme=Gme, gu=gu, fT=fT, Xh=Xh, kc=kc, D=D, xy=xy, d=d, dt=dt, top=top, c3=c3, Dk2=Dk2, Dk3=Dk3, types=ek + tk, lab3=lab3)


def run_spotr():
    import spotr.c2l_gw as M
    p = problem(); args = (p["Y"], p["Gme"], p["gu"], p["fT"], p["Xh"], p["kc"], p["D"], p["xy"]); target = M.ct_target(p["D"]); rec = {}; t0 = time.time()
    arms = {"notree": dict(), "clones2": dict(clone=p["top"], mode="clones"), "clones3": dict(clone=p["c3"], mode="clones"),
            "clone_tree": dict(clone=p["c3"], mode="clone_tree", CT=M.tree_ct("clone_tree", p["c3"], 3, p["d"], p["Dk3"], target)),
            "lineage": dict(clone=p["top"], mode="lineage", lev=M.nested_levels(p["dt"], p["top"], 2), CT=M.tree_ct("lineage", p["top"], 2, p["dt"], p["Dk2"], target))}
    for a, kw in arms.items():
        if (OUT / f"spotr_{a}_phi.npy").exists():
            continue
        r = M.fit(*args, verbose=False, **kw); np.save(OUT / f"spotr_{a}_phi.npy", r["phi"].astype(np.float32)); np.save(OUT / f"spotr_{a}_abund.npy", M.abund(r["th"]))
        rec[a] = dict(trace=r["trace"], weights=r.get("weights", {})); json.dump(rec, open(OUT / f"spotr_{a}.json", "w"), default=float); print(f"SpOTr {a} [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(types=p["types"]), open(OUT / "types.json", "w"))


def run_baselines():
    import ot, anndata as ad, pandas as pd_
    z, ti, lab3 = load(); Xc, Y = z["Xc_fit"].astype(float), z["Ys_fit"].astype(float); mass = np.full(len(Y), 1.0 / len(Y))
    if not (OUT / "exact_ot.npy").exists():
        lx = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1) * 1e4); ly = np.log1p(Y / np.maximum(Y.sum(1, keepdims=True), 1) * 1e4)
        P = ot.emd(np.full(len(Xc), 1.0 / len(Xc)), mass, ot.dist(lx, ly), numItermax=10 ** 7) * len(Xc); np.save(OUT / "exact_ot.npy", P[ti].astype(np.float32)); print("exact OT", flush=True)
    if not (OUT / "tangram.npy").exists():
        import tangram as tg
        genes = [f"g{i}" for i in range(Xc.shape[1])]; a_sc = ad.AnnData(Xc.astype(np.float32)); a_sc.var_names = genes; a_sp = ad.AnnData(Y.astype(np.float32)); a_sp.var_names = genes
        tg.pp_adatas(a_sc, a_sp, genes=None); a_sp.obs["rna_count_based_density"] = mass
        P = np.asarray(tg.map_cells_to_space(a_sc, a_sp, mode="cells", density_prior="rna_count_based", lambda_d=1.0, num_epochs=1000, device="cuda", random_state=0, verbose=False).X, float)
        np.save(OUT / "tangram.npy", P[ti].astype(np.float32)); print("Tangram", flush=True)
    if not (OUT / "destvi.npz").exists():
        import scvi
        labs = labels(z, ti, lab3); scvi.settings.seed = 0
        sc_ = ad.AnnData(Xc.astype(np.float32)); sc_.obs["label"] = pd_.Categorical(labs); scvi.model.CondSCVI.setup_anndata(sc_, labels_key="label")
        msc = scvi.model.CondSCVI(sc_, weight_obs=False); msc.train(max_epochs=300, enable_progress_bar=False)
        st = ad.AnnData(Y.astype(np.float32)); scvi.model.DestVI.setup_anndata(st); mst = scvi.model.DestVI.from_rna_model(st, msc); mst.train(max_epochs=1500, enable_progress_bar=False)
        names = list(msc.adata_manager.get_state_registry("labels").categorical_mapping); pr = mst.get_proportions(); pr = pr[names].values if hasattr(pr, "values") else pr
        np.savez_compressed(OUT / "destvi.npz", w=np.asarray(pr, np.float32), types=np.array(names)); print("DestVI", flush=True)


def labels(z, ti, lab3):
    ct = np.asarray(z["cell_type"]).astype(str); traced = z["traced"].astype(bool); labs = np.where(ct == "tumour", "untraced", ct).astype(object); labs[ti] = lab3; return labs.astype(str)


def run_c2l():
    z, ti, lab3 = load(); f_in = HERE / "spotr_out" / "c2l_package" / "htan_HT260C1v2_in.npz"
    if not f_in.with_name("htan_HT260C1v2_out.npz").exists():
        np.savez_compressed(f_in, Xc=z["Xc_fit"].astype(np.float32), Y=z["Ys_fit"].astype(np.float32), labels=labels(z, ti, lab3), clones=np.array(SUBS), nhat=10.0,
                            genes=np.array([f"g{j}" for j in range(z["Xc_fit"].shape[1])]), src="htan_v2", name=json.dumps(SEC))
        code = f"import importlib.util; s = importlib.util.spec_from_file_location('m355', r'{HERE / '355_c2l_package.py'}'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); m.fit_one(r'{f_in}')"
        subprocess.run([str(Path.home() / "miniconda3/envs/c2l/bin/python"), "-c", code], check=True, env=dict(os.environ, PYTHONNOUSERSITE="1"))
    o = np.load(f_in.with_name("htan_HT260C1v2_out.npz"), allow_pickle=True); np.savez_compressed(OUT / "c2l.npz", w=o["w"], types=o["types"]); print("cell2location", flush=True)


def score():
    from scipy.stats import spearmanr
    z, ti, lab3 = load(); sb = z["spot_barcodes"].astype(str); ref = pd.read_csv(DATA / f"{SEC}_v2_reference.tsv", sep="\t").set_index("barcode").reindex(sb)
    oh = np.stack([(lab3 == s_).astype(float) for s_ in SUBS], 1); masses = {}
    for a in [f.name[len("spotr_"):-len("_phi.npy")] for f in sorted(OUT.glob("spotr_*_phi.npy"))]:   # every SpOTr arm fitted (384_ adds the *sub arms)
        if (OUT / f"spotr_{a}_phi.npy").exists():
            masses[a] = np.load(OUT / f"spotr_{a}_phi.npy").astype(float).T @ oh
    for b in ["exact_ot", "tangram"]:
        if (OUT / f"{b}.npy").exists():
            P = np.load(OUT / f"{b}.npy").astype(float); masses[b] = (P / np.maximum(P.sum(1, keepdims=True), 1e-300)).T @ oh
    for b in ["c2l", "destvi"]:
        if (OUT / f"{b}.npz").exists():
            o = np.load(OUT / f"{b}.npz", allow_pickle=True); ty = [str(t) for t in o["types"]]; masses[b] = np.stack([o["w"][:, ty.index(s_)].astype(float) for s_ in SUBS], 1)
    m359 = mod("m359", "359_htan_morphology.py"); import scanpy as sc
    a_ = m359.visium(); sc.pp.normalize_total(a_, target_sum=1e4); sc.pp.log1p(a_); sc.tl.score_genes(a_, m359.STROMA, score_name="st", random_state=0)
    st = pd.Series(a_.obs["st"].to_numpy(), index=a_.obs_names).reindex(sb).to_numpy(); hi = st >= np.nanquantile(st, 0.75); types = json.load(open(OUT / "types.json"))["types"]
    res = {}
    for k, P in masses.items():
        B, A = P[:, 0] + P[:, 1], P[:, 2]; ok = ref["ref_ba"].notna().to_numpy(); d = (B - A) / np.maximum(B + A, 1e-300)
        okw = ref["ref_wb"].notna().to_numpy() & (B > 1e-9 * B.max()); dw = (P[:, 0] - P[:, 1]) / np.maximum(B, 1e-300)
        r = dict(rho_ba=float(spearmanr(d[ok], ref["ref_ba"].to_numpy()[ok])[0]), rho_wb=float(spearmanr(dw[okw], ref["ref_wb"].to_numpy()[okw])[0]), wb_spots=int(okw.sum()))
        if (OUT / f"spotr_{k}_abund.npy").exists():
            Ab = np.load(OUT / f"spotr_{k}_abund.npy").astype(float); r["tumour_share_stroma_q4"] = float(np.median(Ab[hi, types.index("tumour")] / Ab[hi].sum(1)))
        res[k] = r
    if (OUT / "c2l.npz").exists():
        o = np.load(OUT / "c2l.npz", allow_pickle=True); ty = [str(t) for t in o["types"]]; w = o["w"].astype(float); tum = [ty.index(t) for t in SUBS + ["untraced"]]
        res["c2l"]["tumour_share_stroma_q4"] = float(np.median(w[hi][:, tum].sum(1) / w[hi].sum(1)))
    json.dump(dict(scores=res, masses={k: v.tolist() for k, v in masses.items()}), open(OUT / "scores.json", "w")); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    {"spotr": run_spotr, "baselines": run_baselines, "c2l": run_c2l, "score": score}[sys.argv[1]]()
