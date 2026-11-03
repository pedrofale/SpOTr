"""HT268B1 (breast cancer liver metastasis; HTAN WUSTL) with the current SpOTr: a real-data test of the simulations' prediction that the tree
helps where a subclone is spatially compact (section 145: in the heritable study the gain falls from +0.05 for subclones spanning under 0.3 of
the section's spread to 0 above 0.7).  Pedro, 2026-10-31: "I think it's crucial that we show that the method is useful in real data though.
That's what the HTAN is for"; "Yes please go ahead".

Compactness, measured before any fit (section 146, from each section's own copy-number reference): B spans 0.54 of the section's spread in K3
(Th1K3, a separate piece; 89% of its spots in one 353-spot territory) and 1.02 in U12 (Th1H3, the nuclei's block; no territory).  HT260C1, where the
tree did nothing: B 0.84, A 0.76.

Prep, as 362_ for HT260C1: the genes shared by the nuclei and the section split at random in two halves (seed 0); SpOTr and every method fit
genes of the first half passing cell2location's filter_genes rule on the nuclei, 10% of them held out; the copy-number reference uses only the
second half (B-minus-A arm effect of the screen's k = 2 split, + = B; tumour spots with the authors' purity >= its median).  Nuclei: the authors'
cell types; tumour nuclei B / A from case_assign.py, the rest of the tumour untraced.  Spots in tissue with the authors' purity and >= 200 counts.
SpOTr (one phase, plug-in likelihood): every reference profile cell2location's RegressionModel signature for its label (c2l_signatures.py; each
traced nucleus gets its subclone's); reference types with one nucleus would be dropped (386_), none here.  Arms: no tree; clone labels B, A (each
subclone's spread learned); the clone tree (B and A apart).  The k = 3 split of A (screen: 2,390 / 634) is not used.
Baselines: exact OT and Tangram on the same genes (expression only, as 363_).  cell2location: after the SpOTr results (not while iterating).
Score: Spearman of each map's B-minus-A per unit tumour mass with the section's reference, over the reference spots with tumour mass.

FIXED BEFORE RUNNING (2026-10-31):
  E245  K3 (B compact): the clone tree or clone labels above no tree by >= 0.02.
  E246  U12 (B not compact): the clone tree and clone labels within 0.01 of no tree.
  E247  both sections: SpOTr's best tree arm at or above exact OT and Tangram.
Read: each arm's learned spreads against the section's spread; B's share of SpOTr's tumour mass against its share of the reference's tumour spots.

Usage: python 387_htan_ht268b1.py prep|spotr|baselines|score [U12|K3 ...]   (benchmark_env; GPU; cell2location's regression in the c2l env)
"""
import os, sys, gzip, json, time, importlib.util
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT / "src")); SCR = HERE.parent / "htan_screen"; CASE = "HT268B1"; RAW = SCR / "raw" / CASE
SECS = {"U12": "HT268B1-Th1H3Fc2U12Z1Bs1", "K3": "HT268B1-Th1K3Fc2U1Z1Bs1"}; OUT0 = HERE / "spotr_out" / "ht268b1"; DATA = ROOT / "data" / "htan"; SEED, MIN_SPOT = 0, 200


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def prep(short):
    from scipy.ndimage import uniform_filter1d
    m362 = mod("m362", "362_htan_v2_prep.py"); SEC = SECS[short]; rng = np.random.default_rng(SEED)
    Xs, fs, bs = m362.read10x(RAW / f"{CASE}-Th1H3"); ids_s, names_s = fs[0].to_numpy(), fs[1].to_numpy(); Xv, fv, bv = m362.read10x(RAW / SEC); ids_v = fv[0].to_numpy()
    meta = pd.read_csv(SCR / f"{CASE}_sn_meta.tsv", sep="\t").set_index("Original_barcode"); asg = pd.read_csv(SCR / f"{CASE}_sn_assign_k2.tsv", sep="\t").set_index("barcode")
    keep = np.isin(bs, meta.index[meta.cell_type != "Low quality"]); Xs, bs = Xs[keep], bs[keep]; ct = meta.loc[bs, "cell_type"].to_numpy(); ctype = np.where(ct == "Tumor", "tumour", ct)
    pf = RAW / f"{SEC}-tissue_positions_list.csv"; pos = pd.read_csv(pf, header=None) if pf.exists() else pd.read_csv(RAW / f"{SEC}-tissue_positions.csv")
    pos.columns = ["bc", "in", "r", "c", "py", "px"]; pos = pos.set_index("bc").reindex(bv)
    um = 55.0 / json.load(open(RAW / f"{SEC}-scalefactors_json.json"))["spot_diameter_fullres"]; stm = pd.read_csv(SCR / f"{SEC}_st_meta.tsv", sep="\t").set_index("barcode").reindex(bv)
    shared = [g for g in ids_v if g in set(ids_s)]; js = {g: i for i, g in enumerate(ids_s)}; jv = {g: i for i, g in enumerate(ids_v)}; name = dict(zip(ids_s, names_s))
    Xs_sh = Xs[:, [js[g] for g in shared]].tocsc(); Xv_sh = Xv[:, [jv[g] for g in shared]].tocsc(); col = {g: i for i, g in enumerate(shared)}
    half = rng.permutation(len(shared)); poolF = set(shared[i] for i in half[: len(shared) // 2]); poolR = set(shared) - poolF
    F0 = [g for g in shared if g in poolF]; sel = m362.c2l_filter(Xs_sh[:, [col[g] for g in F0]]); fit_all = [g for g, s_ in zip(F0, sel) if s_]
    ev = sorted(rng.choice(fit_all, size=len(fit_all) // 10, replace=False).tolist()); fitg = [g for g in fit_all if g not in set(ev)]
    intis = (pos["in"].to_numpy() == 1) & stm["Tumor_purity_estimate"].notna().to_numpy()
    kv = intis & (np.asarray(Xv_sh[:, [col[g] for g in fitg]].sum(1)).ravel() >= MIN_SPOT); Xv_k, bv_k = Xv_sh[kv], bv[kv]; xy = pos.loc[bv_k, ["px", "py"]].to_numpy(float) * um
    seen = np.asarray(Xv_k.sum(0)).ravel() > 0; fitg, ev = [g for g in fitg if seen[col[g]]], [g for g in ev if seen[col[g]]]; F, V = [col[g] for g in fitg], [col[g] for g in ev]
    cl = asg["clone"].reindex(bs).fillna("").to_numpy(); cl = np.where(ctype == "tumour", cl, ""); traced = np.isin(cl, ["A", "B"])
    np.savez_compressed(DATA / f"{SEC}_v2.npz", Xc_fit=Xs_sh[:, F].toarray().astype(np.float32), Ys_fit=Xv_k[:, F].toarray().astype(np.float32), Xc_ev=Xs_sh[:, V].toarray().astype(np.float32),
                        Ys_ev=Xv_k[:, V].toarray().astype(np.float32), lib=np.asarray(Xs.sum(1)).ravel().astype(float), spot_total=np.asarray(Xv[kv].sum(1)).ravel().astype(float),
                        ev_genes=np.array([name[g] for g in ev]), fit_genes=np.array([name[g] for g in fitg]), cell_type=ctype, xy=xy, barcodes=bs, spot_barcodes=bv_k, sn_clone=cl, traced=traced)
    print(f"{SEC} v2: nuclei {len(bs)}, traced {traced.sum()} (A {np.sum(cl == 'A')}, B {np.sum(cl == 'B')}); spots {kv.sum()} of {intis.sum()}; shared genes {len(shared)}, "
          f"fit pool {len(F0)} -> cell2location's filter {len(fit_all)} -> fitted {len(fitg)}, held out {len(ev)}", flush=True)
    # the copy-number reference from the second half of the genes (as 362_)
    gp = pd.read_csv(ROOT / "data" / "b189" / "gene_positions.csv").drop_duplicates("gene_id").set_index("gene_id"); Xr = Xv_sh[kv]; det = np.asarray((Xr > 0).mean(0)).ravel() >= 0.10; CEN = m362.CEN
    gi = [i for i, g in enumerate(shared) if det[i] and g in poolR and g in gp.index and str(gp.loc[g, "chrom"]) in CEN]
    chs = np.array([str(gp.loc[shared[i], "chrom"]) for i in gi]); sts = np.array([gp.loc[shared[i], "start"] for i in gi], float)
    o = np.lexsort((sts, chs.astype(int))); gi = [gi[k] for k in o]; chs, sts = chs[o], sts[o]; arm = np.array([f"chr{c}{'p' if s < CEN[c] * 1e6 else 'q'}" for c, s in zip(chs, sts)])
    pur = stm.loc[bv_k, "Tumor_purity_estimate"].to_numpy(float)
    L = np.log1p(np.asarray(Xr[:, gi].todense(), float) / np.asarray(Xv[kv].sum(1), float) * 1e4); L -= L[pur <= np.percentile(pur, 20)].mean(0)
    for c in np.unique(chs):
        j = chs == c; L[:, j] = uniform_filter1d(L[:, j], size=min(101, j.sum()), axis=1, mode="nearest")
    L -= np.median(L, axis=1, keepdims=True)
    scr = json.load(open(SCR / f"{CASE}_screen.json")); e = pd.Series(scr["splits"]["2"]["pairs"][0]["effect"]); clu = pd.read_csv(SCR / f"{CASE}_sn_clusters.tsv", sep="\t")
    small = int(clu.k2[clu.k2 > 0].value_counts().idxmin()); e = e if small == 2 else -e
    arms = [x for x in pd.unique(arm) if (arm == x).sum() >= 20 and x in e.index]; S = np.stack([L[:, arm == x].mean(1) for x in arms], 1); tum = pur >= np.median(pur); T = S[tum]
    T = T - np.outer(pur[tum] - pur[tum].mean(), np.polyfit(pur[tum], T, 1)[0]); T = (T - T.mean(0)) / T.std(0); ev_ = e.reindex(arms).to_numpy(); ref_ba = T @ (ev_ / np.linalg.norm(ev_))
    old = pd.read_csv(DATA / f"{SEC}_reference.tsv", sep="\t").set_index("barcode")["ref_c2"]; both = old.index.intersection(bv_k[tum])
    rho_old = pd.Series(ref_ba, index=bv_k[tum]).loc[both].corr(old.loc[both], method="spearman")
    pd.DataFrame({"barcode": bv_k[tum], "x_um": xy[tum, 0], "y_um": xy[tum, 1], "purity": pur[tum], "ref_ba": ref_ba}).to_csv(DATA / f"{SEC}_v2_reference.tsv", sep="\t", index=False)
    print(f" reference: {len(gi)} second-half genes, {len(arms)} arms, {tum.sum()} tumour spots; against the earlier reference (Spearman) {rho_old:.3f}", flush=True)


def load(short):
    z = dict(np.load(DATA / f"{SECS[short]}_v2.npz", allow_pickle=True)); ti = np.where(z["traced"].astype(bool))[0]; lab = z["sn_clone"].astype(str)[ti]; return z, ti, lab


def labels(z, ti, lab):
    ct = np.asarray(z["cell_type"]).astype(str); labs = np.where(ct == "tumour", "untraced", ct).astype(object); labs[ti] = lab; return labs.astype(str)


def problem(short):
    from scipy.spatial.distance import pdist, squareform
    m379 = mod("m379", "379_c2l_reference.py"); out = OUT0 / short; out.mkdir(parents=True, exist_ok=True); m379.OUT = out
    z, ti, lab = load(short); labs = labels(z, ti, lab); S = m379.signatures(z["Xc_fit"].astype(float), labs, f"{short}_c2l")
    ct = np.asarray(z["cell_type"]).astype(str); ek = sorted(t for t in set(ct.tolist()) if t != "tumour"); n = {e: int((labs == e).sum()) for e in ek}; keep = [e for e in ek if n[e] > 1]
    c2 = np.array([{"B": 0, "A": 1}[l] for l in lab]); Dk2 = 1.0 - np.eye(2); xy = np.asarray(z["xy"], float)
    return dict(Y=z["Ys_fit"].astype(float), Gme=np.stack([S[e] for e in keep]), gu=S["untraced"][None, :], fT=np.array([len(ti) / (ct == "tumour").sum()]),
                Xh=np.stack([S[l] for l in lab]), kc=np.zeros(len(ti), int), D=squareform(pdist(xy)), xy=xy, c2=c2, Dk2=Dk2, types=keep + ["tumour"], dropped=[e for e in ek if n[e] <= 1], n=n, out=out)


def run_spotr(short):
    import spotr.c2l_gw as M
    p = problem(short); out = p["out"]; args = (p["Y"], p["Gme"], p["gu"], p["fT"], p["Xh"], p["kc"], p["D"], p["xy"]); target = M.ct_target(p["D"]); t0 = time.time()
    print(short, "nuclei per reference type", p["n"], "dropped", p["dropped"], flush=True); d2 = p["Dk2"][np.ix_(p["c2"], p["c2"])]
    arms = {"notree": dict(), "clones": dict(clone=p["c2"], mode="clones"), "clone_tree": dict(clone=p["c2"], mode="clone_tree", CT=M.tree_ct("clone_tree", p["c2"], 2, d2, p["Dk2"], target))}
    for a, kw in arms.items():
        if (out / f"spotr_{a}_phi.npy").exists():
            continue
        r = M.fit(*args, verbose=False, **kw); np.save(out / f"spotr_{a}_phi.npy", r["phi"].astype(np.float32)); np.save(out / f"spotr_{a}_abund.npy", M.abund(r["th"]))
        json.dump(dict(trace=r["trace"], weights=r.get("weights", {})), open(out / f"spotr_{a}.json", "w"), default=float); print(f"{short} SpOTr {a} [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(types=p["types"]), open(out / "types.json", "w"))


def run_baselines(short):
    import ot, anndata as ad
    z, ti, lab = load(short); out = OUT0 / short; out.mkdir(parents=True, exist_ok=True); Xc, Y = z["Xc_fit"].astype(float), z["Ys_fit"].astype(float); mass = np.full(len(Y), 1.0 / len(Y))
    if not (out / "exact_ot.npy").exists():
        lx = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1) * 1e4); ly = np.log1p(Y / np.maximum(Y.sum(1, keepdims=True), 1) * 1e4)
        P = ot.emd(np.full(len(Xc), 1.0 / len(Xc)), mass, ot.dist(lx, ly), numItermax=10 ** 7) * len(Xc); np.save(out / "exact_ot.npy", P[ti].astype(np.float32)); print(short, "exact OT", flush=True)
    if not (out / "tangram.npy").exists():
        import tangram as tg
        genes = [f"g{i}" for i in range(Xc.shape[1])]; a_sc = ad.AnnData(Xc.astype(np.float32)); a_sc.var_names = genes; a_sp = ad.AnnData(Y.astype(np.float32)); a_sp.var_names = genes
        tg.pp_adatas(a_sc, a_sp, genes=None); a_sp.obs["rna_count_based_density"] = mass
        P = np.asarray(tg.map_cells_to_space(a_sc, a_sp, mode="cells", density_prior="rna_count_based", lambda_d=1.0, num_epochs=1000, device="cuda", random_state=0, verbose=False).X, float)
        np.save(out / "tangram.npy", P[ti].astype(np.float32)); print(short, "Tangram", flush=True)


def score():
    from scipy.stats import spearmanr
    import spotr.c2l_gw as M
    res = {}
    for short, SEC in SECS.items():
        out = OUT0 / short
        if not out.exists():
            continue
        z, ti, lab = load(short); sb = z["spot_barcodes"].astype(str); xy = np.asarray(z["xy"], float); ref = pd.read_csv(DATA / f"{SEC}_v2_reference.tsv", sep="\t").set_index("barcode").reindex(sb)["ref_ba"].to_numpy()
        oh = np.stack([(lab == "B").astype(float), (lab == "A").astype(float)], 1); r_ = {}; sec_sp = float(np.sqrt(((xy - xy.mean(0)) ** 2).sum(1).mean() / 2))
        for f in sorted(out.glob("*.npy")):
            nm = f.stem.replace("spotr_", "").replace("_phi", "")
            if f.stem.endswith("_abund"):
                continue
            P = np.load(f).astype(float); P = P / np.maximum(P.sum(1, keepdims=True), 1e-300); Pm = P.T @ oh; B, A = Pm[:, 0], Pm[:, 1]; has = (B + A) > 1e-9 * (B + A).max(); ok = np.isfinite(ref) & has
            sp = M.spreads(Pm, xy); r_[nm] = dict(rho_ba=float(spearmanr((B - A)[ok] / (B + A)[ok], ref[ok])[0]), spots=int(ok.sum()), spread_B=sp[0] / sec_sp, spread_A=sp[1] / sec_sp)
            js = out / f"spotr_{nm}.json"
            if js.exists():
                w = json.load(open(js)).get("weights", {}); r_[nm]["weights"] = {k: v for k, v in w.items() if isinstance(v, (float, int, list))} if isinstance(w, dict) else {}
        r_["reference"] = dict(B_share_of_reference_spots=float(np.mean(ref[np.isfinite(ref)] > 0)), B_share_of_traced_nuclei=float(np.mean(lab == "B")))
        res[short] = r_; print(short, json.dumps({k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "weights"} for k, v in r_.items()}, indent=0), flush=True)
    json.dump(res, open(OUT0 / "scores.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); OUT0.mkdir(parents=True, exist_ok=True)
    stage, shorts = sys.argv[1], (sys.argv[2:] or list(SECS))
    if stage == "score":
        score()
    else:
        for s_ in shorts:
            {"prep": prep, "spotr": run_spotr, "baselines": run_baselines}[stage](s_)
