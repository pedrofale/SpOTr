"""355_: cell2location alone, as its authors ship it (the cell2location package, not 328_'s transcription).  2026-10-30.
Pedro, 2026-10-30: "cell2location alone should not be our version of their package. It should be their package."
Workflow, the package's documented one: reference = the dissociated cells, labelled as 328_ / 335_ labelled them (every clone its own type,
the untraced cells of a traced kind as one type, every other annotated cell type its own); genes by cell2location.utils.filtering.filter_genes
(cell_count_cutoff 5, cell_percentage_cutoff2 0.03, nonz_mean_cutoff 1.12) on the reference, shared with the spots; signatures by
cell2location.models.RegressionModel (max_epochs 250, as the tutorials), exported as means_per_cluster_mu_fg; the spatial model
cell2location.models.Cell2location (N_cells_per_location as 328_ / 335_: the mean number of cells per spot in the simulations and on PEtracer,
10 otherwise; detection_alpha 20), trained for 30,000 epochs on the whole data; each clone's abundance its posterior mean
(means_cell_abundance_w_sf).  Three stages, so that the package runs in its own environment (conda env c2l, cell2location from PyPI):
  python 355_c2l_package.py export [sim petracer embryo htan]    (benchmark_env: writes spotr_out/c2l_package/<problem>_in.npz)
  python 355_c2l_package.py fit [k n]                            (env c2l: writes <problem>_out.npz with the abundances; shard k of n)
  python 355_c2l_package.py score                                (benchmark_env: writes <problem>.json, masses["c2l"] and the scores)
"""
import os, sys, json, glob, time, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE / "spotr_out"; OUT = O / "c2l_package"; DATA = HERE.parents[1] / "data"
SIMS = [(w, s) for s in [46, 47] for w in ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]]; PET = ["half:M3-4", "half:M3-1", "half:M3-2", "serial:M3-1>M3-2", "half-visium:M3-4"]   # half-visium: Visium spots (2026-11-02)


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def stem(src, name):
    return {"sim": lambda: f"{name[0]}_{name[1]}", "petracer": lambda: "petracer_" + name.replace(":", "_").replace(">", "_to_"), "embryo": lambda: "embryo_e95",
            "htan": lambda: "htan_HT260C1"}[src]()


def export(src, name):
    """the reference counts and labels, the spots' counts, N_cells_per_location and the clone columns, as 328_ / 335_ built them"""
    if src in ("sim", "petracer"):
        if src == "sim":
            w, seed = name; NB, X, Xc, t_f, _ = mod("m268", "268_lineage_cv.py").build(seed, **mod("m270", "270_lineage_cv_concordance.py").WORLDS[w]); Y = X["Sc"]
        else:
            P = mod("m276", "276_petracer.py").build(name, 0); X, Xc, Y = P["X"], P["Xc"], P["Y"]
        tum = X["kind"] == 0; tr = tum & X["traced"]; me = np.asarray(X["me_type"]); cl = X["cl"]; K = X["K"]
        labs = np.array([f"me{e}" for e in me]); labs[tum & ~tr] = "untraced"; labs[tr] = [f"c{c}" for c in cl[tr]]
        clones = [f"c{k}" for k in range(K)]; nhat = float(np.mean(X["ncell"])); genes = np.array([f"g{j}" for j in range(np.asarray(Xc).shape[1])])
    else:
        sec = "embryo/embryo2a_e95.npz" if src == "embryo" else "htan/HT260C1-Th1K1Fc2U1Z1Bs1.npz"; z = dict(np.load(DATA / sec, allow_pickle=True))
        Xc, Y = z["Xc_fit"].astype(float), z["Ys_fit"].astype(float); ct = np.asarray(z["cell_type"]).astype(str); genes = np.asarray(z["fit_genes"]).astype(str)
        inform = np.asarray(z["clone_informative"], bool); clone_all = z["clone"].astype(int); traced = inform[clone_all]; Kc = int(inform.sum())
        remap = -np.ones(len(inform), int); remap[inform] = np.arange(Kc); tk = sorted({k for k in ct[traced]})
        labs = np.where(np.isin(ct, tk), "untraced", ct); labs[traced] = [f"c{remap[c]}" for c in clone_all[traced]]; clones = [f"c{k}" for k in range(Kc)]; nhat = 10.0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{stem(src, name)}_in.npz", Xc=np.asarray(Xc, np.float32), Y=np.asarray(Y, np.float32), labels=labs, clones=np.array(clones), nhat=nhat, genes=genes,
                        src=src, name=json.dumps(name))


def fit_one(f_in):
    """cell2location's documented workflow on one exported problem"""
    import anndata as ad, pandas as pd, cell2location
    from cell2location.models import RegressionModel, Cell2location
    from cell2location.utils.filtering import filter_genes
    z = np.load(f_in, allow_pickle=True); genes = [str(g) for g in z["genes"]]
    ref = ad.AnnData(X=np.asarray(z["Xc"], np.float32), obs=pd.DataFrame({"label": [str(l) for l in z["labels"]]}, index=[f"cell{i}" for i in range(len(z["labels"]))]), var=pd.DataFrame(index=genes))
    vis = ad.AnnData(X=np.asarray(z["Y"], np.float32), obs=pd.DataFrame(index=[f"spot{i}" for i in range(len(z["Y"]))]), var=pd.DataFrame(index=genes))
    sel = filter_genes(ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12); ref = ref[:, sel].copy()
    RegressionModel.setup_anndata(adata=ref, labels_key="label"); rm = RegressionModel(ref); rm.train(max_epochs=250)
    ref = rm.export_posterior(ref, sample_kwargs={"num_samples": 1000, "batch_size": 2500})
    cols = [f"means_per_cluster_mu_fg_{c}" for c in ref.uns["mod"]["factor_names"]]; inf_aver = ref.varm["means_per_cluster_mu_fg"][cols].copy(); inf_aver.columns = ref.uns["mod"]["factor_names"]
    shared = np.intersect1d(vis.var_names, inf_aver.index); vis = vis[:, shared].copy(); inf_aver = inf_aver.loc[shared, :]
    Cell2location.setup_anndata(adata=vis); m = Cell2location(vis, cell_state_df=inf_aver, N_cells_per_location=float(z["nhat"]), detection_alpha=20)
    m.train(max_epochs=30000, batch_size=None, train_size=1); vis = m.export_posterior(vis, sample_kwargs={"num_samples": 1000, "batch_size": m.adata.n_obs})
    W = vis.obsm["means_cell_abundance_w_sf"]; names = [c.replace("meanscell_abundance_w_sf_", "") for c in W.columns]
    np.savez_compressed(str(f_in).replace("_in.npz", "_out.npz"), w=W.to_numpy(np.float32), types=np.array(names), genes_used=np.array(shared), version=cell2location.__version__,
                        elbo=np.asarray(m.history["elbo_train"]).ravel().astype(np.float32))


def score_one(f_in):
    z = np.load(f_in, allow_pickle=True); o = np.load(str(f_in).replace("_in.npz", "_out.npz"), allow_pickle=True); src, name = str(z["src"]), json.loads(str(z["name"]))
    name = tuple(name) if isinstance(name, list) else name; types = [str(t) for t in o["types"]]; w = np.asarray(o["w"], float)
    Pm = np.stack([w[:, types.index(c)] if c in types else np.zeros(len(w)) for c in [str(c) for c in z["clones"]]], 1); s = {}
    if src == "sim":
        m239, m189 = mod("m239", "239_deconv_fgw.py"), mod("m189", "189_me_sim.py"); w_, seed = name; X = mod("m268", "268_lineage_cv.py").build(seed, **mod("m270", "270_lineage_cv_concordance.py").WORLDS[w_])[1]
        s = {k: v for k, v in m239.readout(None, X, m239.window(X), m189, P_clone=Pm).items() if k in ("clone_tv",)}
    elif src == "petracer":
        s = mod("f289", "289_field_metrics.py").scores(Pm, mod("m276", "276_petracer.py").build(name, 0)["X"]["Ncl"])
    elif src == "htan":
        s = mod("m335", "335_htan_c2l.py").ref_scores("HT260C1-Th1K1Fc2U1Z1Bs1", Pm)
    out = dict(version=str(o["version"]), genes_used=int(len(o["genes_used"])), scores=s, masses={"c2l": Pm.tolist()}); json.dump(out, open(str(f_in).replace("_in.npz", ".json"), "w"), default=float)
    print(stem(src, name), {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    stage = sys.argv[1]
    if stage == "export":
        which = sys.argv[2:] or ["sim", "petracer", "embryo", "htan"]
        jobs = {"sim": [("sim", ws) for ws in SIMS], "petracer": [("petracer", p_) for p_ in PET], "embryo": [("embryo", "e95")], "htan": [("htan", "HT260C1")]}
        for w in which:
            for src, name in jobs[w]:
                if not (OUT / f"{stem(src, name)}_in.npz").exists():
                    export(src, name); print("exported", stem(src, name), flush=True)
    elif stage == "fit":                                                                    # optional shard k of n: python 355_ fit k n
        k, n = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
        for i, f in enumerate(sorted(glob.glob(str(OUT / "*_in.npz")))):
            if i % n != k:
                continue
            if not Path(f.replace("_in.npz", "_out.npz")).exists():
                t0 = time.time(); fit_one(f); print("fitted", Path(f).name, f"{time.time() - t0:.0f}s", flush=True)
    elif stage == "score":
        for f in sorted(glob.glob(str(OUT / "*_in.npz"))):
            if Path(f.replace("_in.npz", "_out.npz")).exists():
                score_one(f)
