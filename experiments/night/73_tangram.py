"""Tangram (Biancalani et al. 2021) on the same splits as 70_spotr.py (PEtracer) and 71_etracer.py (eTRACER).

The standard expression-only mapper: cells mode, all shared genes, density term on (lambda_d = 1) with the same
spot marginal SpOTr uses (cells per spot for PEtracer, bins per spot for eTRACER).  It sees neither the tree nor
the anchor.  Scored with the same metrics: PEtracer placement error and dominant-clade agreement; eTRACER
clone-density agreement on the non-anchor spots of each seed.
Usage: python 73_tangram.py   (benchmark_env, GPU; writes spotr_out/tangram.json)
       LAMBDA_D=0 OUTJSON=tangram_ld0.json python 73_tangram.py   (Tangram's default: no density term)
       PSEEDS=4,5,6,7 ESEEDS= OUTJSON=tangram_s47.json python 73_tangram.py   (the confirmation splits)
       PSEEDS= ESEEDS=3,4,5,6,7 OUTJSON=tangram_et37.json python 73_tangram.py   (eTRACER, more anchor draws)
"""
import json, time, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def tangram(Xsc, Xst, q, seed, epochs=1000, lambda_d=1.0):
    import anndata as ad, tangram as tg
    genes = [f"g{i}" for i in range(Xsc.shape[1])]
    a_sc = ad.AnnData(np.asarray(Xsc, np.float32)); a_sc.var_names = genes
    a_sp = ad.AnnData(np.asarray(Xst, np.float32)); a_sp.var_names = genes
    tg.pp_adatas(a_sc, a_sp, genes=None)
    a_sp.obs["rna_count_based_density"] = np.asarray(q, float) / np.sum(q)   # 1.0.4 cannot take an array prior
    amap = tg.map_cells_to_space(a_sc, a_sp, mode="cells", density_prior="rna_count_based", lambda_d=lambda_d,
                                 num_epochs=epochs, device=__import__("os").environ.get("TANGRAM_DEVICE", "cuda"), random_state=seed, verbose=False)
    return np.asarray(amap.X, float)


if __name__ == "__main__":
    import os, ot
    LD = float(os.environ.get("LAMBDA_D", 1.0))             # 1 = density term on (main run); 0 = Tangram's default
    PSEEDS = [int(x) for x in os.environ.get("PSEEDS", "0,1,2,3").split(",") if x]     # PEtracer splits
    ESEEDS = [int(x) for x in os.environ.get("ESEEDS", "0,1,2").split(",") if x]       # eTRACER anchor draws
    m70 = mod("m70", "70_spotr.py"); m71 = mod("m71", "71_etracer.py")
    res, t0 = [], time.time()
    for sec in ["M3-4", "M3-1", "M3-2"]:
        z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
        for seed in PSEEDS:
            X = m70.data_real(sec, seed)
            n_all = len(z["C_tree"]); use = np.random.default_rng(seed).permutation(n_all)[n_all // 2:]
            Xsc = np.asarray(z["X_sc"], float)[use]; Xst = np.asarray(z["X_st"], float)
            Mc = ot.dist(Xsc, Xst)
            assert np.allclose(Mc / Mc.max(), X["M"] / X["M"].max()), "split differs from 70_spotr.py"
            T = tangram(Xsc, Xst, X["q"], seed, lambda_d=LD)
            r = dict(data="PEtracer", sec=sec, seed=seed, arm="tangram", lambda_d=LD, err=m70.err(T, X), agree=m70.agree(T, X))
            res.append(r); print(r, f"{time.time() - t0:.0f}s", flush=True)
    for tag in ["LE", "EE"]:
        d = np.load(HERE / f"etracer_spots_{tag}.npz", allow_pickle=True)
        for seed in ESEEDS:
            X = m71.data_etracer(tag, seed)
            T = tangram(d["X_sc"], d["X_st"], X["q"], seed, lambda_d=LD)
            r = dict(data="eTRACER", tag=tag, seed=seed, arm="tangram", lambda_d=LD, dens=m71.density_agreement(T, X))
            res.append(r); print(r, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(HERE / "spotr_out" / os.environ.get("OUTJSON", "tangram.json"), "w"), indent=1)
    pe = [r for r in res if r["data"] == "PEtracer"]
    if pe: print(f"PEtracer Tangram: error {np.mean([r['err'] for r in pe]):.1f} um, agreement {np.mean([r['agree'] for r in pe]):.3f}")
    for tag in ["LE", "EE"] if ESEEDS else []:
        print(f"eTRACER {tag} Tangram: {np.mean([r['dens'] for r in res if r.get('tag') == tag]):.3f}")
