"""HT260C1 against its H&E: where is the tumour, and where do the maps put it? (Pedro, 2026-10-31: "spotr is still putting too much mass on
cancer clones in regions that don't seem to have copy number signal. I see that the tumor purity is always above 0.5 though ... Do we have the
pathology slide?")

The section's low-resolution H&E (HTAN Level 3 Visium files, experiments/htan_screen/raw/HT260C1/) with, per spot in tissue: the authors'
tumour purity; marker scores from the spot's own counts (scanpy score_genes on log1p CP10k; epithelial: EPCAM, KRT8, KRT18, KRT19, KRT20,
CDH17, CEACAM5, CEACAM6, LGALS4, PHGR1; stromal: COL1A1, COL1A2, COL3A1, COL6A3, DCN, LUM, FBLN1, SPARC, FN1, TAGLN); the tumour share of
each spot's cell abundances in SpOTr's map (clone labels; 347_, spotr_out/spotr_c2l_cgw_split) and in cell2location's (the package, 355_);
SpOTr's B share; and the copy-number signal (the reference, higher-purity half only).  Plots with squidpy.

Usage: python 359_htan_morphology.py   (benchmark_env; writes spotr_out/htan_morphology/)
"""
import json, gzip
from pathlib import Path
import numpy as np, pandas as pd, scipy.io, scipy.sparse
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; SCR = HERE.parent / "htan_screen"; RAW = SCR / "raw" / "HT260C1"; P = "HT260C1-Th1K1Fc2U1Z1Bs1"
O = HERE / "spotr_out"; OUT = O / "htan_morphology"; OUT.mkdir(parents=True, exist_ok=True); LIB = "HT260C1"
EPI = ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT20", "CDH17", "CEACAM5", "CEACAM6", "LGALS4", "PHGR1"]
STROMA = ["COL1A1", "COL1A2", "COL3A1", "COL6A3", "DCN", "LUM", "FBLN1", "SPARC", "FN1", "TAGLN"]


def visium():
    import anndata as ad
    from PIL import Image
    X = scipy.sparse.csr_matrix(scipy.io.mmread(gzip.open(RAW / f"{P}-matrix.mtx.gz")).T)
    f = pd.read_csv(RAW / f"{P}-features.tsv.gz", sep="\t", header=None); b = pd.read_csv(RAW / f"{P}-barcodes.tsv.gz", header=None)[0].to_numpy()
    pos = pd.read_csv(RAW / f"{P}-tissue_positions_list.csv", header=None, names=["bc", "in", "r", "c", "py", "px"]).set_index("bc").reindex(b)
    a = ad.AnnData(X.astype(np.float32), obs=pd.DataFrame(index=b), var=pd.DataFrame(index=pd.Index(f[1].to_numpy()).astype(str)))
    a.var_names_make_unique(); a = a[pos["in"].to_numpy() == 1].copy(); a.obsm["spatial"] = pos.loc[a.obs_names, ["px", "py"]].to_numpy(float)
    a.uns["spatial"] = {LIB: dict(images=dict(lowres=np.asarray(Image.open(RAW / f"{P}-tissue_lowres_image.png").convert("RGB")) / 255.0),
                                  scalefactors=json.load(open(RAW / f"{P}-scalefactors_json.json")))}
    return a


def main():
    import scanpy as sc, squidpy as sq
    from scipy.stats import spearmanr
    a = visium(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.tl.score_genes(a, [g for g in EPI if g in a.var_names], score_name="epithelial score", random_state=0)
    sc.tl.score_genes(a, [g for g in STROMA if g in a.var_names], score_name="stromal score", random_state=0)
    meta = pd.read_csv(SCR / f"{P}_st_meta.tsv", sep="\t").set_index("barcode"); a.obs["authors' purity"] = meta["Tumor_purity_estimate"].reindex(a.obs_names).to_numpy(float)
    z = np.load(ROOT / "data" / "htan" / f"{P}.npz", allow_pickle=True); sb = z["spot_barcodes"].astype(str)
    types = json.load(open(O / "spotr_c2l_cgw_split" / "htan_HT260C1.json"))["meta"]["types"]; ti = types.index("tumour")
    A = np.load(O / "spotr_c2l_cgw_split" / "htan_HT260C1_clones_abund.npy").astype(float); fr = A[:, ti] / A.sum(1)
    phi = np.load(O / "spotr_c2l_cgw_split" / "htan_HT260C1_clones_phi.npy").astype(float)
    o = json.load(open(O / "spotr_c2l_cgw_split" / "htan_HT260C1.json")); m = np.asarray(o["masses"]["clones"], float)[:, :2]; bsh = m[:, 0] / np.maximum(m.sum(1), 1e-300)
    c = np.load(O / "c2l_package" / "htan_HT260C1_out.npz", allow_pickle=True); ct = [str(t) for t in c["types"]]; w = c["w"].astype(float)
    tum_c = [ct.index(t) for t in ["c0", "c1", "untraced"]]; fr_c = w[:, tum_c].sum(1) / w.sum(1); fib = A[:, types.index("Fibroblasts")] / A.sum(1)
    ref = pd.read_csv(ROOT / "data" / "htan" / f"{P}_reference.tsv", sep="\t").set_index("barcode")["ref_c2"]
    for k, v in {"SpOTr: tumour share": fr, "cell2location: tumour share": fr_c, "SpOTr: fibroblast share": fib, "SpOTr: B share of tumour": bsh}.items():
        a.obs[k] = pd.Series(v, index=sb).reindex(a.obs_names).to_numpy(float)
    a.obs["copy-number signal for B"] = ref.reindex(a.obs_names).to_numpy(float)
    keys = ["authors' purity", "epithelial score", "stromal score", "SpOTr: tumour share", "cell2location: tumour share", "SpOTr: fibroblast share", "SpOTr: B share of tumour", "copy-number signal for B"]
    ok = a.obs[keys[:6]].notna().all(1); rho = {}
    for k in ["authors' purity", "SpOTr: tumour share", "cell2location: tumour share"]:
        rho[k] = dict(epithelial=float(spearmanr(a.obs.loc[ok, k], a.obs.loc[ok, "epithelial score"])[0]), stromal=float(spearmanr(a.obs.loc[ok, k], a.obs.loc[ok, "stromal score"])[0]))
    q = a.obs.loc[ok, "stromal score"]; hi = ok & (a.obs["stromal score"] >= q.quantile(0.75)); lo = ok & (a.obs["stromal score"] <= q.quantile(0.25))
    summ = dict(spots=int(ok.sum()), rho=rho, stroma_top_quartile={k: float(a.obs.loc[hi, k].median()) for k in keys[:6]}, stroma_bottom_quartile={k: float(a.obs.loc[lo, k].median()) for k in keys[:6]},
                share_spots_tumour_share_above_half_in_stroma_top_quartile=float((a.obs.loc[hi, "SpOTr: tumour share"] > 0.5).mean()))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
    sq.pl.spatial_scatter(a, color=None, library_id=LIB, img_res_key="lowres", title="H&E (low resolution)", figsize=(6, 6), save=None); plt.savefig(OUT / "he.png", dpi=110, bbox_inches="tight"); plt.close("all")
    from matplotlib.colors import Normalize, TwoSlopeNorm
    cmaps = ["Greys", "Purples", "Greens", "Reds", "Reds", "Greens", "coolwarm", "coolwarm"]; lim = float(np.nanquantile(np.abs(a.obs["copy-number signal for B"]), 0.98))
    norms = [None] * 6 + [Normalize(0, 1), TwoSlopeNorm(0.0, -lim, lim)]
    fig, axs = plt.subplots(2, 4, figsize=(22, 11))
    for ax, k, cm, nm in zip(axs.ravel(), keys, cmaps, norms):
        sq.pl.spatial_scatter(a, color=k, library_id=LIB, img_res_key="lowres", cmap=cm, norm=nm, size=1.3, alpha=0.85, na_color=(0.0, 0.0, 0.0, 0.0), ax=ax, title=k)
    fig.savefig(OUT / "htan_morphology.png", dpi=90, bbox_inches="tight"); fig.savefig(OUT / "htan_morphology.pdf", bbox_inches="tight"); plt.close("all")


if __name__ == "__main__":
    main()
