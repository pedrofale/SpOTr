"""cell2location's map of HTAN, each spot coloured by the label it got most of (Pedro, 2026-11-01: "can you show me the cell2location mapping
results? Each spot colored by the majority label it got."), over the full-resolution H&E (403_; shown at 3.4 um per pixel), beside the section's
copy-number signal, the authors' clones (405_), SpOTr's map read the same way, and DestVI's (its per-spot proportions; 409_'s documented setup).

Majority label: cell2location -- the largest of its per-spot abundances over every reference label (the annotated cell types, the subclones, and
the tumour nuclei not assigned to a subclone, "tumour, unassigned").  SpOTr (subclone profiles; HT260C1 the copy-number tree, K3 clone labels,
which is what its clone tree is with two subclones) -- its abundances of the cell types, its tumour abundance split into the traced share mapped
to each subclone and the untraced share -- tumour cells of unknown subclone, a fixed share of every tumour spot in SpOTr's model -- spread over the
subclones in proportion to the map (as 385_).  cell2location's "tumour, unassigned" is a label of its own, with its own signature, and stays one.

Usage: python 404_htan_majority_labels.py   (benchmark_env; writes spotr_out/htan_majority/fig_<case>.{pdf,png})
"""
import json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; import sys; sys.path.insert(0, str(ROOT / "src")); RAW = HERE.parent / "htan_screen" / "raw"; O = HERE / "spotr_out"; OUT = O / "htan_majority"
INK = "#111111"; RED = 10
COL = {"A": "#e86a33", "B": "#0b3a7e", "B1": "#0b3a7e", "B2": "#7fb0ea", "tumour, unassigned": "#b9b8b2", "Fibroblasts": "#1baf7a", "Macrophages": "#eda100", "Endothelial": "#e87ba4",
       "T-cells": "#4cc1d0", "B-cells": "#6250d6", "Plasma": "#a35f2d", "Hepatocytes": "#b07fd0", "Cholangiocytes": "#7fd1b0", "DC": "#d4b106", "Pericytes": "#8c564b", "Luminal mature": "#f4a6c1"}


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def he(case, sec, tif):
    import tifffile
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None; sf = json.load(open(RAW / tif.parent.name / f"{sec}-scalefactors_json.json"))
    img = np.asarray(Image.fromarray(tifffile.imread(tif)).reduce(RED), float) / 255.0
    return dict(img=img, scalefactors=dict(tissue_lowres_scalef=1.0 / RED, spot_diameter_fullres=sf["spot_diameter_fullres"])), 55.0 / sf["spot_diameter_fullres"]


def majority(W, labs):
    W = np.asarray(W, float); return [labs[i] for i in W.argmax(1)], W / np.maximum(W.sum(1, keepdims=True), 1e-300)


def legend(ax, labels):
    for t in labels:
        ax.scatter([], [], c=COL.get(t, "#9a9994"), s=40, marker="s", label=t)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.01))


def panel(ax, pix, labels, title, img):
    from spotr.plot_spatial import slide
    slide(ax, pix, [COL.get(l, "#9a9994") for l in labels], title, image=img, size=1.6, img_alpha=0.5, fontsize=11)


def case_fig(case, sec, tif, ref_col, others, spotr):
    """others: [(name, npz with per-spot abundances or proportions 'w' over labels 'types')] -- cell2location, DestVI"""
    from spotr.plot_spatial import slide
    img, um = he(case, sec, tif); z = np.load(ROOT / "data" / "htan" / f"{sec}_v2.npz", allow_pickle=True); sb = z["spot_barcodes"].astype(str); pix = np.asarray(z["xy"], float) / um
    ref = pd.read_csv(ROOT / "data" / "htan" / f"{sec}_v2_reference.tsv", sep="\t").set_index("barcode").reindex(sb)[ref_col].to_numpy()
    others = [(nm, f) for nm, f in others if f.exists()]; n = 4 + len(others); fig, axs = plt.subplots(1, n, figsize=(8 * n, 9)); out = {}
    slide(axs[0], pix, None, f"{case}: H&E (full resolution, shown at 3.4 um per pixel)", image=img, fontsize=11)
    lim = float(np.nanquantile(np.abs(ref), 0.98)); slide(axs[1], pix, np.where(np.isfinite(ref), ref, np.nan), "copy-number signal, B minus A\n(genes no method fits)", image=img, size=1.6, img_alpha=0.5,
                                                          cmap="coolwarm", norm=TwoSlopeNorm(0.0, -lim, lim), fontsize=11)
    # the authors' clones per spot, digitised from their figure (405_): clone 1 is our B, clone 2 our A
    au = pd.read_csv(O / "authors_microregions" / f"{'K3' if 'K3' in case else 'HT260C1'}_spots.tsv", sep="\t").set_index("barcode").reindex(sb)["clone"].fillna("").astype(str).to_numpy()
    axs = list(axs); ax_a = axs.pop(2); slide(ax_a, pix, [{"clone_1": COL["B"], "clone_2": COL["A"]}.get(c, "#e4e2ea") for c in au], "the authors' clones (digitised from their Extended Data Fig. 9g)\nclone 1 (our B) navy, clone 2 (our A) orange",
                                            image=img, size=1.6, img_alpha=0.5, fontsize=11)
    # SpOTr
    Ab, types, P, subs, fT = spotr; ti = types.index("tumour"); sh = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    comp = np.concatenate([np.delete(Ab, ti, 1), Ab[:, [ti]] * sh], 1); labs_s = [t for t in types if t != "tumour"] + subs      # unassigned tumour spread over the subclones as mapped
    maj_s, fr_s = majority(comp, labs_s); panel(axs[2], pix, maj_s, "SpOTr: majority label per spot", img); out["SpOTr"] = pd.Series(maj_s).value_counts().to_dict()
    shown = set(maj_s)
    for j, (nm, f) in enumerate(others):
        o = np.load(f, allow_pickle=True); ty = [("tumour, unassigned" if str(t) == "untraced" else str(t)) for t in o["types"]]; maj_c, fr_c = majority(o["w"], ty)
        panel(axs[3 + j], pix, maj_c, f"{nm}: majority label per spot", img); out[nm] = pd.Series(maj_c).value_counts().to_dict(); shown |= set(maj_c)
    order = [t for t in COL if t in shown] + sorted(shown - set(COL))
    for ax in axs[2:]:
        legend(ax, order)
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / f"fig_{case}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"fig_{case}.png", dpi=70, bbox_inches="tight"); plt.close(fig)
    print(case, json.dumps(out), flush=True); return out


def main():
    res = {}
    # HT260C1: A, B1, B2; SpOTr = 386_'s copy-number tree (lineage form), refitted with the converged solver
    z = np.load(ROOT / "data" / "htan" / "HT260C1-Th1K1Fc2U1Z1Bs1_v2.npz", allow_pickle=True); tr = np.where(z["traced"])[0]; cl = z["sn_clone"].astype(str)[tr]; lab3 = np.where(cl == "A", "A", z["b_sub"].astype(str)[tr])
    oh = np.stack([lab3 == s for s in ("B1", "B2", "A")], 1).astype(float); types = json.load(open(O / "htan_v2" / "types.json"))["types"]; ct = np.asarray(z["cell_type"]).astype(str)
    fT = len(tr) / float((ct == "tumour").sum()); P = np.load(O / "htan_v2" / "spotr_lineagesubnb_phi.npy").astype(float).T @ oh; Ab = np.load(O / "htan_v2" / "spotr_lineagesubnb_abund.npy").astype(float)
    res["HT260C1"] = case_fig("HT260C1", "HT260C1-Th1K1Fc2U1Z1Bs1", RAW / "HT260C1" / "B1-HT260C1-Th1K1Fc2U1.tif", "ref_ba",
                              [("cell2location", O / "htan_v2" / "c2l.npz"), ("DestVI", O / "htan_v2" / "destvi_documented.npz")], (Ab, types, P, ["B1", "B2", "A"], fT))
    # HT268B1 K3: B, A; SpOTr = 387_'s clone labels (the solver-fixed refit)
    m387 = mod("m387", "387_htan_ht268b1.py"); z, ti, lab = m387.load("K3"); ct = np.asarray(z["cell_type"]).astype(str); fT = len(ti) / float((ct == "tumour").sum())
    types = json.load(open(O / "ht268b1" / "K3" / "types.json"))["types"]; P = np.load(O / "ht268b1" / "K3" / "spotr_clones_phi.npy").astype(float).T @ np.stack([lab == "B", lab == "A"], 1).astype(float)
    Ab = np.load(O / "ht268b1" / "K3" / "spotr_clones_abund.npy").astype(float)
    res["HT268B1_K3"] = case_fig("HT268B1_K3", "HT268B1-Th1K3Fc2U1Z1Bs1", RAW / "HT268B1" / "D1-HT268B1-Th1K3Fc2U1.tif", "ref_ba",
                                 [("cell2location", O / "c2l_package" / "htan_HT268B1K3v2_out.npz"), ("DestVI", O / "ht268b1" / "K3" / "destvi_documented.npz")], (Ab, types, P, ["B", "A"], fT))
    json.dump(res, open(OUT / "majority_counts.json", "w"), indent=1)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
