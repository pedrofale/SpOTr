"""The paper's PEtracer figure (spotr.tex, fig:petracer), with the results we have as placeholders (Pedro, 2026-11-02: "Let's make the figures
for PETracer and the colorectal cancer, and use whatever results we have currently as placeholders").

Section M3-4, half design: one random half of the section's cells measured with Visium spots (276_ half-visium: 55 um on a hexagonal
lattice 100 um apart, cells between spots not measured; `square` as the argument: the earlier 55 um bins tiling the section), the other half
the dissociated sample.
(A) The truth from the in situ lineage readout and (B-F) each method's map, every spot coloured by its majority label across ALL labels -- the 17
    microenvironment types (muted shades), the 18 clades and the pooled small clades (bright), the unassigned tumour (grey) -- never among the
    clades only: cell2location (the package, 355_'s run: every type's abundance), DestVI and scDOT (pending: not yet run on PEtracer), Tangram
    (the package, rerun here once to keep every cell's mapping; cached), SpOTr with the tree (450_'s quick check: the Methods' settings,
    distance matching at alpha 0.2).
(A) The single cells of the section's half that the spots were built from, at their in situ positions, by clade (non-tumor pooled, tumor
    without a readout grey; Pedro, 2026-11-02: "show in Panel A the actual single cell spatial data ... so that it's clear we have ground
    truth data that we generated Visium from").  (B-G) the Visium truth and each method's map, letters shifted by one; metrics H-K.
(B-G) are coloured in two levels (Pedro, 2026-11-02): non-tumor (every type pooled) where non-tumor cells outnumber tumor cells, traced or
    not; otherwise the majority clade among the cells with a lineage readout (69% of tumor cells have none, so they count as tumor but
    cannot name a clade).  Spots drawn as circles at 1.6 x the spot diameter, as the HTAN panels.
    SpOTr comes first, in orange.
(G-J) Per-clone agreement with the truth on the four standard metrics of Li et al. 2022 (as 289_field_metrics.scores, per clone): PCC, SSIM
    (higher is better), RMSE, JS (lower is better); boxes with one point per clone; SpOTr last, in its own colour.
Style: decision-writing-style (bold uppercase panel letters, plain titles).

Usage: python 451_fig_petracer_paper.py [square]   (benchmark_env; needs 450_ quick petracer-visium and 355_'s half-visium fit; the Tangram rerun uses the GPU; writes SpOTr-overleaf/figures/fig_petracer_methods.{pdf,png})
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; O = HERE / "spotr_out"; sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT.parent / "SpOTr-overleaf" / "figures"; CACHE = O / "paper_figures"; INK = "#111111"; GRID = "#e6e5e0"
VIS = "square" not in sys.argv[1:]; SECTION = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("section=")), "M3-4")   # other sections: a draft figure in spotr_out/paper_figures                                                       # Visium spots (default since 2026-11-02); `square`: the 55 um bins
NAME, TAG, CASE = (f"half-visium:{SECTION}", f"half-visium_{SECTION}", f"PEtracer_{SECTION}_visium_share") if VIS else (f"half:{SECTION}", f"half_{SECTION}", f"PEtracer_{SECTION}")
METHODS = [("SpOTr", "#F47B20"), ("cell2location", "#1f77b4"), ("DestVI", "#9467bd"), ("Tangram", "#2ca02c"), ("scDOT", "#d62728")]   # SpOTr orange and first (decision-writing-style)


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


PANELS = []                                                                                   # (axes, letter), laid out by spotr.figlayout.finish


def letter(ax, s, x=None, y=None):
    """registers the panel; spotr.figlayout.finish puts the letter on its column's edge and aligns the panels with the figure's edges"""
    PANELS.append((ax, s))


def per_clone(f289, P_map, N_true, min_cells):
    """289_field_metrics.scores, per clone instead of averaged"""
    P_map, N_true = np.asarray(P_map, float), np.asarray(N_true, float); h = N_true.sum(1) > 0
    Q = N_true[h] / N_true[h].sum(1, keepdims=True); P = P_map[h] / np.maximum(P_map[h].sum(1, keepdims=True), 1e-300)
    ks = [k for k in range(Q.shape[1]) if N_true[:, k].sum() >= min_cells and Q[:, k].std() > 0]; out = dict(PCC=[], SSIM=[], RMSE=[], JS=[])
    for k in ks:
        a, b = Q[:, k], P[:, k]
        out["PCC"].append(float(np.corrcoef(a, b)[0, 1]) if b.std() > 0 else 0.0); out["SSIM"].append(f289._ssim(a, b)); out["RMSE"].append(f289._rmse(a, b)); out["JS"].append(f289._js(a, b))
    return out


def tangram_full(B, X):
    """Tangram (the package, 400_'s settings) on every dissociated cell; each spot's mapped cells per label, cached"""
    fn = CACHE / f"petracer_{TAG}_tangram_full.npz"
    if fn.exists():
        return np.load(fn)["comp"]
    import anndata as ad, tangram as tg
    Xc = np.asarray(B["Xc"], float); Y = np.asarray(B["Y"], float); genes = [f"g{i}" for i in range(Xc.shape[1])]
    a_sc = ad.AnnData(Xc.astype(np.float32)); a_sc.var_names = genes; a_sp = ad.AnnData(Y.astype(np.float32)); a_sp.var_names = genes
    tg.pp_adatas(a_sc, a_sp, genes=None); a_sp.obs["rna_count_based_density"] = np.full(len(Y), 1.0 / len(Y))
    Pc = np.asarray(tg.map_cells_to_space(a_sc, a_sp, mode="cells", density_prior="rna_count_based", lambda_d=1.0, num_epochs=1000, device="cuda", random_state=0, verbose=False).X, float)
    comp = Pc.T @ label_onehot(B, X); CACHE.mkdir(parents=True, exist_ok=True); np.savez_compressed(fn, comp=comp.astype(np.float32)); return comp


def labels_of(B, X):
    """one label per dissociated cell, in the order of LABELS"""
    tum = X["kind"] == 0; tr = tum & X["traced"]; me = np.asarray(X["me_type"]); lab = np.array([f"me{e}" for e in me], dtype=object)
    lab[tum & ~X["traced"]] = "untraced"; lab[tr] = [f"c{c}" for c in X["cl"][tr]]; return lab.astype(str)


def label_onehot(B, X):
    lab = labels_of(B, X); keys = label_keys(X); return np.stack([lab == k for k in keys], 1).astype(float)


def label_keys(X):
    return [f"c{c}" for c in range(X["K"])] + [f"me{e}" for e in range(X["Nme"].shape[1])] + ["untraced"]


def main():
    import warnings; warnings.filterwarnings("ignore")
    from spotr.plot_spatial import slide, cells
    m276, f289 = mod("m276", "276_petracer.py"), mod("f289", "289_field_metrics.py")
    B = m276.build(NAME, 0); X = B["X"]; K = X["K"]; xy = np.asarray(X["spots"], float); meta = B["meta"]; keys = label_keys(X)
    names = {f"c{c}": (f"clade {c + 1}" if c < K - 1 else "small clades, pooled") for c in range(K)}
    names.update({f"me{e}": str(n) for e, n in enumerate(meta["others"])}); names["untraced"] = "tumor, unassigned"
    tab = plt.get_cmap("tab20"); pal = [to_hex(tab(i)) for i in list(range(0, 20, 2)) + list(range(1, 20, 2)) if i not in (14, 15)]   # tab20 without its greys
    col = {f"c{c}": pal[c % len(pal)] for c in range(K - 1)}; col[f"c{K - 1}"] = "#e0b0d0"
    muted = ["#d8d2c4", "#a89f8e", "#c9c0ad", "#8f8676", "#e2dccf", "#b5ac99", "#9c9381", "#cfc7b6", "#7f7768", "#ddd6c7", "#aea593", "#c2b9a6",
             "#958c7b", "#d5cdbd", "#a39a88", "#bdb4a1", "#877f70"]
    col.update({f"me{e}": muted[e % len(muted)] for e in range(X["Nme"].shape[1])}); col["untraced"] = "#5f5f5f"
    truth = np.concatenate([X["Ncl"], X["Nme"], np.maximum(X["Ntum"] - X["Ncl"].sum(1), 0)[:, None]], 1)
    # cell2location: the package's abundance of every type
    fc = O / "c2l_package" / f"petracer_{TAG}_out.npz"; c2l = None                       # pending until the package has run (it is slow: run once the story has converged)
    if fc.exists():
        c = np.load(O / "c2l_package" / f"petracer_{TAG}_out.npz", allow_pickle=True); ty = [str(t) for t in c["types"]]; Wc = np.asarray(c["w"], float)
        c2l = np.stack([Wc[:, ty.index(k)] if k in ty else np.zeros(len(Wc)) for k in keys], 1)
    # SpOTr (450_ quick check): microenvironment and unassigned abundances, the cancer cells' clades through the map
    spotr = None; fs = O / "writeup_check" / f"{CASE}_tree_0.2.npz"
    if fs.exists():
        z = np.load(fs); phi = z["phi"].astype(float); Ab = z["abund"].astype(float); tum = X["kind"] == 0; tr = tum & X["traced"]
        me_types = sorted(set(np.asarray(X["me_type"])[~tum].tolist())); E = len(me_types); share = phi / np.maximum(phi.sum(0, keepdims=True), 1e-300)
        cols_ = {f"me{e}": Ab[:, i] for i, e in enumerate(me_types)}; cols_["untraced"] = Ab[:, E]; trd = Ab[:, E + 1]; cl_tr = X["cl"][tr]
        cols_.update({f"c{k}": share[cl_tr == k].sum(0) * trd for k in range(K)}); spotr = np.stack([cols_.get(k, np.zeros(len(xy))) for k in keys], 1)
    tg_comp = tangram_full(B, X)
    pk = {m: (np.load(CACHE / f"petracer_{TAG}_{m.lower()}.npz")["comp"].astype(float) if (CACHE / f"petracer_{TAG}_{m.lower()}.npz").exists() else None) for m in ("DestVI", "scDOT")}   # 458_
    maps = [("SpOTr", spotr), ("cell2location", c2l), ("DestVI", pk["DestVI"]), ("Tangram", tg_comp), ("scDOT", pk["scDOT"])]
    fig = plt.figure(figsize=(28, 9.5)); gs = fig.add_gridspec(2, 8, height_ratios=[1, 0.95], width_ratios=[1] * 7 + [0.75], hspace=0.32, wspace=0.08)   # not a whole page (Pedro)
    me_idx = [i for i, k in enumerate(keys) if k.startswith("me")]; cl_idx = [i for i, k in enumerate(keys) if k.startswith("c")]; dkeys = [keys[i] for i in cl_idx] + ["nontumor"]
    col["nontumor"] = "#cfc6b4"; names["nontumor"] = "non-tumor cells"
    un_idx = keys.index("untraced"); col["notrace"] = "#a3a3a3"; names["notrace"] = "tumor, no readout"
    def majority(comp):
        """two levels (Pedro, 2026-11-02): a spot is non-tumor when its non-tumor cells (every type pooled) outnumber its tumor cells (traced or
        not), else it takes its majority clade among the cells with a lineage readout (tumor without any such cell: its own grey)"""
        comp = np.asarray(comp, float); C = comp[:, cl_idx]; tum = C.sum(1) + comp[:, un_idx]; non = comp[:, me_idx].sum(1)
        lab = np.where(non >= tum, "nontumor", np.where(C.sum(1) > 1e-9, np.array([keys[cl_idx[i]] for i in C.argmax(1)]), "notrace"))
        return [col[l] if t > 1e-6 else "#ffffff" for l, t in zip(lab, tum + non)], set(lab.tolist())
    # (A) the single cells of the section's half that the Visium spots were built from (276_.build's split: its first draw, seed 0)
    zc = np.load(ROOT / "data" / "petracer" / f"petracer_{SECTION}.npz", allow_pickle=True); n_all = len(zc["xy"]); half = np.random.default_rng(0).permutation(n_all)[: n_all // 2]
    cx, mal, trc, cld = zc["xy"][half].astype(float), zc["malignant"][half].astype(bool), zc["traced"][half].astype(bool), zc["clade"][half]
    cc_ = np.array([col[f"c{c}"] if t else col["notrace"] for c, t in zip(cld, trc)], dtype=object); cc_[~mal] = col["nontumor"]
    o_ = np.argsort(np.where(~mal, 0, np.where(trc, 2, 1)), kind="stable")                  # non-tumor, then tumor without a readout, traced cells on top
    pad = 1.6 * 55.0 / 2; ext = (xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax = fig.add_subplot(gs[0, 0]); letter(ax, "A"); cells(ax, cx[o_], list(cc_[o_]), "In situ cells", size=1.2, fontsize=11, extent=ext)
    seen = {"nontumor", "notrace"} | {f"c{c}" for c in np.unique(cld[mal & trc])}
    slots = [gs[0, j] for j in range(1, 7)]
    for j, (ttl, comp) in enumerate([("Synthetic Visium", truth)] + maps):
        ax = fig.add_subplot(slots[j]); letter(ax, "BCDEFG"[j])
        if comp is None:
            ax.axis("off"); ax.text(0.5, 0.5, f"{ttl}\n(pending)", ha="center", va="center", fontsize=11, color="#7a7a7a", transform=ax.transAxes); continue
        cs, s_ = majority(comp); seen |= s_; slide(ax, xy, cs, ttl, fontsize=11, **(dict(pitch=1.6 * 55.0, shape="circle") if VIS else dict(pitch=55.0, shape="square")))   # 1.6: as the HTAN panels
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
    # per-clone metrics
    onehot = np.eye(K)[X["cl"][(X["kind"] == 0) & X["traced"]]]
    clade = lambda comp: None if comp is None else np.asarray(comp, float)[:, :K]
    pc = {"cell2location": per_clone(f289, clade(c2l), X["Ncl"], f289.MIN_CELLS) if c2l is not None else None, "Tangram": per_clone(f289, clade(tg_comp), X["Ncl"], f289.MIN_CELLS),
          "SpOTr": per_clone(f289, clade(spotr), X["Ncl"], f289.MIN_CELLS) if spotr is not None else None,
          **{m: (per_clone(f289, clade(pk[m]), X["Ncl"], f289.MIN_CELLS) if pk[m] is not None else None) for m in ("DestVI", "scDOT")}}
    gs2 = gs[1, :7].subgridspec(1, 4, wspace=0.28)
    for i, (met, lab_) in enumerate([("PCC", "PCC"), ("SSIM", "SSIM"), ("RMSE", "RMSE"), ("JS", "JS divergence")]):
        ax = fig.add_subplot(gs2[0, i]); letter(ax, "HIJK"[i], x=-0.14)
        for x_, (nm, cc) in enumerate(METHODS):
            v = pc.get(nm)
            if v is None:
                ax.text(x_, 0.5, "pending", ha="center", va="center", fontsize=8, color="#9a9a9a", rotation=90, transform=ax.get_xaxis_transform()); continue
            vals = np.asarray(v[met]); bp = ax.boxplot([vals], positions=[x_], widths=0.6, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=cc, alpha=0.35, edgecolor=cc)
            for k_ in ("whiskers", "caps", "medians"):
                for l_ in bp[k_]:
                    l_.set(color=cc)
            ax.scatter(x_ + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals)), vals, s=12, color=cc, zorder=3)
        ax.set_xticks(range(len(METHODS)), [m for m, _ in METHODS], rotation=35, ha="right"); ax.set_title(lab_, fontsize=11)
        ax.grid(color=GRID, axis="y"); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]
    # legend of the slide colours: the clades, then the other labels shown
    lax = fig.add_subplot(gs[0, 7]); lax.axis("off")
    for k in [k for k in dkeys + ["notrace"] if k in seen]:
        lax.scatter([], [], c=col[k], s=40, marker="s", label=names[k])
    lax.legend(loc="center left", fontsize=8, frameon=False, ncol=1, title="label", title_fontsize=9, markerscale=1.2, handletextpad=0.3, labelspacing=0.35)
    from spotr.figlayout import finish; finish(fig, PANELS)
    FIGD, FN = (FIG, "fig_petracer_methods") if SECTION == "M3-4" else (CACHE, f"fig_petracer_{SECTION}")
    FIGD.mkdir(parents=True, exist_ok=True); fig.savefig(FIGD / f"{FN}.pdf", bbox_inches="tight", dpi=250); __import__('spotr.figlayout', fromlist=['compress_pdf']).compress_pdf(FIGD / f"{FN}.pdf"); fig.savefig(FIGD / f"{FN}.png", dpi=70, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({m: ({k: round(float(np.mean(v_)), 3) for k, v_ in v.items()} if v else None) for m, v in pc.items()}))


if __name__ == "__main__":
    main()
