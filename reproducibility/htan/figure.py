"""Fig. 4 (fig_htan_beyond) and Fig. S3 (fig_htan_purity) of the paper: HT260C1 over its H&E.

(A) Every cancer nucleus's copy-number profile (39 arm scores, log UMI regressed out, standardised; each drawn as the mean over the 21
    neighbouring nuclei in tree order within its clone), rows in the order of the tree SpOTr is given, with the clone of each nucleus and B1's
    two sub-lineages (blank for the B1 nuclei without a sub-lineage call).
(B) SpOTr (alpha 0.2) and (C) cell2location: each spot by its majority label across all labels.
(D) B1's sub-lineages, which no method is given: their copy-number signal on held-out genes inside B1's territory, and (E) SpOTr's share of
    B1b among B1's mapped nuclei, against B1b's share of those nuclei (spots with at least a quarter of a mapped B1 nucleus).
(F) The tree in space: each clade (A, B1, B2, B1a, B1b) and each ancestor (B, the common ancestor) at the centroid of its nuclei's expected
    positions under SpOTr's map, arrows from parent to child, over SpOTr's clone map (B) faded; marker area grows with the number of nuclei.
Fig. S3: balanced accuracy against the copy-number references -- within B1 (B1b vs B1a), between B1 and B2, between B and A -- for SpOTr,
    SpOTr with the expression alone (alpha 0) and cell2location where its labels allow.  Purity: the spots beyond 0.5 sd of a reference on
    either side; the mean of the two sides' shares of the expected group.

Inputs: $SPOTR_DATA/htan/{sc,st}.h5ad (prepare.py), $SPOTR_RESULTS/htan/{spotr_alpha0.2,spotr_alpha0.0,c2l}.npz (fit.py).
Usage: python figure.py   (benchmark_env, CPU; writes fig_htan_beyond.{pdf,png} and fig_htan_purity.{pdf,png} to $SPOTR_FIGURES)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import paths, figlayout  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent)); import copynumber as cn  # noqa: E402

INK, GRID = "#111111", "#e6e5e0"
COL = {"A": "#e86a33", "B": "#0b3a7e", "B1": "#0b3a7e", "B2": "#7fb0ea", "B1a": "#061f45", "B1b": "#3f6fc4", "Fibroblasts": "#1baf7a",
       "Macrophages": "#eda100", "Endothelial": "#e87ba4", "T-cells": "#4cc1d0", "B-cells": "#6250d6", "Plasma": "#a35f2d", "Hepatocytes": "#b07fd0",
       "Cholangiocytes": "#7fd1b0", "DC": "#d4b106", "Pericytes": "#8c564b"}
OTHER = "#9a9994"


def he_image(st_, f=3):
    """the H&E as stored by prepare.py (a tenth of full resolution), averaged f x f"""
    lib = next(iter(st_.uns["spatial"])); s = st_.uns["spatial"][lib]; im = np.asarray(s["images"]["lowres"], float) / 255.0
    h, w = (np.array(im.shape[:2]) // f) * f; im = im[:h, :w].reshape(h // f, f, w // f, f, 3).mean((1, 3))
    sf = dict(s["scalefactors"]); sf["tissue_lowres_scalef"] = float(sf["tissue_lowres_scalef"]) / f; return dict(img=im, scalefactors=sf)


def groups(sc_, cells):
    """each mapped nucleus's group for the scores: its clone, B1's nuclei by sub-lineage (B1new without one)"""
    o = sc_.obs.loc[cells]; cl = o["clone"].astype(str).to_numpy(); sl = o["sublineage"].astype(str).to_numpy()
    return np.where(cl == "B1", np.where(np.isin(sl, ["B1a", "B1b"]), sl, "B1new"), cl)


def contrasts(st_):
    o = st_.obs; b1 = np.where(o["b1_territory"].to_numpy(bool), o["b1_signal"].to_numpy(float), np.nan)
    return {"B1b vs B1a": (["B1b"], ["B1a"], b1), "B vs A": (["B1a", "B1b", "B1new", "B2"], ["A"], o["ref_ba"].to_numpy(float)),
            "B1 vs B2": (["B1a", "B1b", "B1new"], ["B2"], o["ref_wb"].to_numpy(float))}


def purity(G, con):
    """balanced accuracy per contrast, from each group's mass per spot"""
    out = {}
    for nm, (g1, g0, ref) in con.items():
        if not all(g in G for g in g1 + g0):
            continue
        a = sum(G[g] for g in g1); b = sum(G[g] for g in g0); ok = np.isfinite(ref); z = np.full(len(ref), np.nan); z[ok] = (ref[ok] - ref[ok].mean()) / ref[ok].std()
        with np.errstate(invalid="ignore"):
            t1, t0 = ok & (z > 0.5), ok & (z < -0.5)
        out[nm] = float((a[t1].sum() / max((a + b)[t1].sum(), 1e-300) + b[t0].sum() / max((a + b)[t0].sum(), 1e-300)) / 2)
    return out


def map_purity(phi, grp, con):
    P = phi / np.maximum(phi.sum(1, keepdims=True), 1e-300); return purity({g: P[grp == g].sum(0) for g in set(grp)}, con)


def main():
    import warnings; warnings.filterwarnings("ignore")
    import anndata as ad
    from scipy.cluster.hierarchy import dendrogram
    from scipy.ndimage import uniform_filter1d
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from spotr.plotting import spots
    sc_, st_ = ad.read_h5ad(paths.DATA / "htan" / "sc.h5ad"), ad.read_h5ad(paths.DATA / "htan" / "st.h5ad")
    R = {a: np.load(paths.RESULTS / "htan" / f"spotr_alpha{a}.npz", allow_pickle=True) for a in (0.2, 0.0)}; o = np.load(paths.RESULTS / "htan" / "c2l.npz", allow_pickle=True)
    img = he_image(st_); sf = img["scalefactors"]["tissue_lowres_scalef"]; pix = np.asarray(st_.obsm["spatial"], float) / float(st_.uns["um_per_pixel"])
    terr = st_.obs["b1_territory"].to_numpy(bool); cells = R[0.2]["cells"].astype(str); lab3 = sc_.obs.loc[cells, "clone"].astype(str).to_numpy()
    sl = sc_.obs.loc[cells, "sublineage"].astype(str).to_numpy(); isa, isb = sl == "B1a", sl == "B1b"; n = len(cells)
    phi = R[0.2]["phi"].astype(float); P = phi / np.maximum(phi.sum(1, keepdims=True), 1e-300)
    PANELS = []
    fig = plt.figure(figsize=(28, 14)); gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1, 1, 1], hspace=0.2, wspace=0.26)
    def cat(ax, colours, title):
        k = np.array([c is not None for c in colours]); spots(ax, pix[k], [c for c in colours if c is not None], title, image=img, size=1.6, img_alpha=0.5, fontsize=12)
    seen = set()
    # (A) the cancer nuclei's copy-number profiles, in the order of the tree SpOTr is given
    Zs = sc_.obsm["cn_profile"][sc_.obs_names.get_indexer(cells)]; arms = [str(a) for a in sc_.uns["cn_arms"]]
    c3 = np.array([["B1", "B2", "A"].index(l) for l in lab3]); Lk = cn.clone_linkage(cn.profile_distances(Zs), c3, (0, 1, 2))
    colA = []; sa = gs[:, 0].subgridspec(1, 4, width_ratios=[0.8, 0.09, 0.09, 2.6], wspace=0.04)
    axd = fig.add_subplot(sa[0, 0]); PANELS.append((axd, "A"))
    with plt.rc_context({"lines.linewidth": 0.4}):
        R_ = dendrogram(Lk, orientation="left", no_labels=True, ax=axd, color_threshold=0, above_threshold_color="#3a3a3a")
    for c_ in axd.collections:
        c_.set_rasterized(True)
    lv = np.array(R_["leaves"]); axd.set_ylim(0, 10 * n); axd.axis("off")
    for k_, (lab_, bar) in enumerate([("clone", np.array([to_rgb(COL[l]) for l in lab3[lv]])),
                                      ("B1 sub-lineage", np.array([to_rgb(COL["B1a"]) if isa[i] else to_rgb(COL["B1b"]) if isb[i] else (1, 1, 1) for i in lv]))]):
        axb = fig.add_subplot(sa[0, 1 + k_]); colA.append(axb); axb.imshow(bar[:, None, :], aspect="auto", origin="lower", extent=(0, 1, 0, 10 * n), interpolation="nearest")
        axb.set_xticks([0.5], [lab_], rotation=90, fontsize=10); axb.set_yticks([]); [axb.spines[s_].set_visible(False) for s_ in axb.spines]
    axh = fig.add_subplot(sa[0, 3]); colA += [axd, axh]; lim, W = 1.5, 21
    Zo = Zs[lv].copy(); co = c3[lv]
    for k_ in np.unique(co):
        r_ = np.where(co == k_)[0]; Zo[r_] = uniform_filter1d(Zo[r_], size=W, axis=0, mode="nearest")
    im = axh.imshow(np.clip(Zo, -lim, lim), aspect="auto", origin="lower", extent=(0, len(arms), 0, 10 * n), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest", rasterized=True)
    chrom = [a[3:-1] for a in arms]; bounds = [i for i in range(1, len(arms)) if chrom[i] != chrom[i - 1]]
    for b_ in bounds:
        axh.axvline(b_, color="white", lw=1.2)
    mids = [(s0 + s1) / 2 for s0, s1 in zip([0] + bounds, bounds + [len(arms)])]
    axh.set_xticks(mids, [chrom[s0] for s0 in [0] + bounds], fontsize=8); axh.xaxis.tick_top(); axh.set_yticks([]); [axh.spines[s_].set_visible(False) for s_ in axh.spines]
    axh.set_title("snRNA-seq CNV", fontsize=12, pad=16)
    cax = axh.inset_axes([0.45, -0.075, 0.5, 0.013]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("CNV score", fontsize=9); cb.ax.tick_params(labelsize=8)
    axd.legend(handles=[Line2D([], [], ls="", marker="s", ms=10, color=COL[l], label=l) for l in ("A", "B1", "B2", "B1a", "B1b")], loc="upper left",
               bbox_to_anchor=(0.0, -0.11), ncol=3, frameon=False, fontsize=10, handletextpad=0.3, columnspacing=0.8)
    # (B, C) the assignments
    L = [str(l) for l in R[0.2]["labels"]]; labC = np.array(L)[R[0.2]["proportions"].argmax(1)]; seen |= set(labC.tolist())
    axB = fig.add_subplot(gs[0, 1]); PANELS.append((axB, "B")); cat(axB, [COL.get(l, OTHER) for l in labC], "SpOTr")
    ty = [str(t) for t in o["types"]]; lab = np.array(ty)[np.asarray(o["w"]).argmax(1)]; seen |= set(lab.tolist())
    axC = fig.add_subplot(gs[0, 2]); PANELS.append((axC, "C")); cat(axC, [COL.get(l, OTHER) for l in lab], "cell2location")
    axC.legend(handles=[Line2D([], [], ls="", marker="s", ms=11, color=COL[t], label=t) for t in COL if t in seen], loc="center left", bbox_to_anchor=(1.02, 0.5),
               fontsize=12, frameon=False, title="majority label", title_fontsize=12)
    # (D, E) B1's sub-lineages
    v = np.where(terr, st_.obs["b1_signal"].to_numpy(float), np.nan); l_ = float(np.nanquantile(np.abs(v), 0.98)); cbars = []
    axD = fig.add_subplot(gs[1, 1]); PANELS.append((axD, "D"))
    spots(axD, pix, v, "Visium CNV: B1 sub-lineages", image=img, size=1.6, img_alpha=0.5, cmap="coolwarm", norm=TwoSlopeNorm(0.0, -l_, l_), fontsize=12, colorbar=False)
    cbars.append((axD, "CNV score"))
    fb = isb.sum() / (isa.sum() + isb.sum()); a_, b_ = P[isa].sum(0), P[isb].sum(0); b1 = a_ + b_
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(terr & (b1 >= 0.25), b_ / np.maximum(b1, 1e-300) - fb, np.nan)
    axE = fig.add_subplot(gs[1, 2]); PANELS.append((axE, "E"))
    spots(axE, pix, sh, "SpOTr: B1 sub-lineages", image=img, size=1.6, img_alpha=0.5, cmap="coolwarm", norm=TwoSlopeNorm(0.0, -0.5, 0.5), fontsize=12, colorbar=False)
    cbars.append((axE, "B1b share"))
    # (F) the tree in space
    pos_n = P @ pix; clades = {"ancestor": np.ones(n, bool), "A": lab3 == "A", "B": np.isin(lab3, ["B1", "B2"]), "B1": lab3 == "B1", "B2": lab3 == "B2", "B1a": isa, "B1b": isb}
    cen = {k: pos_n[m_].mean(0) * sf for k, m_ in clades.items()}; ccol = dict(ancestor="#5f5f5f", **{k: COL[k] for k in ("A", "B", "B1", "B2", "B1a", "B1b")})
    fade = lambda c: tuple(0.45 * np.array(to_rgb(c)) + 0.55)
    axF = fig.add_subplot(gs[1, 3]); PANELS.append((axF, "F")); cat(axF, [fade(COL.get(l, OTHER)) if l in ("A", "B1", "B2") else None for l in labC], "CNV tree in SpOTr map")
    for par, ch in [("ancestor", "A"), ("ancestor", "B"), ("B", "B1"), ("B", "B2"), ("B1", "B1a"), ("B1", "B1b")]:
        axF.annotate("", xy=cen[ch], xytext=cen[par], arrowprops=dict(arrowstyle="-|>", color=ccol[ch], lw=2.6, mutation_scale=22, shrinkA=9, shrinkB=9), zorder=4)
    for k, c_ in cen.items():
        axF.scatter(*c_, marker="D", s=40 + 260 * np.sqrt(clades[k].sum() / n), color=ccol[k], edgecolor="white", linewidth=1.5, zorder=5)
        dx, dy, ha = {"B1b": (-10, 8, "right"), "B1": (2, 12, "left"), "B": (10, 2, "left"), "B2": (10, -8, "left"), "B1a": (8, 6, "left"), "ancestor": (10, 4, "left"), "A": (10, 2, "left")}[k]
        axF.annotate(k, c_, xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=11, fontweight="bold", color=INK, zorder=6)
    for ax_, _ in PANELS[1:]:                                                                   # the slides pinned to the top of their cells, so that the
        ax_.set_anchor("NE" if ax_.get_subplotspec().colspan.start == 3 else "NW")              # profiles (A) span exactly from the first row's top to the
    fig.canvas.draw(); pt, pb = axB.get_position(), axD.get_position()                         # second row's bottom
    for a_ in colA:
        p_ = a_.get_position(); a_.set_position([p_.x0, pb.y0, p_.width, pt.y1 - pb.y0])
    figlayout.finish(fig, PANELS)
    for ax_, lab_ in cbars:                                                                     # each colorbar the height of its slide, in the gap beside it
        p_ = ax_.get_position(); m_ = [c_ for c_ in ax_.collections if c_.get_array() is not None][-1]
        cb_ = fig.colorbar(m_, cax=fig.add_axes([p_.x1 + 0.005, p_.y0, 0.008, p_.height])); cb_.ax.tick_params(labelsize=9); cb_.set_label(lab_, fontsize=10, labelpad=2)
    f = paths.figure("fig_htan_beyond.pdf"); fig.savefig(f, bbox_inches="tight", dpi=200); figlayout.compress_pdf(f)
    fig.savefig(paths.figure("fig_htan_beyond.png"), dpi=60, bbox_inches="tight"); plt.close(fig)
    # Fig. S3: balanced accuracy against the copy-number references
    con = contrasts(st_); Q = {f"alpha {a}": map_purity(R[a]["phi"].astype(float), groups(sc_, R[a]["cells"].astype(str)), con) for a in (0.2, 0.0)}
    w_ = np.asarray(o["w"], float); Q["c2l"] = purity({k: w_[:, ty.index(k)] for k in ("A", "B1", "B2")},
                                                    {"B1 vs B2": (["B1"], ["B2"], con["B1 vs B2"][2]), "B vs A": (["B1", "B2"], ["A"], con["B vs A"][2])})
    print({k: {m: round(v, 3) for m, v in q.items()} for k, q in Q.items()})
    fs, ax = plt.subplots(figsize=(9, 5)); w = 0.18
    arms_ = [("alpha 0.2", "SpOTr", "#F47B20"), ("alpha 0.0", "SpOTr, expression only", "#FBD3B0"), ("c2l", "cell2location", "#1f77b4")]
    mets = [("B1b vs B1a", "within B1\n(B1b vs B1a)"), ("B1 vs B2", "between B1 and B2"), ("B vs A", "between B and A")]
    for i, (key, nm, cc) in enumerate(arms_):
        for j, (m_, _) in enumerate(mets):
            x = j + (i - (len(arms_) - 1) / 2) * w; v_ = Q[key].get(m_)
            if v_ is None:
                ax.text(x, 0.52, "n/a", rotation=90, ha="center", va="bottom", fontsize=8, color="#9a9a9a")
            else:
                ax.bar(x, v_, width=w * 0.9, color=cc)
    ax.set_xticks(range(len(mets)), [t for _, t in mets]); ax.set_ylim(0.5, 1.0); ax.set_ylabel("balanced accuracy"); ax.grid(color=GRID, axis="y")
    ax.set_axisbelow(True); [ax.spines[s_].set_visible(False) for s_ in ("top", "right")]
    ax.legend(handles=[Patch(color=cc, label=nm) for _, nm, cc in arms_], frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(arms_))
    fs.savefig(paths.figure("fig_htan_purity.pdf"), bbox_inches="tight"); fs.savefig(paths.figure("fig_htan_purity.png"), dpi=80, bbox_inches="tight"); plt.close(fs)
    return Q


if __name__ == "__main__":
    main()
