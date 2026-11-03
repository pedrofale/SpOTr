"""The paper's colorectal liver metastasis figure (spotr.tex, fig:htan), with the results we have as placeholders (Pedro, 2026-11-02: "Let's
make the figures for PETracer and the colorectal cancer, and use whatever results we have currently as placeholders"); what SpOTr enables
beyond cell2location.

HT260C1 v4 (445_), over the H&E (Pedro, 2026-11-02: "show the CNV profiles of all the cells, with their lineage tree ... as one of the first
panels.  Then the subsequent panels show the assignments, and the final panel should show indeed the lineage tree in space, maybe similar to
a figure that CallicoST might have"; "keep the metrics in a supplementary figure").
(A) Every cancer nucleus's copy-number profile (39 RNA arm scores, log UMI regressed out, standardised -- the profiles whose distances are
    d_cn; drawn as the mean over the 21 neighbouring nuclei in tree order within the clone, as inferCNV-style heatmaps do), rows in the order
    of the tree SpOTr is given (Ward within each clone, clones joined by their mean distances), with the tree, the clone of each nucleus and
    B1's two sub-lineages (414_; nuclei of B1 without a sub-lineage call blank).
(B) SpOTr with the tree (450_'s quick check, the Methods' settings, alpha 0.2) and (C) cell2location (the package on v2's labels, 363_), each
    spot by its majority label across all labels.  The clones' copy-number signal in space (453_) is no longer drawn (Pedro, 2026-11-02).
(D) B1's two sub-lineages, which no method is given as labels: their copy-number signal on held-out genes inside B1's territory (414_), and
    (E) SpOTr's share of B1b among B1's mapped nuclei, against B1b's share of those nuclei.
(F) The tree in space, as CalicoST's phylogeography (Ma et al. 2024, Fig. 4a): each clade (A, B1, B2, B1a, B1b) and each ancestor (B, the
    common ancestor) at the centroid of its nuclei's expected positions under SpOTr's map, arrows from parent to child, over the H&E and
    SpOTr's clone map (B) faded; diamond area grows with the number of nuclei.
Titles short (Pedro, 2026-11-02: "figure titles and annotations should be short without too much info").
Supplementary (fig_htan_purity): purity against the copy-number reference, within B1 and between clones, for SpOTr with the tree, clone
    labels and expression only (alpha 0.2 / 0) and cell2location where its labels allow.
Style: decision-writing-style.

Usage: python 452_fig_htan_paper.py   (benchmark_env, CPU; writes SpOTr-overleaf/figures/fig_htan_beyond.{pdf,png} and fig_htan_purity.{pdf,png})
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; O = HERE / "spotr_out"; RAW = HERE.parent / "htan_screen" / "raw"; DATA = ROOT / "data" / "htan"
FIG = ROOT.parent / "SpOTr-overleaf" / "figures"; sys.path.insert(0, str(ROOT / "src")); SEC = "HT260C1-Th1K1Fc2U1Z1Bs1"; INK = "#111111"; GRID = "#e6e5e0"
UN = "tumour, unassigned"


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


PANELS = []                                                                                   # (axes, letter), laid out by spotr.figlayout.finish


def letter(ax, s, x=None, y=None):
    """registers the panel; spotr.figlayout.finish puts the letter on its column's edge and aligns the panels with the figure's edges"""
    PANELS.append((ax, s))


def cn_profiles(z, ti):
    """each traced nucleus's 39 RNA copy-number arm scores, log UMI regressed out and standardised: the profiles whose Euclidean distances are
    d_cn (362_ / 445_; reproduced to 1e-6), from which SpOTr's tree is built"""
    bt = z["barcodes"].astype(str)[ti]; SCR = HERE.parent / "htan_screen"
    S = pd.read_csv(SCR / "HT260C1_sn_own_arm_scores.tsv", sep="\t", index_col=0); asg = pd.read_csv(SCR / "HT260C1_sn_assign_k2.tsv", sep="\t").set_index("barcode")
    X = S.reindex(bt).to_numpy(float); u = np.log(asg.loc[bt, "umi"].to_numpy(float)); Dm = np.column_stack([np.ones(len(u)), u])
    X = X - Dm @ np.linalg.lstsq(Dm, X, rcond=None)[0]; return (X - X.mean(0)) / X.std(0), list(S.columns)


def spotr_linkage(d, c3):
    """the tree SpOTr is given (446_, via 419_.within_tree) as a scipy linkage: Ward within each clone on d_cn, clones joined by their mean
    distances (B1 with B2, then with A).  within_tree scales each clone's tree so its root sits at the shallowest between-clone distance; drawn
    at 95% of it, so that the clone's own stem shows"""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    n, K = len(c3), int(c3.max()) + 1
    Dk = np.array([[d[np.ix_(c3 == a, c3 == b)].mean() if a != b else 0.0 for b in range(K)] for a in range(K)]); off = Dk[~np.eye(K, dtype=bool)].min()
    rows, nid, root = [], n, {}
    for k in range(K):
        idx = np.where(c3 == k)[0]; Lk = linkage(squareform(d[np.ix_(idx, idx)], checks=False), "ward"); sc = 0.95 * off / Lk[:, 2].max(); g = list(idx) + [None] * (len(idx) - 1)
        for r, (a, b, h, c) in enumerate(Lk):
            rows.append([g[int(a)], g[int(b)], h * sc, c]); g[len(idx) + r] = nid; nid += 1
        root[k] = (nid - 1, len(idx))
    rows.append([root[0][0], root[1][0], Dk[0, 1], root[0][1] + root[1][1]]); nid += 1
    rows.append([nid - 1, root[2][0], (Dk[0, 2] + Dk[1, 2]) / 2, n]); return np.array(rows, float)


def main():
    import warnings; warnings.filterwarnings("ignore")
    from spotr.plot_spatial import slide
    from scipy.cluster.hierarchy import dendrogram
    from matplotlib.colors import to_rgb, ListedColormap
    from matplotlib.lines import Line2D
    m404, m446 = mod("m404", "404_htan_majority_labels.py"), mod("m446", "446_htan_v4_run.py"); col = dict(m404.COL); col["B1a"] = "#061f45"; col["B1b"] = "#3f6fc4"
    z2 = np.load(DATA / f"{SEC}_v2.npz", allow_pickle=True); xy = np.asarray(z2["xy"], float); sb = z2["spot_barcodes"].astype(str)
    img, um = m404.he("HT260C1", SEC, RAW / "HT260C1" / "B1-HT260C1-Th1K1Fc2U1.tif"); pix = xy / um
    f_ = 3; h_, w_ = (np.array(img["img"].shape[:2]) // f_) * f_                                   # the H&E averaged 3 x 3: enough for a panel this size, and a
    img["img"] = img["img"][:h_, :w_].reshape(h_ // f_, f_, w_ // f_, f_, 3).mean((1, 3))        # PDF that loads (it is embedded once per panel)
    img["scalefactors"]["tissue_lowres_scalef"] /= f_; sf = img["scalefactors"]["tissue_lowres_scalef"]
    z = np.load(DATA / f"{SEC}_v4.npz", allow_pickle=True); ti = np.where(z["traced"].astype(bool))[0]; bt = z["barcodes"].astype(str)[ti]
    lab3 = np.where(z["sn_clone"].astype(str)[ti] == "A", "A", z["b_sub"].astype(str)[ti]); ct = np.asarray(z["cell_type"]).astype(str)
    keep = sorted(e for e in set(ct.tolist()) if e != "tumour" and (ct == e).sum() > 1)
    sp = np.load(O / "subclone_spatial" / f"HT260C1_B1_{SEC}.npz", allow_pickle=True); hb = dict(zip(sp["nuclei"].astype(str), sp["h"])); terr = sp["terr"].astype(bool)
    isa = np.array([l == "B1" and b in hb and hb[b] != 1 for b, l in zip(bt, lab3)]); isb = np.array([l == "B1" and b in hb and hb[b] == 1 for b, l in zip(bt, lab3)])
    Q = json.load(open(O / "writeup_check" / "HT260C1_v4.json")) if (O / "writeup_check" / "HT260C1_v4.json").exists() else {}
    zq = np.load(O / "writeup_check" / "HT260C1_v4_tree_0.2_share.npz"); phi = zq["phi"].astype(float); Ab = zq["abund"].astype(float)   # capacities from cancer shares (frozen 2026-11-02)
    share = phi / np.maximum(phi.sum(0, keepdims=True), 1e-300); tum = Ab[:, len(keep)]
    comp = {e: Ab[:, i] for i, e in enumerate(keep)}; comp.update({k: share[lab3 == k].sum(0) * tum for k in ("A", "B1", "B2")})
    P = phi / np.maximum(phi.sum(1, keepdims=True), 1e-300)
    fig = plt.figure(figsize=(28, 14)); gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1, 1, 1], hspace=0.2, wspace=0.26)
    def cat(ax, colours, title):
        keep_ = np.array([c is not None for c in colours]); slide(ax, pix[keep_], [c for c in colours if c is not None], title, image=img, size=1.6, img_alpha=0.5, fontsize=12)
    seen = set()
    # (A) the copy-number profiles of every cancer nucleus, ordered by the tree SpOTr is given
    Zs, arms = cn_profiles(z, ti); c3 = np.array([["B1", "B2", "A"].index(l) for l in lab3]); Lk = spotr_linkage(np.asarray(z["d_cn"], float), c3); n = len(c3)
    colA = []; sa = gs[:, 0].subgridspec(1, 4, width_ratios=[0.8, 0.09, 0.09, 2.6], wspace=0.04)
    axd = fig.add_subplot(sa[0, 0]); letter(axd, "A")
    with plt.rc_context({"lines.linewidth": 0.4}):
        R_ = dendrogram(Lk, orientation="left", no_labels=True, ax=axd, color_threshold=0, above_threshold_color="#3a3a3a")
    for c_ in axd.collections:
        c_.set_rasterized(True)
    lv = np.array(R_["leaves"]); axd.set_ylim(0, 10 * n); axd.axis("off")
    for k_, (lab_, bar) in enumerate([("clone", np.array([to_rgb(col[l]) for l in lab3[lv]])),
                                      ("B1 sub-lineage", np.array([to_rgb(col["B1a"]) if isa[i] else to_rgb(col["B1b"]) if isb[i] else (1, 1, 1) for i in lv]))]):
        axb = fig.add_subplot(sa[0, 1 + k_]); colA.append(axb); axb.imshow(bar[:, None, :], aspect="auto", origin="lower", extent=(0, 1, 0, 10 * n), interpolation="nearest")
        axb.set_xticks([0.5], [lab_], rotation=90, fontsize=10); axb.set_yticks([]); [axb.spines[s_].set_visible(False) for s_ in axb.spines]
    axh = fig.add_subplot(sa[0, 3]); colA += [axd, axh]; lim = 1.5; W = 21                     # each profile smoothed over its W neighbours in tree order, within its clone
    from scipy.ndimage import uniform_filter1d
    Zo = Zs[lv].copy(); co = c3[lv]
    for k_ in np.unique(co):
        r_ = np.where(co == k_)[0]; Zo[r_] = uniform_filter1d(Zo[r_], size=W, axis=0, mode="nearest")
    im = axh.imshow(np.clip(Zo, -lim, lim), aspect="auto", origin="lower", extent=(0, len(arms), 0, 10 * n), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest", rasterized=True)
    chrom = [a[3:-1] for a in arms]; bounds = [i for i in range(1, len(arms)) if chrom[i] != chrom[i - 1]]
    for b_ in bounds:
        axh.axvline(b_, color="white", lw=1.2)
    mids = [(s0 + s1) / 2 for s0, s1 in zip([0] + bounds, bounds + [len(arms)])]; names_ = [chrom[s0] for s0 in [0] + bounds]
    axh.set_xticks(mids, names_, fontsize=8); axh.xaxis.tick_top(); axh.set_yticks([]); [axh.spines[s_].set_visible(False) for s_ in axh.spines]
    axh.set_title("snRNA-seq CNV", fontsize=12, pad=16)
    cax = axh.inset_axes([0.45, -0.075, 0.5, 0.013]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("CNV score", fontsize=9); cb.ax.tick_params(labelsize=8)
    hd = [Line2D([], [], ls="", marker="s", ms=10, color=col[l], label=l) for l in ("A", "B1", "B2", "B1a", "B1b")]
    axd.legend(handles=hd, loc="upper left", bbox_to_anchor=(0.0, -0.11), ncol=3, frameon=False, fontsize=10, handletextpad=0.3, columnspacing=0.8)
    # (B-D) the assignments
    L = sorted(comp); M_ = np.stack([comp[l] for l in L], 1); labC = np.array(L)[M_.argmax(1)]; seen |= set(labC.tolist())
    ax = fig.add_subplot(gs[0, 1]); axB = ax; letter(ax, "B"); cat(ax, [col.get(l, "#9a9994") for l in labC], "SpOTr")
    fc4 = O / "c2l_package" / "htan_HT260C1_v4_out.npz"                                    # cell2location on the v4 calls (468_, 355_ fit) once it has run
    o = np.load(fc4 if fc4.exists() else O / "htan_v2" / "c2l.npz", allow_pickle=True); ty = [UN if str(t) == "untraced" else str(t) for t in o["types"]]; lab = np.array(ty)[np.asarray(o["w"]).argmax(1)]; seen |= set(lab.tolist())
    axD = fig.add_subplot(gs[0, 2]); letter(axD, "C"); cat(axD, [col.get(l, "#9a9994") for l in lab], "cell2location")
    hd = [Line2D([], [], ls="", marker="s", ms=11, color=col[t], label=t) for t in col if t in seen]
    axD.legend(handles=hd, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=False, title="majority label", title_fontsize=12)
    # (E, F) B1's sub-lineages
    v = np.where(terr, sp["p"], np.nan); l_ = float(np.nanquantile(np.abs(v), 0.98))          # D a copy-number score, E a share: each on its own scale,
    cbars = []; ax = fig.add_subplot(gs[1, 1]); axD2 = ax; letter(ax, "D"); slide(ax, pix, v, "Visium CNV: B1 sub-lineages", image=img, size=1.6, img_alpha=0.5, cmap="coolwarm", norm=TwoSlopeNorm(0.0, -l_, l_), fontsize=12, colorbar=False)
    cbars.append((ax, "CNV score"))
    fb = isb.sum() / (isa.sum() + isb.sum()); a_, b_ = P[isa].sum(0), P[isb].sum(0); b1 = a_ + b_; sh = np.where(terr & (b1 >= 0.25), b_ / np.maximum(b1, 1e-300) - fb, np.nan)
    ax = fig.add_subplot(gs[1, 2]); letter(ax, "E"); slide(ax, pix, sh, "SpOTr: B1 sub-lineages", image=img, size=1.6, img_alpha=0.5, cmap="coolwarm", norm=TwoSlopeNorm(0.0, -0.5, 0.5), fontsize=12, colorbar=False)
    cbars.append((ax, "B1b share"))
    # (G) the tree in space, as CalicoST's phylogeography (Ma et al. 2024, Fig. 4a): every clade at the centroid of its nuclei's locations under
    # SpOTr's map (each nucleus at its expected position), ancestors at the centroid of all their descendants, arrows from parent to child
    pos_n = P @ pix; clades = {"ancestor": np.ones(n, bool), "A": lab3 == "A", "B": np.isin(lab3, ["B1", "B2"]), "B1": lab3 == "B1", "B2": lab3 == "B2", "B1a": isa, "B1b": isb}
    cen = {k: pos_n[m_].mean(0) * sf for k, m_ in clades.items()}; ccol = {"ancestor": "#5f5f5f", "A": col["A"], "B": col["B"], "B1": col["B1"], "B2": col["B2"], "B1a": col["B1a"], "B1b": col["B1b"]}
    fade = lambda c: tuple(0.45 * np.array(to_rgb(c)) + 0.55)
    ax = fig.add_subplot(gs[1, 3]); letter(ax, "F"); cat(ax, [fade(col.get(l, "#9a9994")) if l in ("A", "B1", "B2") else None for l in labC], "CNV tree in SpOTr map")
    for par, ch in [("ancestor", "A"), ("ancestor", "B"), ("B", "B1"), ("B", "B2"), ("B1", "B1a"), ("B1", "B1b")]:
        ax.annotate("", xy=cen[ch], xytext=cen[par], arrowprops=dict(arrowstyle="-|>", color=ccol[ch], lw=2.6, mutation_scale=22, shrinkA=9, shrinkB=9), zorder=4)
    for k, c_ in cen.items():
        ax.scatter(*c_, marker="D", s=40 + 260 * np.sqrt(clades[k].sum() / n), color=ccol[k], edgecolor="white", linewidth=1.5, zorder=5)
        dx, dy, ha = {"B1b": (-10, 8, "right"), "B1": (2, 12, "left"), "B": (10, 2, "left"), "B2": (10, -8, "left"), "B1a": (8, 6, "left"), "ancestor": (10, 4, "left"), "A": (10, 2, "left")}[k]
        ax.annotate(k, c_, xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=11, fontweight="bold", color=INK, zorder=6)
    from spotr.figlayout import finish
    for ax_, _ in PANELS[1:]:                                                                  # the slides pinned to the top of their cells, as finish does,
        ax_.set_anchor("NE" if ax_.get_subplotspec().colspan.start == 3 else "NW")             # so that A can span exactly from the first row's top to the
    fig.canvas.draw(); pt, pb = axB.get_position(), axD2.get_position()                        # second row's bottom
    for a_ in colA:
        p_ = a_.get_position(); a_.set_position([p_.x0, pb.y0, p_.width, pt.y1 - pb.y0])
    finish(fig, PANELS)
    for ax_, lab_ in cbars:                                                                    # each colorbar the height of its slide, in the gap beside it
        p_ = ax_.get_position(); m_ = [c_ for c_ in ax_.collections if c_.get_array() is not None][-1]
        cb_ = fig.colorbar(m_, cax=fig.add_axes([p_.x1 + 0.005, p_.y0, 0.008, p_.height])); cb_.ax.tick_params(labelsize=9); cb_.set_label(lab_, fontsize=10, labelpad=2)
    FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "fig_htan_beyond.pdf", bbox_inches="tight", dpi=200); __import__('spotr.figlayout', fromlist=['compress_pdf']).compress_pdf(FIG / "fig_htan_beyond.pdf"); fig.savefig(FIG / "fig_htan_beyond.png", dpi=60, bbox_inches="tight"); plt.close(fig)
    print({k: np.round(c_ / sf, 0).tolist() for k, c_ in cen.items()})
    # Supplementary: purity against the copy-number reference (Pedro, 2026-11-02: "keep the metrics in a supplementary figure")
    m415 = mod("m415", "415_deeper_trees.py")
    def c2l_purity():
        """cell2location's purity on the comparisons its labels allow (B1 vs B2, B vs A), as 446_'s scorer, from its per-spot clone abundances"""
        C = m415.section("HT260C1")["contrasts"]; g = {"A": o["w"][:, ty.index("A")], "B1": o["w"][:, ty.index("B1")], "B2": o["w"][:, ty.index("B2")]}
        con = {"B1 vs B2": (["B1"], ["B2"], C["B1 vs B2"][2]), "B vs A": (["B1", "B2"], ["A"], C["B vs A"][2])}
        return {k: v["purity"] for k, v in m415.gscore(g, dict(contrasts=con)).items()}
    fs, ax = plt.subplots(figsize=(9, 5))
    arms_ = [("tree 0.2", "SpOTr, tree", "#F47B20"), ("labels 0.2", "SpOTr, clone labels", "#F8A869"), ("tree 0.0", "SpOTr, expression only", "#FBD3B0"), ("c2l", "cell2location", "#1f77b4")]
    S_ = json.load(open(O / "writeup_check" / "capacity_share_arms.json"))                    # the share capacity's three arms (456_)
    Q = {k: dict(scores=S_[f"share | {k}"]["scores"]) for k in ("tree 0.2", "labels 0.2", "tree 0.0")}; Q["c2l"] = dict(scores=c2l_purity())
    mets = [("B1b vs B1a", "within B1\n(B1b vs B1a)"), ("B1 vs B2", "between B1 and B2"), ("B vs A", "between B and A")]; w = 0.18
    for i, (key, nm, cc) in enumerate(arms_):
        for j, (m_, _) in enumerate(mets):
            v_ = Q.get(key, {}).get("scores", {}).get(m_)
            if v_ is None:
                ax.text(j + (i - 1.5) * w, 0.52, "n/a", rotation=90, ha="center", va="bottom", fontsize=8, color="#9a9a9a")
            else:
                ax.bar(j + (i - 1.5) * w, v_, width=w * 0.9, color=cc, label=nm if j == 0 else None)
    ax.set_xticks(range(len(mets)), [t for _, t in mets]); ax.set_ylim(0.5, 1.0); ax.set_ylabel("purity against the copy-number signal"); ax.grid(color=GRID, axis="y"); ax.set_axisbelow(True)
    [ax.spines[s_].set_visible(False) for s_ in ("top", "right")]; ax.legend(frameon=False, fontsize=9, loc="upper right")
    fs.savefig(FIG / "fig_htan_purity.pdf", bbox_inches="tight"); fs.savefig(FIG / "fig_htan_purity.png", dpi=80, bbox_inches="tight"); plt.close(fs)


if __name__ == "__main__":
    main()
