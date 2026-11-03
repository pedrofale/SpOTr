"""Fig. 3: SpOTr and the compared methods on PEtracer section M3-4 (other sections: a figure of their own name).

(A)    the cells of the measured half at their in situ positions: traced malignant cells by clade, malignant cells without a lineage readout
       in grey, the non-tumor cells pooled.
(B-G)  the synthetic Visium spots' truth and each method's estimate, every spot in two levels: non-tumor where its non-tumor cells (every
       type pooled) outnumber its tumor cells (traced or not), otherwise the majority clade among its cells with a lineage readout (grey when
       it has none).  Spots drawn as circles of 1.6 x the spot diameter.
(H-K)  per-clone agreement with the truth (common.metrics.per_clone over the clones with at least 20 true cells): PCC, SSIM (higher is
       better), RMSE and Jensen-Shannon divergence (lower is better); one point per clone.

Reads $SPOTR_DATA/petracer/<section>/ (prepare.py) and $SPOTR_RESULTS/petracer/<section>/<method>.npz (fit.py); writes
$SPOTR_FIGURES/fig_petracer_methods.{pdf,png} and prints each method's mean scores.

Usage: python figure.py [section=M3-4]   (environment benchmark_env)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, metrics, figlayout  # noqa: E402

METHODS = [("SpOTr", "spotr", "#F47B20"), ("cell2location", "cell2location", "#1f77b4"), ("DestVI", "destvi", "#9467bd"),
           ("Tangram", "tangram", "#2ca02c"), ("scDOT", "scdot", "#d62728")]
CANCER, PITCH, GRID, MUTED_TEXT = "Malignant", 55.0, "#e6e5e0", "#7a7a7a"
MUTED = ["#d8d2c4", "#a89f8e", "#c9c0ad", "#8f8676", "#e2dccf", "#b5ac99", "#9c9381", "#cfc7b6", "#7f7768", "#ddd6c7", "#aea593", "#c2b9a6",
         "#958c7b", "#d5cdbd", "#a39a88", "#bdb4a1", "#877f70"]


def palette(clones, others):
    """clade colours (tab20 without its greys), muted shades for the non-tumor types, greys for the tumor without a clone"""
    tab = plt.get_cmap("tab20"); pal = [to_hex(tab(i)) for i in list(range(0, 20, 2)) + list(range(1, 20, 2)) if i not in (14, 15)]
    col = {c: pal[i % len(pal)] for i, c in enumerate(clones) if c != "pooled"}; col["pooled"] = "#e0b0d0"
    col.update({e: MUTED[i % len(MUTED)] for i, e in enumerate(others)})
    col.update({CANCER: "#5f5f5f", "nontumor": "#cfc6b4", "notrace": "#a3a3a3"})
    names = {c: c for c in clones} | {"pooled": "small clades, pooled", "nontumor": "non-tumor cells", "notrace": "tumor, no readout"}
    return col, names


def two_level(comp, keys, clones, others, col):
    """each spot's colour: non-tumor, else its majority clade among the cells with a lineage readout, else grey"""
    comp = np.asarray(comp, float); ci = [keys.index(c) for c in clones]; mi = [keys.index(e) for e in others]; ui = keys.index(CANCER)
    C = comp[:, ci]; tum = C.sum(1) + comp[:, ui]; non = comp[:, mi].sum(1)
    lab = np.where(non >= tum, "nontumor", np.where(C.sum(1) > 1e-9, np.array(clones)[C.argmax(1)], "notrace"))
    return [col[l] if t > 1e-6 else "#ffffff" for l, t in zip(lab, tum + non)], set(lab.tolist())


def main(section="M3-4"):
    import warnings; warnings.filterwarnings("ignore")
    import anndata as ad
    from spotr.plotting import spots, cells
    D = paths.DATA / "petracer" / section; R = paths.RESULTS / "petracer" / section
    st = ad.read_h5ad(D / "st.h5ad"); ins = np.load(D / "insitu.npz", allow_pickle=True); xy = np.asarray(st.obsm["spatial"], float)
    Ncl, Nme = st.obsm["clone_counts"], st.obsm["type_counts"]; clones, others = list(Ncl.columns), list(Nme.columns); keys = clones + others + [CANCER]
    col, names = palette(clones, others)
    truth = np.concatenate([Ncl.to_numpy(), Nme.to_numpy(), np.maximum(st.obs["n_tumor"].to_numpy() - Ncl.to_numpy().sum(1), 0)[:, None]], 1)
    def comp_of(method):
        f = R / f"{method}.npz"
        if not f.exists():
            return None
        z = np.load(f, allow_pickle=True); lab = [str(l) for l in z["labels"]]; W = np.asarray(z["comp"], float)
        return np.stack([W[:, lab.index(k)] if k in lab else np.zeros(len(W)) for k in keys], 1)
    maps = [(title, comp_of(m)) for title, m, _ in METHODS]
    panels = []
    fig = plt.figure(figsize=(28, 9.5)); gs = fig.add_gridspec(2, 8, height_ratios=[1, 0.95], width_ratios=[1] * 7 + [0.75], hspace=0.32, wspace=0.08)
    # (A) the measured half's cells in situ: non-tumor first, then tumor without a readout, traced cells on top
    lab = ins["label"].astype(str); is_clone = np.isin(lab, clones); is_non = ~is_clone & (lab != CANCER)
    cc = np.array([col[l] if c else (col["nontumor"] if n else col["notrace"]) for l, c, n in zip(lab, is_clone, is_non)], dtype=object)
    order = np.argsort(np.where(is_non, 0, np.where(is_clone, 2, 1)), kind="stable")
    pad = 1.6 * PITCH / 2; ext = (xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax = fig.add_subplot(gs[0, 0]); panels.append((ax, "A")); cells(ax, ins["xy"].astype(float)[order], list(cc[order]), "In situ cells", size=1.2, fontsize=11, extent=ext)
    seen = {"nontumor", "notrace"} | set(lab[is_clone].tolist())
    # (B-G) truth and each method's map
    for j, (title, comp) in enumerate([("Synthetic Visium", truth)] + maps):
        ax = fig.add_subplot(gs[0, j + 1]); panels.append((ax, "BCDEFG"[j]))
        if comp is None:
            ax.axis("off"); ax.text(0.5, 0.5, f"{title}\n(not run)", ha="center", va="center", fontsize=11, color=MUTED_TEXT, transform=ax.transAxes); continue
        cs, s_ = two_level(comp, keys, clones, others, col); seen |= s_
        spots(ax, xy, cs, title, fontsize=11, pitch=1.6 * PITCH, shape="circle"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
    # (H-K) per-clone metrics
    nc = len(clones); pc = {title: (metrics.per_clone(comp[:, :nc], Ncl.to_numpy()) if comp is not None else None) for title, comp in maps}
    gs2 = gs[1, :7].subgridspec(1, 4, wspace=0.28)
    for i, (met, title) in enumerate([("PCC", "PCC"), ("SSIM", "SSIM"), ("RMSE", "RMSE"), ("JS", "JS divergence")]):
        ax = fig.add_subplot(gs2[0, i]); panels.append((ax, "HIJK"[i]))
        for x, (name, _, c) in enumerate(METHODS):
            v = pc.get(name)
            if v is None:
                ax.text(x, 0.5, "not run", ha="center", va="center", fontsize=8, color="#9a9a9a", rotation=90, transform=ax.get_xaxis_transform()); continue
            vals = np.asarray(v[met]); bp = ax.boxplot([vals], positions=[x], widths=0.6, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=c, alpha=0.35, edgecolor=c)
            for k in ("whiskers", "caps", "medians"):
                for line in bp[k]:
                    line.set(color=c)
            ax.scatter(x + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals)), vals, s=12, color=c, zorder=3)
        ax.set_xticks(range(len(METHODS)), [m for m, _, _ in METHODS], rotation=35, ha="right"); ax.set_title(title, fontsize=11)
        ax.grid(color=GRID, axis="y"); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]
    # legend: the clades, then the other colours shown
    lax = fig.add_subplot(gs[0, 7]); lax.axis("off")
    for k in [k for k in clones + ["nontumor", "notrace"] if k in seen]:
        lax.scatter([], [], c=col[k], s=40, marker="s", label=names[k])
    lax.legend(loc="center left", fontsize=8, frameon=False, ncol=1, title="label", title_fontsize=9, markerscale=1.2, handletextpad=0.3, labelspacing=0.35)
    figlayout.finish(fig, panels)
    stem = "fig_petracer_methods" if section == "M3-4" else f"fig_petracer_{section}"; pdf = paths.figure(f"{stem}.pdf")
    fig.savefig(pdf, bbox_inches="tight", dpi=250); figlayout.compress_pdf(pdf); fig.savefig(paths.figure(f"{stem}.png"), dpi=70, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({m: ({k: round(float(np.mean(x)), 3) for k, x in v.items()} if v else None) for m, v in pc.items()}))


if __name__ == "__main__":
    main(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("section=")), "M3-4"))
