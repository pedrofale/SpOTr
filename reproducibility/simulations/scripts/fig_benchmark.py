"""Fig. 2 (fig_sim_benchmark) and Fig. S1 (fig_sim_metrics): the simulation benchmark (environment benchmark_env; CPU).

(A) A UMAP of the dissociated cells of the example tumor (the base world, seed 1), the tumor cells by clone and the non-tumor cells in beige.
(B) Its Visium spots, each colored in two levels: non-tumor where the spot's non-tumor cells outnumber its tumor cells, otherwise its majority
    clone.  (C) The same from SpOTr (non-tumor against tumor from its deconvolution, the clones from its map) and from cell2location (from its
    abundances of each label, the clones through the true clones of each label's cells).
(D-F) The per-clone Pearson correlation between each clone's true and estimated share of the spots (common/metrics.py, clones with at least 20
    cells), the clones of both seeds pooled, one box per method, against the strength of the heritable programs (D), the number of dispersed
    clones (E) and the number of clones (F).
Fig. S1: the four metrics of Li et al. (2022), averaged over the clones of each simulated tumor, every tumor of the three studies pooled.

Usage: python fig_benchmark.py   (reads config.yaml and its results; writes to common.paths.FIGURES)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from common import paths, metrics, figlayout  # noqa: E402
import world  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

GRID, NT = "#e6e5e0", "#d8cfbd"
METHODS = [("SpOTr", "#F47B20"), ("cell2location", "#1f77b4"), ("DestVI", "#9467bd"), ("Tangram", "#2ca02c"), ("scDOT", "#d62728"),
           ("expression-only OT", "#7f7f7f"), ("FGW", "#8c564b")]
STUDIES = [("heritable", "Clone expression", "heritable program scale"), ("migration", "Dispersed clones", "dispersed clones"),
           ("clones", "Number of clones", "clones")]


def tidy(ax):
    ax.grid(color=GRID, axis="y"); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]


def box(ax, x, v, w, cc):
    bp = ax.boxplot([v], positions=[x], widths=w, patch_artist=True, showfliers=False)
    for bx in bp["boxes"]:
        bx.set(facecolor=cc, alpha=0.35, edgecolor=cc)
    for k in ("whiskers", "caps", "medians"):
        for l in bp[k]:
            l.set(color=cc, lw=1.5 if k == "medians" else 0.9)


def umap(ax, X, colours, title, seed=0):
    """counts normalised to 10^4 and log1p, 30 principal components, 15-neighbour graph, UMAP (scanpy)"""
    import anndata as ad, scanpy as sc
    a = ad.AnnData(np.asarray(X, np.float32)); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a); sc.pp.pca(a, n_comps=30, random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, random_state=seed); sc.tl.umap(a, random_state=seed)
    cats = list(dict.fromkeys(colours.tolist())); a.obs["c"] = pd.Categorical(colours, categories=cats); a.uns["c_colors"] = cats
    sc.pl.umap(a, color="c", ax=ax, show=False, frameon=False, legend_loc=None, size=12, title=title)
    for c in ax.collections:
        c.set_rasterized(True)
    ax.title.set_fontsize(11); ax.set_xlabel(""); ax.set_ylabel("")


def scores(cfg, root):
    """per world-seed and per clone, every method's scores"""
    rows, crow = [], []
    for study, knob, value, w, _ in world.studies(cfg):
        for s in cfg["seeds"]:
            d = root / "worlds" / w / f"seed{s}"
            if not (d / "problem.npz").exists():
                continue
            ms, q = world.masses(d)
            for nm, Pm in ms.items():
                sc = metrics.scores(Pm, q["Ncl"]); rows.append(dict(study=study, value=value, world=w, seed=s, method=nm, **{k: sc[k] for k in ("PCC", "SSIM", "RMSE", "JS")}))
                crow += [dict(study=study, value=value, world=w, seed=s, clone=i, method=nm, PCC=v) for i, v in enumerate(metrics.per_clone(Pm, q["Ncl"])["PCC"])]
    df = pd.DataFrame(rows).drop_duplicates(["world", "seed", "method"]); return df, pd.DataFrame(crow)


def tick(root, cfg, study, value, w):
    """the varied value as the worlds hold it: the number of dispersed clones, or of clones"""
    if study == "heritable":
        return f"{value:g}"
    n = []
    for s in cfg["seeds"]:
        d = root / "worlds" / w / f"seed{s}"
        if (d / "problem.npz").exists():
            n.append(len(np.load(d / "sim.npz")["migrating"]) if study == "migration" else int(np.load(d / "problem.npz")["K"]) - 1)
    return "/".join(str(x) for x in sorted(set(n))) if n else str(value)


def main():
    import warnings; warnings.filterwarnings("ignore")
    from spotr.plotting import spots
    cfg = yaml.safe_load(open(HERE.parent / "config.yaml")); root = HERE.parent / cfg["out"]
    df, dc = scores(cfg, root); print(df.groupby("method")[["PCC", "SSIM", "RMSE", "JS"]].mean().round(3).to_string())
    fig = plt.figure(figsize=(19, 10.0)); gs = fig.add_gridspec(2, 12, hspace=0.3, wspace=1.6); panels = []
    d = world.example(cfg, HERE.parent); q = np.load(d / "problem.npz", allow_pickle=True); K = int(q["K"]); tu = q["kind"] == 0; tr = q["traced"].astype(bool)
    tab = plt.get_cmap("tab10"); ccol = [to_hex(tab(k)) for k in range(K - 1)] + ["#bdbdbd"]; oh = np.eye(K)[q["clone"][tr]]
    sp = np.asarray(q["spots"], float); h = float(q["pitch"]) / 2; ext = (sp[:, 0].min() - h, sp[:, 0].max() + h, sp[:, 1].min() - h, sp[:, 1].max() + h)
    two_level = lambda non, tum, C: [NT if n >= t else ccol[int(np.argmax(c))] for n, t, c in zip(non, tum, C)]
    def draw(ax, cols, title):
        spots(ax, sp, cols, title, fontsize=11, shape="circle", pitch=1.6 * float(q["side"])); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
    ax = fig.add_subplot(gs[0, 0:3]); panels.append((ax, "A"))
    umap(ax, q["Xc"], np.array([ccol[c] if t else NT for c, t in zip(q["clone"], tu)]), "scRNA-seq UMAP")
    axB = fig.add_subplot(gs[0, 3:6]); panels.append((axB, "B")); draw(axB, two_level(q["Nme"].sum(1), q["Ncl"].sum(1), q["Ncl"]), "Visium spots (truth)")
    s = np.load(d / "spotr.npz", allow_pickle=True); P = s["proportions"].astype(float); lab = [str(x) for x in s["labels"]]; me = [i for i, l in enumerate(lab) if l.startswith("me")]
    ax = fig.add_subplot(gs[0, 6:9]); panels.append((ax, "C"))
    draw(ax, two_level(P[:, me].sum(1), np.delete(P, me, axis=1).sum(1), s["phi"].astype(float).T @ oh), "SpOTr")
    z = np.load(d / "c2l.npz", allow_pickle=True); W = z["w"].astype(float); mw = [i for i, t in enumerate(z["types"]) if str(t).startswith("me")]
    axC = fig.add_subplot(gs[0, 9:12]); panels.append((axC, ""))
    draw(axC, two_level(W[:, mw].sum(1), np.delete(W, mw, axis=1).sum(1), world.masses(d)[0]["cell2location"]), "cell2location")
    hd = [Line2D([], [], ls="", marker="o", ms=8, color=ccol[k], label=f"clone {k + 1}") for k in range(K - 1)]
    hd += [Line2D([], [], ls="", marker="o", ms=8, color=ccol[K - 1], label="smaller clones, pooled"), Line2D([], [], ls="", marker="o", ms=8, color=NT, label="non-tumor cells")]
    table = world.studies(cfg); w_ = 0.8 / len(METHODS)
    for i, (st, title, xl) in enumerate(STUDIES):
        ax = fig.add_subplot(gs[1, 4 * i:4 * i + 4]); panels.append((ax, "DEF"[i])); rows = [(v, w) for s_, _, v, w, _ in table if s_ == st]
        for j, (v, w) in enumerate(rows):
            for k, (nm, cc) in enumerate(METHODS):
                pv = dc[(dc.study == st) & np.isclose(dc.value.astype(float), float(v)) & (dc.method == nm)].PCC.to_numpy()
                if len(pv):
                    x = j + (k - (len(METHODS) - 1) / 2) * w_; box(ax, x, pv, w_ * 0.85, cc)
                    ax.scatter(x + np.random.default_rng(j * 10 + k).uniform(-w_ * 0.25, w_ * 0.25, len(pv)), pv, s=5, color=cc, zorder=3, lw=0)
        ax.set_xticks(range(len(rows)), [tick(root, cfg, st, v, w) for v, w in rows]); ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_title(title, fontsize=11); ax.set_xlabel(xl, fontsize=10); ax.set_ylabel("per-clone PCC" if i == 0 else ""); ax.set_ylim(-0.45, 1.0); tidy(ax)
    fig.legend(handles=[Patch(fc=cc, ec=cc, alpha=0.6, label=nm) for nm, cc in METHODS], loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=len(METHODS),
               frameon=False, fontsize=10)
    figlayout.finish(fig, panels); fig.canvas.draw(); pB, pC = axB.get_position(), axC.get_position()
    fig.legend(handles=hd, title="majority label", loc="upper center", bbox_to_anchor=((pB.x0 + pC.x1) / 2, pB.y0 - 0.005), ncol=len(hd), frameon=False, fontsize=9,
               title_fontsize=9, columnspacing=1.0, handletextpad=0.3)
    f = paths.figure("fig_sim_benchmark.pdf"); fig.savefig(f, bbox_inches="tight", dpi=250); figlayout.compress_pdf(f)
    fig.savefig(f.with_suffix(".png"), dpi=60, bbox_inches="tight"); plt.close(fig)
    supp_metrics(df)


def supp_metrics(df):
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.8)); fig.subplots_adjust(wspace=0.3); panels = []
    for i, (met, title) in enumerate([("PCC", "PCC"), ("SSIM", "SSIM"), ("RMSE", "RMSE"), ("JS", "JS divergence")]):
        ax = axs[i]; panels.append((ax, "ABCD"[i]))
        for x, (nm, cc) in enumerate(METHODS):
            v = df.loc[df.method == nm, met].to_numpy()
            if len(v):
                box(ax, x, v, 0.6, cc); ax.scatter(x + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=10, color=cc, zorder=3)
        ax.set_xticks(range(len(METHODS)), [m for m, _ in METHODS], rotation=45, ha="right", fontsize=10); ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.6, len(METHODS) - 0.4); tidy(ax)
    figlayout.finish(fig, panels); f = paths.figure("fig_sim_metrics.pdf"); fig.savefig(f, bbox_inches="tight"); figlayout.compress_pdf(f)
    fig.savefig(f.with_suffix(".png"), dpi=70, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
