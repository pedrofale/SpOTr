"""Figures of the benchmark (run in the spotr environment).

fig_benchmark.pdf   per study, the per-spot subclone error of every method against the study's knob (mean over seeds; dots: seeds)
fig_ablation.pdf    (A) sub-clade resolution in the heritable study; (B) the tree's gain over no tree against migration; (C) the true against the
                    reconstructed tree, and barcode saturation; (D) learned against true spreads (clone labels) and the true-spread ablation;
                    (E) the lineage-resolution readout's precision and recall per study
fig_example.pdf     the standard example figure on one world: the dissociated traced cells and the spots (UMAP), the section by true dominant cell
                    type or clone and each clone's true spread, the dominant cell type or clone per spot for SpOTr (lineage tree, clone labels, no
                    tree), exact OT, Tangram and cell2location (slides with squidpy); the learning (log-likelihood and spreads along the rounds,
                    each fit from the same start); true against learned spreads

Usage: python figures.py <summary.csv> <results dir> <example world dir> <out dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Liberation Sans", "pdf.fonttype": 42, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984"})
INK, GRID, NONE, OTHER = "#111111", "#e6e5e0", "#ecebe6", "#9a9994"
M = {"lineage": ("SpOTr (lineage tree)", "#0b3a7e"), "clone_tree": ("SpOTr (clone tree)", "#2a6fc9"), "clones": ("SpOTr (clone labels)", "#7fb0ea"),
     "notree": ("SpOTr without tree", "#c3d3e8"), "c2l": ("cell2location", "#8a3fb0"), "destvi": ("DestVI", "#8a8984"), "tangram": ("Tangram", "#c9c8c2"), "exact_ot": ("Exact OT", "#3a3a3a"),
     "clones_true_spreads": ("clone labels, true spreads", "#eb6834")}
CLONES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#4cc1d0", "#a35f2d", "#0b3a7e", "#f3cf6b", "#7fd1b0", "#8a3fb0"]


def letter(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom", color=INK)


def grid(ax):
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)


def lines(ax, df, metric, methods, label_x):
    for m in methods:
        d = df[(df.method == m) & (df.metric == metric)]
        if d.empty:
            continue
        xs = sorted(d.value.unique(), key=lambda v: (str(type(v)), v)); xi = np.arange(len(xs)); mean = [d[d.value == v].score.mean() for v in xs]
        ax.plot(xi, mean, "o-", color=M[m][1], label=M[m][0], lw=1.8, ms=5)
        for i, v in enumerate(xs):
            ax.plot([i] * len(d[d.value == v]), d[d.value == v].score, ".", color=M[m][1], alpha=0.5, ms=4)
        ax.set_xticks(xi, [str(v) for v in xs])
    ax.set_xlabel(label_x); grid(ax)


def main(summary, res, ex, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True); S = pd.read_csv(summary); studies = list(dict.fromkeys(S.study))
    meth = ["lineage", "clone_tree", "clones", "notree", "c2l", "destvi", "tangram", "exact_ot"]
    # ---- benchmark
    fig, axs = plt.subplots(1, len(studies), figsize=(4.2 * len(studies), 4), squeeze=False); axs = axs[0]
    for i, st in enumerate(studies):
        d = S[S.study == st]; lines(axs[i], d, "clone_tv", meth, d.knob.iloc[0]); axs[i].set_title(st, fontsize=10, loc="left"); letter(axs[i], "ABCDEFGH"[i])
    hl = {}
    for ax_ in axs:
        for h, l_ in zip(*ax_.get_legend_handles_labels()):
            hl.setdefault(l_, h)
    order = [M[m][0] for m in meth if M[m][0] in hl]; axs[0].set_ylabel("per-spot subclone error")
    fig.legend([hl[l_] for l_ in order], order, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(order))
    fig.savefig(out / "fig_benchmark.pdf", bbox_inches="tight"); fig.savefig(out / "fig_benchmark.png", dpi=70, bbox_inches="tight"); plt.close(fig)
    # ---- ablations
    fig, axs = plt.subplots(1, 5, figsize=(23, 4.2), gridspec_kw=dict(wspace=0.35))
    d = S[(S.study == "heritable") & S.method.isin(["lineage", "clone_tree", "clones", "notree"])]; ks = ["clone_tv", "subclade_tv_l1", "subclade_tv_l2", "subclade_tv_l3"]
    for m in ["notree", "clones", "clone_tree", "lineage"]:
        axs[0].plot(range(4), [d[(d.method == m) & (d.metric == k) & (d.value.astype(float) > 0)].score.mean() for k in ks], "o-", color=M[m][1], label=M[m][0])
    axs[0].set_xticks(range(4), ["clones", "level 1", "level 2", "level 3"]); axs[0].set_ylabel("per-spot error"); axs[0].legend(frameon=False, fontsize=8); grid(axs[0])
    axs[0].set_title("Resolution (heritable study, tree_sd > 0)", fontsize=10, loc="left"); letter(axs[0], "A")
    d = S[(S.study == "migration") & (S.metric == "clone_tv")]
    if not d.empty:
        g = d.pivot_table(index=["value", "seed"], columns="method", values="score")
        for m in ["clones", "clone_tree", "lineage"]:
            if m in g:
                gg = (g["notree"] - g[m]).groupby(level=0).mean(); axs[1].plot(range(len(gg)), gg.values, "o-", color=M[m][1], label=M[m][0]); axs[1].set_xticks(range(len(gg)), [str(v) for v in gg.index])
        axs[1].axhline(0, color=INK, lw=0.8); axs[1].set_xlabel("share of clones migrating"); axs[1].set_ylabel("error without tree minus with tree"); grid(axs[1])
    axs[1].set_title("When lineage and space disagree", fontsize=10, loc="left"); letter(axs[1], "B")
    d = S[S.study.isin(["tree_source"]) & (S.metric == "clone_tv") & S.method.isin(["lineage", "clone_tree", "clones", "notree"])].copy()
    if not d.empty:
        d["cond"] = d.study + ": " + d.value.astype(str); conds = list(dict.fromkeys(d.cond)); w = 0.2
        for j, m in enumerate(["notree", "clones", "clone_tree", "lineage"]):
            axs[2].bar(np.arange(len(conds)) + (j - 1.5) * w, [d[(d.cond == c) & (d.method == m)].score.mean() for c in conds], width=w, color=M[m][1], label=M[m][0])
        axs[2].set_xticks(range(len(conds)), conds, rotation=30, ha="right", fontsize=8); axs[2].set_ylabel("per-spot subclone error"); grid(axs[2])
    axs[2].set_title("True or reconstructed tree", fontsize=10, loc="left"); letter(axs[2], "C")
    d = S[(S.metric.isin(["clone_tv"])) & S.method.isin(["clones", "clones_true_spreads"])]
    r = S[(S.metric == "spread_ratio") & (S.method == "clones")]
    axs[3].hist(r.score, bins=20, color=M["clones"][1]); axs[3].axvline(1, color=INK, lw=0.8); axs[3].set_xlabel("learned / true spread (clone labels, every world)")
    if not d.empty:
        g = d.pivot_table(index=["world", "seed"], columns="method", values="score").dropna()
        axs[3].set_title(f"Spreads; true spreads change the error by {(g['clones_true_spreads'] - g['clones']).mean():+.3f}", fontsize=9.5, loc="left")
    letter(axs[3], "D")
    d = S[(S.method == "readout") & S.metric.isin(["precision", "recall"])]
    if not d.empty:
        g = d.groupby(["study", "metric"]).score.mean().unstack(); x = np.arange(len(g))
        axs[4].bar(x - 0.2, g["precision"], width=0.4, color="#0b3a7e", label="precision"); axs[4].bar(x + 0.2, g["recall"], width=0.4, color="#7fb0ea", label="recall")
        axs[4].set_xticks(x, g.index, rotation=30, ha="right", fontsize=8); axs[4].set_ylim(0, 1.05); axs[4].legend(frameon=False, fontsize=8); grid(axs[4])
    axs[4].set_title("Lineage-resolution readout against the truth", fontsize=10, loc="left"); letter(axs[4], "E")
    fig.savefig(out / "fig_ablation.pdf", bbox_inches="tight"); fig.savefig(out / "fig_ablation.png", dpi=70, bbox_inches="tight"); plt.close(fig)
    # ---- the standard example figure (slides with squidpy: spotr.plot_spatial; each spot's dominant cell type or clone)
    from spotr.plot_spatial import slide as sslide
    E = Path(ex); q = np.load(E / "problem.npz"); s = np.load(E / "spotr.npz"); rec = json.load(open(E / "spotr.json"))["record"]; sc = json.load(open(E / "scores.json"))
    K = int(q["K"]); Ncl = q["Ncl"].astype(float); Nme = q["Nme"].astype(float); tr = q["traced"]; ct = q["clone"][tr]; onehot = np.eye(K)[ct]; spots, side = q["spots"], float(q["side"]); shp = str(q["shape"]) if "shape" in q.files else "square"
    kind, met = q["kind"], q["me_type"]; top = list(np.argsort(-Ncl[:, :K - 1].sum(0)))[: len(CLONES)]; ME_C, ME_N = ["#dcdad3", "#c2b49f", "#6e6b63"], ["fibroblast", "T cell", "endothelial"]
    def dom(me, cl):
        comp = np.concatenate([me, cl], 1); d = comp.argmax(1); tot = comp.sum(1)
        return [NONE if t <= 0 else (ME_C[k] if k < 3 else (CLONES[top.index(k - 3)] if (k - 3) in top else OTHER)) for k, t in zip(d, tot)]
    def spotr_map(arm):
        Ab = np.asarray(rec[arm]["abund"], float); P = s[f"phi_{arm}"].astype(float).T @ onehot; sh = P / np.maximum(P.sum(1, keepdims=True), 1e-300); return Ab[:, :3], Ab[:, [3]] * sh
    def cell_map(P):
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-300); me = P[kind == 1].T @ np.eye(3)[met[kind == 1]]; cl = P[tr].T @ onehot
        return me, cl / np.maximum(cl.sum(1, keepdims=True), 1e-300) * P[kind == 0].sum(0)[:, None]
    from sklearn.decomposition import PCA
    import umap
    def um(A):
        L = np.log1p(A / np.maximum(A.sum(1, keepdims=True), 1) * 1e4); Z = PCA(n_components=min(30, L.shape[1] - 1), random_state=0).fit_transform(L - L.mean(0))
        return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(Z)
    fig = plt.figure(figsize=(22, 15)); gs = fig.add_gridspec(3, 6, hspace=0.4, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0]); U = um(q["Xc"][tr].astype(float)); ax.scatter(*U.T, c=[CLONES[top.index(k)] if k in top else OTHER for k in ct], s=3, lw=0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Dissociated traced cells (UMAP)", fontsize=9.5); letter(ax, "A")
    ax = fig.add_subplot(gs[0, 1]); U = um(q["Y"].astype(float)); ax.scatter(*U.T, c=dom(Nme, Ncl), s=4, lw=0); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Spots by true dominant type or clone (UMAP)", fontsize=9.5)
    ax = fig.add_subplot(gs[0, 2]); sslide(ax, spots, dom(Nme, Ncl), "Truth: dominant cell type or clone", pitch=side, shape=shp, scalebar_um=200, flip=True, fontsize=9.5)
    for c_, n_ in zip(ME_C, ME_N):
        ax.scatter([], [], c=c_, marker="s", s=25, label=n_)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    xyt = q["xy_true"][tr]; tru = np.array([np.sqrt(((xyt[ct == k] - xyt[ct == k].mean(0)) ** 2).sum(1).mean() / 2.0) if (ct == k).sum() > 1 else np.nan for k in range(K - 1)])
    ax = fig.add_subplot(gs[0, 3]); kk = [k for k in top if np.isfinite(tru[k])]; ax.bar(range(len(kk)), tru[kk], color=[CLONES[top.index(k)] for k in kk]); ax.set_xticks(range(len(kk)), [str(i + 1) for i in range(len(kk))])
    ax.set_xlabel("clone"); ax.set_ylabel("true spread (um)"); ax.set_title("True spread of each clone", fontsize=9.5); grid(ax)
    maps = [(a, spotr_map(a)) for a in ["lineage", "clones", "notree"]]
    for nm in ["exact_ot", "tangram"]:
        if (E / f"{nm}.npz").exists():
            maps.append((nm, cell_map(np.load(E / f"{nm}.npz")["P"].astype(float))))
    if (E / "c2l.npz").exists():
        z = np.load(E / "c2l.npz", allow_pickle=True); ty = [str(t) for t in z["types"]]; w = z["w"].astype(float); col = lambda n: w[:, ty.index(n)] if n in ty else np.zeros(len(w))
        cl = np.stack([col(f"c{k}") for k in range(K - 1)] + [col("pooled")], 1); tum = cl.sum(1) + col("untraced")
        maps.append(("c2l", (np.stack([col(f"me{e}") for e in range(3)], 1), cl / np.maximum(cl.sum(1, keepdims=True), 1e-300) * tum[:, None])))
    for j, (nm, (me, cl)) in enumerate(maps):
        ax = fig.add_subplot(gs[0, 4 + j] if j < 2 else gs[1, j - 2]); sslide(ax, spots, dom(me, cl), f"{M[nm][0]}\nerror {sc[nm]['clone_tv']:.2f}", pitch=side, shape=shp, flip=True, fontsize=9.5)
    ax = fig.add_subplot(gs[2, 0:2]); t0, t1 = rec["notree"]["trace"], rec["lineage"]["trace"]
    ax.plot(range(1, len(t0) + 1), [x["loglik"] for x in t0], "o-", color=M["notree"][1], label="without tree"); ax.plot(range(1, len(t1) + 1), [x["loglik"] for x in t1], "o-", color=M["lineage"][1], label="lineage tree")
    ax.set_xlabel("round (each fit from the same start)"); ax.set_ylabel("spot log-likelihood"); ax.legend(frameon=False, fontsize=8); grid(ax); ax.set_title("Learning: likelihood", fontsize=10, loc="left"); letter(ax, "B")
    ax = fig.add_subplot(gs[2, 2:4]); S2 = np.sqrt(np.array([x["s2"] for x in t1])); used = s["clone_used"]
    for k in range(S2.shape[1]):
        ax.plot(range(1, len(t1) + 1), S2[:, k], "-", color=CLONES[top.index(k)] if k in top else OTHER, lw=1.5)
    ax.set_xlabel("round (lineage tree)"); ax.set_ylabel("clone spread (um)"); grid(ax); ax.set_title("Learning: clone spreads", fontsize=10, loc="left")
    ax = fig.add_subplot(gs[2, 4:6]); l = S2[-1]; tu = np.array([np.sqrt(((xyt[used == c] - xyt[used == c].mean(0)) ** 2).sum(1).mean() / 2.0) for c in range(len(l))])
    ax.plot(tu, l, "o", color=M["lineage"][1]); hi = 1.1 * max(tu.max(), l.max()); ax.plot([0, hi], [0, hi], color=INK, lw=0.7); ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_box_aspect(1)
    ax.set_xlabel("true spread (um)"); ax.set_ylabel("learned spread (um)"); grid(ax); ax.set_title("True against learned spreads", fontsize=10, loc="left"); letter(ax, "C")
    fig.savefig(out / "fig_example.pdf", bbox_inches="tight"); fig.savefig(out / "fig_example.png", dpi=60, bbox_inches="tight"); plt.close(fig); print("figures written")


if __name__ == "__main__":
    main(*sys.argv[1:5])
