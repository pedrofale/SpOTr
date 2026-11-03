"""The paper's simulation benchmark figure (spotr.tex, fig:sim): SpOTr's target scenario (vault exp-unmatched-oct12 section 190).

Run: the Cassiopeia workflow's config.yaml (results_territories/; tumour cells in nested territories along the tree, 4-5 clones, 2,000 genes),
8 worlds x seeds 1-2, each varied parameter with the others at the base world.
(A) The base world, seed 1: a UMAP of the dissociated cells' expression (scanpy), tumor cells by their true clone, non-tumor cells in beige --
    the clones weakly distinguishable by expression.  (B) The section half's Visium spots in two levels (non-tumor where non-tumor cells
    outnumber tumor cells, else the majority clone), drawn at 1.6 x the spot diameter as the HTAN panels -- the clones in clear territories.
(C) The same for SpOTr (Pedro, 2026-11-03: "a panel with 2 subpanels, showing SpOTr's result and scDOT next to it"): non-tumor against tumor
    from its deconvolution (470_sim_example_abund.py, spotr_example.npz), the clone from the benchmark's map; and for scDOT, from its map of
    every dissociated cell.
(D-F) The per-clone Pearson correlation between each clone's true and estimated share of the spots (451_.per_clone, clones with at least
    20 cells), the clones of both seeds pooled, one box per method, against the strength of the heritable programs (D), the number of
    clones dispersed over the section (E) and the number of clones (F), as counted in the worlds.  A method is shown once it has run on every
    world-seed.  The four pooled metrics (289_field_metrics.scores) in the supplementary fig_sim_metrics.
Style: decision-writing-style (SpOTr orange and first); spatial panels through squidpy (plot_spatial); panels aligned (figlayout.finish).

Usage: python 455_fig_sim_benchmark.py   (benchmark_env, CPU; writes SpOTr-overleaf/figures/fig_sim_benchmark.{pdf,png} and fig_sim_metrics.{pdf,png})
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; WF = ROOT / "workflows" / "cassiopeia_benchmark" / "results"; FIG = ROOT.parent / "SpOTr-overleaf" / "figures"
sys.path.insert(0, str(ROOT / "src")); INK = "#111111"; GRID = "#e6e5e0"
METHODS = [("SpOTr", "#F47B20"), ("cell2location", "#1f77b4"), ("DestVI", "#9467bd"), ("Tangram", "#2ca02c"), ("scDOT", "#d62728"), ("expression-only OT", "#7f7f7f"),
           ("FGW", "#8c564b")]   # SpOTr with the lineage tree, one mode (Pedro, 2026-11-02)


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


PANELS = []                                                                                   # (axes, letter), laid out by spotr.figlayout.finish


def letter(ax, s, x=None, y=None):
    """registers the panel; spotr.figlayout.finish puts the letter on its column's edge and aligns the panels with the figure's edges"""
    PANELS.append((ax, s))


def umap_cells(ax, X, colours, title, seed=0):
    """scanpy: counts normalised to 10^4 and log1p, 30 principal components, 15-neighbour graph, UMAP; drawn with sc.pl.umap"""
    import anndata as ad, scanpy as sc
    a = ad.AnnData(np.asarray(X, np.float32)); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a); sc.pp.pca(a, n_comps=30, random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, random_state=seed); sc.tl.umap(a, random_state=seed)
    cats = list(dict.fromkeys(colours.tolist())); a.obs["c"] = pd.Categorical(colours, categories=cats); a.uns["c_colors"] = cats
    sc.pl.umap(a, color="c", ax=ax, show=False, frameon=False, legend_loc=None, size=12, title=title)
    for c_ in ax.collections:
        c_.set_rasterized(True)
    ax.title.set_fontsize(11); ax.set_xlabel(""); ax.set_ylabel("")


def example_world(study=None, knob=None, value=None, seed=1):
    """the example world's problem as the workflow's config.yaml now defines it (world id as in the Snakefile); by default the base world"""
    import yaml, hashlib
    c = yaml.safe_load(open(WF.parent / "config.yaml")); p = dict(c["base"])
    if study is not None:
        p.update(c["studies"][study].get("fix", {})); p[knob] = value
    return WF.parent / c["out"] / "worlds" / hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10] / f"seed{seed}" / "problem.npz"


def current_run():
    """the run the workflow's config.yaml now defines (Visium spots, current engine) once all of its world-seeds are scored, else the square-bin run of
    2026-10-30/31; and its table of (study, knob, value, world) as the Snakefile builds it"""
    import yaml, hashlib
    c = yaml.safe_load(open(WF.parent / "config.yaml")); out = WF.parent / c["out"]; table = []
    for st, spec in c["studies"].items():
        (knob, values), = spec["vary"].items()
        for v in values:
            if st == "subclonal_mixing":                                                          # dropped from the paper (Pedro, 2026-11-02); kept on disk
                continue
            p = dict(c["base"]); p.update(spec.get("fix", {})); p[knob] = v; table.append((st, knob, v, hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]))
    done = [(out / "worlds" / w / f"seed{sd}" / "problem.npz").exists() for _, _, _, w in table for sd in c["seeds"]]
    if done and all(done):                                                                       # every world-seed simulated: each method shown once it has
        return out, table, True                                                                  # run on all of them (else "pending")
    return WF, json.load(open(WF / "worlds_table.json")), False


def masses(d):
    """each method's clone mass per spot (spots x K) in one world-seed directory, as the benchmark's score.py builds it: maps of cells through each
    traced cell's true clone (rows normalised); DestVI's proportion per reference label through the labelled cells' true clones"""
    pr = np.load(d / "problem.npz", allow_pickle=True); K = int(pr["K"]); tr = pr["traced"].astype(bool); oh = np.eye(K)[pr["clone"][tr]]; out = {}
    rn = lambda P: P / np.maximum(P.sum(1, keepdims=True), 1e-300); used = None
    if (d / "spotr.npz").exists():                                                                    # the current engine at the fixed alpha (or the old run's arms)
        s = np.load(d / "spotr.npz"); used = s["clone_used"]; new = "phi_tree" in s.files
        out["SpOTr"] = np.asarray(s["phi_tree" if new else "phi_lineage"], float).T @ oh
    if used is None and (d / "trees.npz").exists():
        t = np.load(d / "trees.npz"); used = t["clone_true_tree"]
    for f, nm in (("exact_ot.npz", "expression-only OT"), ("tangram.npz", "Tangram"), ("scdot.npz", "scDOT"), ("fgw.npz", "FGW")):
        if (d / f).exists():
            out[nm] = rn(np.load(d / f)["P"].astype(float)[np.where(tr)[0]]).T @ oh
    for f, nm in (("destvi.npz", "DestVI"), ("c2l.npz", "cell2location")):                         # proportions / abundances per label, through the
        if (d / f).exists() and used is not None:                                                      # labelled cells' true clones (score.py)
            z = np.load(d / f, allow_pickle=True); ty = [str(t) for t in z["types"]]; W = z["w"].astype(float); lab = np.array([f"c{c}" if c >= 0 else "pooled" for c in used])
            P = np.zeros((len(W), K))
            for L in np.unique(lab):
                if L in ty:
                    P += W[:, [ty.index(L)]] * oh[lab == L].mean(0)[None, :]
            out[nm] = P
    return out, pr


def main():
    import warnings; warnings.filterwarnings("ignore")
    from spotr.plot_spatial import slide, cells
    f289 = mod("f289", "289_field_metrics.py"); RUN, table, NEW = current_run(); print("run:", RUN.name)
    rows, crow = [], []; m451 = mod("m451", "451_fig_petracer_paper.py")
    import yaml; SEEDS = {f"seed{s_}" for s_ in yaml.safe_load(open(WF.parent / "config.yaml"))["seeds"]}   # the benchmark's seeds only (seed 3: the tuning world)
    for study, knob, value, world in table:
        for sd in sorted(d_ for d_ in (RUN / "worlds" / world).glob("seed*") if d_.name in SEEDS):
            try:
                ms, pr = masses(sd)
            except Exception as e:
                continue
            for nm, Pm in ms.items():
                sc = f289.scores(Pm, pr["Ncl"]); rows.append(dict(study=study, value=value, world=world, seed=sd.name, method=nm, **{k: sc[k] for k in ("PCC", "SSIM", "RMSE", "JS")}))
                for ci_, v_ in enumerate(m451.per_clone(f289, Pm, pr["Ncl"], f289.MIN_CELLS)["PCC"]):
                    crow.append(dict(study=study, value=value, world=world, seed=sd.name, clone=ci_, method=nm, PCC=v_))
    df = pd.DataFrame(rows).drop_duplicates(["world", "seed", "method"]); dc = pd.DataFrame(crow); nws = df[["world", "seed"]].drop_duplicates().shape[0]
    if NEW:
        cnt = df.groupby("method").size(); df = df[df.method.isin(cnt[cnt == nws].index)]            # a method only once it has every world-seed
    print(f"{nws} world-seeds; complete:", sorted(df.method.unique())); print(df.groupby("method")[["PCC", "SSIM", "RMSE", "JS"]].mean().round(3).to_string())
    fig = plt.figure(figsize=(19, 11.5)); gs = fig.add_gridspec(2, 12, hspace=0.42, wspace=1.6)   # 2 rows of 3 panels (Pedro, 2026-11-03); C two subpanels
    def tidy(ax):
        ax.grid(color=GRID, axis="y"); ax.set_axisbelow(True); [ax.spines[s_].set_visible(False) for s_ in ("top", "right")]
    # (A, B) the example (base) world: clones weakly distinguishable by expression (A), in clear territories (B); (C) SpOTr's and scDOT's spots
    d_ex = example_world().parent; pr = np.load(d_ex / "problem.npz", allow_pickle=True)          # the measurement of the benchmark as it now stands
    K = int(pr["K"]); tab = plt.get_cmap("tab10"); ccol = [to_hex(tab(k)) for k in range(K - 1)] + ["#bdbdbd"]; NT = "#d8cfbd"
    ax = fig.add_subplot(gs[0, 0:3]); letter(ax, "A"); tu = pr["kind"] == 0
    sp_, h_ = np.asarray(pr["spots"], float), float(pr["pitch"] if "pitch" in pr.files else pr["side"]) / 2; ext = (sp_[:, 0].min() - h_, sp_[:, 0].max() + h_, sp_[:, 1].min() - h_, sp_[:, 1].max() + h_)
    umap_cells(ax, pr["Xc"], np.array([ccol[c] if t else NT for c, t in zip(pr["clone"], tu)]), "scRNA-seq UMAP")
    spot_kw = dict(shape="circle", pitch=1.6 * float(pr["side"])) if "shape" in pr.files and str(pr["shape"]) == "circle" else dict(shape="square")   # 1.6: as HTAN
    def two_level(non, tum, Cm):                                                                    # non-tumor vs tumor first, then the majority clone
        return [NT if n_ >= t_ else ccol[int(np.argmax(c_))] for n_, t_, c_ in zip(non, tum, Cm)]
    def spots(ax, cols, ttl):
        slide(ax, sp_, cols, ttl, fontsize=11, **spot_kw); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
    ax = fig.add_subplot(gs[0, 3:6]); letter(ax, "B"); axB = ax
    spots(ax, two_level(pr["Nme"].sum(1), pr["Ncl"].sum(1), pr["Ncl"]), "Visium spots (truth)")
    tr_ = pr["traced"].astype(bool); oh_ = np.eye(K)[pr["clone"][tr_]]; axC = []
    if (d_ex / "spotr_example.npz").exists() and (d_ex / "spotr.npz").exists():                     # SpOTr: the benchmark's map for the clones, the
        ab = np.load(d_ex / "spotr_example.npz")["abund"].astype(float); ne = int(pr["Nme"].shape[1])  # deconvolution (470_) for non-tumor vs tumor
        ax = fig.add_subplot(gs[0, 6:9]); letter(ax, "C"); axC.append(ax)
        spots(ax, two_level(ab[:, :ne].sum(1), ab[:, ne:].sum(1), np.load(d_ex / "spotr.npz")["phi_tree"].astype(float).T @ oh_), "SpOTr")
    if (d_ex / "scdot.npz").exists():                                                               # scDOT: its map of every dissociated cell
        P_ = np.load(d_ex / "scdot.npz")["P"].astype(float); P_ = P_ / np.maximum(P_.sum(1, keepdims=True), 1e-300)
        ax = fig.add_subplot(gs[0, 9:12]); letter(ax, "" if axC else "C"); axC.append(ax)
        spots(ax, two_level(P_[~tu].sum(0), P_[tu].sum(0), P_[tr_].T @ oh_), "scDOT")
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], ls="", marker="o", ms=8, color=ccol[k], label=f"clone {k + 1}") for k in range(K - 1)] + [Line2D([], [], ls="", marker="o", ms=8, color=ccol[K - 1], label="smaller clones, pooled"),
          Line2D([], [], ls="", marker="o", ms=8, color=NT, label="non-tumor cells")]
    # (D-F) per-clone PCC against each varied parameter, the others at the base world (each axis's middle level); one box per method and value,
    # the clones of both seeds pooled (Pedro, 2026-11-02: "pick a main metric, and then vary these simulation parameters")
    dc = dc[dc.method.isin(df.method.unique())] if len(dc) else dc
    STUD = [("heritable", "Clone expression", "heritable program scale"), ("migration", "Dispersed clones", "dispersed clones"), ("clones", "Number of clones", "clones")]   # true tree only; no "All scenarios" panel (Pedro, 2026-11-02)
    def tick(st, v, w):                                                                            # scattered clones and clones as counted in the worlds
        if st == "heritable":
            return f"{v:g}"
        n = [len(np.load(d_ / "sim.npz")["migrating"]) if st == "migration" else int(np.load(d_ / "problem.npz")["K"]) - 1 for d_ in sorted((RUN / "worlds" / w).glob("seed[12]")) if (d_ / "problem.npz").exists()]
        return "/".join(str(x) for x in sorted(set(n))) if n else str(v)
    for i, (st, ttl, xl) in enumerate(STUD):
        ax = fig.add_subplot(gs[1, 4 * i:4 * i + 4]); letter(ax, "DEF"[i]); vals = [v for s_, k_, v, w in table if s_ == st]; wv = {v: w for s_, k_, v, w in table if s_ == st}; w_ = 0.8 / len(METHODS)
        for j_, v in enumerate(vals):
            for m_, (nm, cc) in enumerate(METHODS):
                x_ = j_ + (m_ - (len(METHODS) - 1) / 2) * w_
                if not len(dc):
                    pv = []
                else:
                    pv = dc[(dc.study == st) & np.isclose(dc.value.astype(float), float(v)) & (dc.method == nm)].PCC.to_numpy()   # numeric: 3 and 3.0
                if len(pv) == 0:
                    continue
                bp = ax.boxplot([pv], positions=[x_], widths=w_ * 0.85, patch_artist=True, showfliers=False)
                for bx in bp["boxes"]:
                    bx.set(facecolor=cc, alpha=0.35, edgecolor=cc)
                for k_ in ("whiskers", "caps", "medians"):
                    for l_ in bp[k_]:
                        l_.set(color=cc, lw=1.5 if k_ == "medians" else 0.9)
                ax.scatter(x_ + np.random.default_rng(j_ * 10 + m_).uniform(-w_ * 0.25, w_ * 0.25, len(pv)), pv, s=5, color=cc, zorder=3, lw=0)
        ax.set_xticks(range(len(vals)), [tick(st, v, wv[v]) for v in vals]); ax.set_xlim(-0.6, len(vals) - 0.4); ax.set_title(ttl, fontsize=11); ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel("per-clone PCC" if i == 0 else ""); ax.set_ylim(-0.45, 1.0); tidy(ax)                        # one scale across C-E
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=cc, ec=cc, alpha=0.6, label=nm) for nm, cc in METHODS], loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=len(METHODS), frameon=False, fontsize=10)
    from spotr.figlayout import finish; finish(fig, PANELS)
    fig.canvas.draw(); pB, pC = axB.get_position(), (axC[-1] if axC else axB).get_position()   # the spot colours' legend, once, under B and C
    fig.legend(handles=hd, title="majority label", loc="upper center", bbox_to_anchor=((pB.x0 + pC.x1) / 2, pB.y0 - 0.005), ncol=len(hd), frameon=False, fontsize=9,
               title_fontsize=9, columnspacing=1.0, handletextpad=0.3)
    FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "fig_sim_benchmark.pdf", bbox_inches="tight", dpi=250); __import__('spotr.figlayout', fromlist=['compress_pdf']).compress_pdf(FIG / "fig_sim_benchmark.pdf"); fig.savefig(FIG / "fig_sim_benchmark.png", dpi=60, bbox_inches="tight"); plt.close(fig)
    supp_metrics(df)


def supp_metrics(df):
    """Supplementary: the four standard metrics (Li et al. 2022) pooled over every world-seed, one point per world-seed and method"""
    from spotr.figlayout import finish, compress_pdf
    P0 = len(PANELS); fig, axs = plt.subplots(1, 4, figsize=(20, 4.8)); fig.subplots_adjust(wspace=0.3)
    for i, (met, lab_) in enumerate([("PCC", "PCC"), ("SSIM", "SSIM"), ("RMSE", "RMSE"), ("JS", "JS divergence")]):
        ax = axs[i]; PANELS.append((ax, "ABCD"[i]))
        for x_, (nm, cc) in enumerate(METHODS):
            v = df.loc[df.method == nm, met].to_numpy()
            if len(v) == 0:
                ax.text(x_, 0.5, "pending", ha="center", va="center", fontsize=9, color="#9a9a9a", rotation=90, transform=ax.get_xaxis_transform()); continue
            bp = ax.boxplot([v], positions=[x_], widths=0.6, patch_artist=True, showfliers=False)
            for bx in bp["boxes"]:
                bx.set(facecolor=cc, alpha=0.35, edgecolor=cc)
            for k_ in ("whiskers", "caps", "medians"):
                for l_ in bp[k_]:
                    l_.set(color=cc, lw=1.5 if k_ == "medians" else 1)
            ax.scatter(x_ + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=10, color=cc, zorder=3)
        ax.set_xticks(range(len(METHODS)), [m for m, _ in METHODS], rotation=45, ha="right", fontsize=10); ax.set_title(lab_, fontsize=11); ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.grid(color=GRID, axis="y"); ax.set_axisbelow(True); [ax.spines[s_].set_visible(False) for s_ in ("top", "right")]
    finish(fig, PANELS[P0:])
    fig.savefig(FIG / "fig_sim_metrics.pdf", bbox_inches="tight"); compress_pdf(FIG / "fig_sim_metrics.pdf"); fig.savefig(FIG / "fig_sim_metrics.png", dpi=70, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
