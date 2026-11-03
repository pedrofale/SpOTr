"""One simulated tumour, end to end: the data, the true per-clade parameters, the map, the learning curves and the
inferred against the true parameters.  The standard example figure for a simulation (see the vault's
decision-simulation-example-figure), 2026-10-13.

Data: 84_growth_sim.simulate(3, m_hi=0.3) -- 3,000 cells, surface growth, half of the founder lineages making runs
at 0.3 / cycle and the rest at 0.02; half the cells dissociated, half pooled into ~300 spots.  Learner:
70_spotr.learn with the growth loss (anchor-free), one L per clone and the tree weight learned; the law (L0, c0) is
the reference tumour's, as in 85_growth_sweep.py.  The true L of a clone is measured on its cells' true positions
with the law's slope c0 (70_spotr.measured_L).
Usage: python 89_growth_example_figure.py [--outer growth|anchor] [--seed 3] [--mhi 0.3] [--mig 0] [--pos 0] [--tag name]
       (writes SpOTr-overleaf/figures/growth_example_<outer>[_<tag>].pdf and a PNG beside it)
       --mig R: growth with every lineage static, then half the major clones make R runs per cell (90_contrast_oracle);
       --pos w: positional programs of strength w in the expression.
       --adv a --fmig R: one expanding clade (divides a times faster) that alone makes R runs per cell after growth
       (92_expanding_clade); it is marked as clade A.
       --prior sd: the learner's prior on log L (0.5 so far); --groups two-level: one L for the expanding clade A and
       one for the rest (94_prior_learn).  --niche R: every cell in a disc of 30% of the cells makes R runs confined
       to it (95_niche_oracle); the disc is outlined and the oracle row adds the map with the true L in and out of it.
       --oracle: a third row with the maps under plain FGW, the best single within-clade L and each clone's true L,
       each at the tree weight (and single L) that gives the best clone map -- what one global penalty does.
"""
import argparse, json, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "cm", "pdf.fonttype": 42,
                     "axes.unicode_minus": False, "font.size": 9.5, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8,
                     "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a", "axes.labelcolor": "#222222"})
INK, INK2, GRID = "#111111", "#4a4a4a", "#e6e5e0"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # reference order
OTHER, MINOR = "#8a8984", "#d6d5cf"
LOSS_NAME = {"growth": ("growth loss", "r(fitness, mapped edge distance)"), "anchor": ("anchor loss", "anchor NLL per cell")}


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)


def umap2(A, seed=0):
    from sklearn.decomposition import PCA
    import umap
    Z = PCA(n_components=min(30, A.shape[1] - 1), random_state=seed).fit_transform(A - A.mean(0))
    return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed).fit_transform(Z)


NICHE = None                                                  # (centre, radius) in um, outlined on every slide


def baselines(X, seed=0):
    """the expression-only maps every example figure shows as baselines (Pedro, 2026-10-19): exact OT on the expression
    cost alone, and Tangram (73_tangram.py's settings: cells mode, density term with the same spot marginal), both as
    couplings with rows summing to 1/n.  Returns {name: coupling}."""
    import ot
    spec = importlib.util.spec_from_file_location("m73", Path(__file__).resolve().parent / "73_tangram.py")
    m73 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m73)
    n = len(X["cl"]); p = np.ones(n) / n
    Tt = m73.tangram(X["Xsc"], X["Xst"], X["q"], seed); Tt = Tt / np.maximum(Tt.sum(1, keepdims=True), 1e-300) / n
    return {"Expression only (OT)": ot.emd(p, X["q"], X["M"]), "Tangram": Tt}


def oracle(X):
    """the truth a map is compared with (Pedro, 2026-10-21: "The 'truth' should be the true positions of the cells we use for the
    method - not of more cells than that"): every mapped (dissociated) cell at the spot nearest its true position, as a coupling
    with rows 1/n.  Colour it with the same rule as the maps.  (Figures before 2026-10-21 drew the truth from the spots' own cells.)"""
    from scipy.spatial import cKDTree
    n = len(X["xy_true"]); T = np.zeros((n, len(X["spots"])))
    T[np.arange(n), cKDTree(X["spots"]).query(X["xy_true"])[1]] = 1.0 / n
    return T

def slide(ax, spots, pitch, colors, title):
    if NICHE is not None:
        from matplotlib.patches import Circle
        ax.add_patch(Circle(NICHE[0], NICHE[1], fill=False, ls="--", lw=1.4, color=INK, zorder=5))
    ax.add_collection(PatchCollection([Rectangle(s - pitch / 2, pitch, pitch) for s in spots], facecolor=colors,
                                      edgecolor="white", linewidth=0.3))
    lo, hi = spots.min(0) - pitch, spots.max(0) + pitch
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal"); ax.set_anchor("N"); ax.axis("off")
    ax.plot([hi[0] - pitch - 200, hi[0] - pitch], [lo[1] + 0.4 * pitch] * 2, color=INK, lw=2, solid_capstyle="butt")
    ax.text(hi[0] - pitch - 100, lo[1] + 0.9 * pitch, "200 µm", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def oracle_row(fig, cell, X, m70, ccol, L0, c0):
    """maps under one global penalty against each clone's true L, each at its best tree weight by the clone map"""
    import ot
    m92 = mod("m92", "92_expanding_clade.py")
    n, cl, Km = len(X["cl"]), X["cl"], X["Kmaj"]; A = np.asarray(X.get("fast_clades", []), int)
    p = np.ones(n) / n; D = X["D"]; same = cl[:, None] == cl[None, :]
    meas = m70.measured_L(X, cl, c0); Lt = np.array([meas.get(k, np.nan) for k in range(Km)])
    W0, WP0 = m70.moments_np(X, np.full((n, n), L0), c0); sc = m70.scale(X, W0, WP0)
    Wf = np.ones((n, n)); np.fill_diagonal(Wf, 0.0); WPf = Wf * 2.0 * X["AGE"]; scf = m70.scale(X, Wf, WPf)
    law = lambda Lmat: tuple(M * sc for M in m70.moments_np(X, Lmat, c0))
    models = {"plain FGW (tree distance)": [(Wf * scf, WPf * scf)],
              "growth law, one L for all clones": [law(np.where(same, s * L0, L0)) for s in m92.SCAN],
              "growth law, each clone's true L": [law(np.where(same, np.r_[np.where(np.isfinite(Lt), Lt, L0), L0][cl][:, None], L0))]}
    if X.get("spot_in_niche") is not None:                    # niche mode: the niche model replaces plain FGW
        m95 = mod("m95", "95_niche_oracle.py"); Lin, Lout = m95.niche_L(X, c0)
        z = X["spot_in_niche"].astype(float); masks = [np.outer(z, z), np.outer(1 - z, 1 - z), np.outer(z, 1 - z) + np.outer(1 - z, z)]
        blocks = []
        for Lb, Mk in zip([Lin, Lout, np.sqrt((Lin ** 2 + Lout ** 2) / 2)], masks):
            W, WP = law(np.where(same, Lb, L0)); blocks.append((W, WP, WP * WP / np.where(W > 0, W, 1.0), Mk, D * Mk, D ** 2 * Mk))
        del models["plain FGW (tree distance)"]
        models[f"growth law, true L in / out of the niche ({Lin:.0f} / {Lout:.0f} µm)"] = [("blocks", blocks)]
    sub = cell.subgridspec(1, 3, wspace=0.12)
    for j, (name, Ws) in enumerate(models.items()):
        best = None
        for W, WP in Ws:
            T = None
            for a in m92.ALPHAS:
                T = (m95.solve_blocks(X["M"], WP, p, X["q"], a, T0=T) if isinstance(W, str)
                     else m70.solve_exact(X["M"], W, WP, D, D ** 2, p, X["q"], a, T0=T))
                r = m92.readouts(T, X, A)
                if best is None or r["r"] > best[0]["r"]:
                    best = (r, T.copy())
        r, T = best; inferred = (T.T @ np.eye(X["K"])[cl]).argmax(1)
        slide(fig.add_subplot(sub[j]), X["spots"], X["pitch"], [ccol[k] for k in inferred],
              f"{name}\nclone map r {r['r']:.2f}" + (f" (localized clones {r['r_loc']:.2f})" if len(A) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", default="growth", choices=list(LOSS_NAME)); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--mhi", type=float, default=0.3); ap.add_argument("--mig", type=float, default=0.0)
    ap.add_argument("--pos", type=float, default=0.0); ap.add_argument("--tag", default="")
    ap.add_argument("--adv", type=float, default=0.0); ap.add_argument("--fmig", type=float, default=0.0)
    ap.add_argument("--oracle", action="store_true"); ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--groups", default="per-clade", choices=["per-clade", "two-level"])
    ap.add_argument("--niche", type=float, default=0.0)
    args = ap.parse_args()
    m70 = mod("m70", "70_spotr.py"); m84 = mod("m84", "84_growth_sim.py")
    law = json.load(open(HERE / "spotr_out" / "growth_sweep.json")); L0, c0 = law["L0"], law["c0"]
    global NICHE
    if args.niche > 0:
        sim = m84.simulate(args.seed, m_hi=0.02, niche_mig=args.niche, pos_w=args.pos); X = m84.problem(sim)
        X["inB"] = np.zeros(X["K"], bool); mot_lab = None; NICHE = sim["niche"]
    elif args.adv > 0:
        X = m84.problem(m84.simulate(args.seed, m_hi=0.02, fast=1, adv=args.adv, fast_mig=args.fmig, pos_w=args.pos))
        X["inB"] = np.isin(np.arange(X["K"]), X["fast_clades"])
        mot_lab = (f"expanding ({args.adv:g}x division), {args.fmig:g} runs / cell", "localized")
    elif args.mig > 0:
        X = m84.problem(m84.simulate(args.seed, m_hi=0.02, mig=args.mig, pos_w=args.pos))
        mot_lab = (f"motile ({args.mig:g} runs / cell)", "static")
    else:
        X = m84.problem(m84.simulate(args.seed, m_hi=args.mhi, pos_w=args.pos))
        mot_lab = None
    X["L0"], X["c0"] = L0, c0
    cl, Km, K = X["cl"], X["Kmaj"], X["K"]
    # clones numbered by size; the 8 largest take the reference hues, the rest grey
    size = np.bincount(cl, minlength=K)[:Km]; order = np.argsort(-size); rank = np.empty(Km, int); rank[order] = np.arange(Km)
    ccol = [SLOTS[rank[k]] if rank[k] < len(SLOTS) else OTHER for k in range(Km)] + [MINOR]
    label = {k: str(rank[k] + 1) for k in range(Km)}
    meas = m70.measured_L(X, cl, c0); Ltrue = np.array([meas.get(k, np.nan) for k in range(Km)])
    motile = np.asarray(X["inB"], bool)[:Km]; rate = X["motility"][:Km]

    groups = cl.copy() if args.groups == "per-clade" else np.where(np.isin(cl, X["fast_clades"]), 0, 1)
    log, best = m70.learn(X, groups, learn_a=True, a0=0.165, outer=args.outer, prior_sd=args.prior)
    gidx = np.array([groups[cl == k][0] for k in range(Km)])
    steps = np.array([r["step"] for r in log]); Lpath = np.array([r["L"] for r in log])[:, gidx]
    Lhat = np.array(best["L"])[gidx]
    Y = np.eye(K)[cl]; inferred = (best["T"].T @ Y).argmax(1); true_dom = X["Nspot"].argmax(1)
    agree = float((inferred == true_dom).mean())

    fig = plt.figure(figsize=(15.5, 12.0 if args.oracle else 8.4))
    gs = fig.add_gridspec(3 if args.oracle else 2, 4, hspace=0.38, wspace=0.3, height_ratios=[1, 1, 1.1] if args.oracle else None)
    if args.oracle:
        oracle_row(fig, gs[2, :], X, m70, ccol, L0, c0)
    # a: single-cell expression
    ax = fig.add_subplot(gs[0, 0]); U = umap2(X["Xsc"])
    o = np.argsort(cl == Km)[::-1]                                          # small lineages drawn first, underneath
    ax.scatter(U[o, 0], U[o, 1], s=4, c=[ccol[k] for k in cl[o]], lw=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Single-cell expression, {len(cl):,} dissociated cells", fontsize=10, color=INK, loc="left")
    # b: spot expression
    ax = fig.add_subplot(gs[0, 1]); V = umap2(X["Xst"])
    ax.scatter(V[:, 0], V[:, 1], s=14, c=[ccol[k] for k in true_dom], lw=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Spot expression, {len(true_dom)} spots (dominant clone)", fontsize=10, color=INK, loc="left")
    # c: slide, true clones
    ax = fig.add_subplot(gs[0, 2])
    slide(ax, X["spots"], X["pitch"], [ccol[k] for k in true_dom], f"Slide: true dominant clone per spot ({X['pitch']:.0f} µm spots)")
    # d: true parameters
    ax = fig.add_subplot(gs[0, 3]); style(ax)
    ys = rank + 1
    for k in range(Km):
        ax.scatter(Ltrue[k], ys[k], s=60, color=ccol[k] if motile[k] else "white", edgecolor=ccol[k], lw=1.6, zorder=3)
    ax.axvline(L0, color=INK2, lw=0.9, ls="--"); ax.text(L0, -0.15, f" law's shared L₀ = {L0:.0f} µm ", fontsize=8, color=INK2, va="center",
                                                         ha="right" if L0 > np.nanmean(Ltrue) else "left")
    ax.set_yticks(np.arange(1, Km + 1), [f"clone {i}" for i in range(1, Km + 1)]); ax.set_ylim(Km + 0.7, -0.6)
    ax.set_xlabel("true mixing length L (µm, from true positions)")
    ax.set_title("True parameters per clone", fontsize=10, color=INK, loc="left")
    key = [Line2D([], [], marker="o", ls="", color=INK2, markersize=7, label=mot_lab[0] if mot_lab else f"motile ({rate.max():g} runs / cycle)"),
           Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, markeredgewidth=1.4,
                  markersize=7, label=mot_lab[1] if mot_lab else f"static ({rate.min():g} runs / cycle)")]
    if NICHE is not None:
        key = [Line2D([], [], ls="--", color=INK, label="the niche: every cell inside mixes, whatever its clone")]
    # e: slide, inferred clones
    ax = fig.add_subplot(gs[1, 0])
    slide(ax, X["spots"], X["pitch"], [ccol[k] for k in inferred],
          f"Map: inferred clone ({100 * agree:.0f}% of spots right)")
    # f: loss curves, with the tree weight below
    sub = gs[1, 1].subgridspec(2, 1, hspace=0.12, height_ratios=[1.5, 1])
    ax = fig.add_subplot(sub[0]); style(ax)
    ax.plot(steps, [r["nll_fit"] for r in log], color="#184f95", lw=1.8, label="fit half")
    ax.plot(steps, [r["nll_val"] for r in log], color="#eb6834", lw=1.8, label="validation half")
    ax.axvline(best["step"], color=INK2, lw=0.8, ls=":")
    ax.set_title(f"Learning: the {LOSS_NAME[args.outer][0]}", fontsize=10, color=INK, loc="left")
    ax.set_ylabel(f"{LOSS_NAME[args.outer][1]}\n(lower is better)", fontsize=8.5); ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = fig.add_subplot(sub[1], sharex=ax); style(ax)
    ax.plot(steps, [r["a"] for r in log], color=INK, lw=1.8)
    ax.axvline(best["step"], color=INK2, lw=0.8, ls=":")
    late = best["step"] > 0.7 * steps[-1]
    ax.text(best["step"], ax.get_ylim()[1], "chosen step " if late else " chosen step", fontsize=8, color=INK2, va="top",
            ha="right" if late else "left")
    ax.set_ylabel("tree weight α"); ax.set_xlabel("step")
    # g: L per clone along the learning
    ax = fig.add_subplot(gs[1, 2]); style(ax)
    for k in order:
        ax.plot(steps, Lpath[:, k], color=ccol[k], lw=1.6)
        ax.text(steps[-1] + 0.6, Lpath[-1, k], label[k], color=ccol[k], fontsize=8, va="center", fontweight="bold")
    ax.axvline(best["step"], color=INK2, lw=0.8, ls=":")
    ax.set_xlim(steps[0], steps[-1] + 3); ax.set_xlabel("step"); ax.set_ylabel("learned L (µm)")
    ax.set_title("Learning: each clone's L", fontsize=10, color=INK, loc="left")
    # h: true against inferred
    ax = fig.add_subplot(gs[1, 3]); style(ax)
    ok = np.isfinite(Ltrue); lim = [0.8 * min(Ltrue[ok].min(), Lhat.min()), 1.15 * max(Ltrue[ok].max(), Lhat.max())]
    xt = lim[0] + 0.9 * (lim[1] - lim[0]); ax.plot(lim, lim, color=INK2, lw=0.8); ax.text(xt, xt, "y = x ", ha="right", va="bottom", fontsize=8, color=INK2)
    for k in np.where(ok)[0]:
        ax.scatter(Ltrue[k], Lhat[k], s=60, color=ccol[k] if motile[k] else "white", edgecolor=ccol[k], lw=1.6, zorder=3)
        ax.annotate(label[k], (Ltrue[k], Lhat[k]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("true L (µm)"); ax.set_ylabel("learned L (µm)")
    ax.set_title(f"True against learned L (α learned: {best['a']:.2f})", fontsize=10, color=INK, loc="left")
    # clone key
    handles = [Line2D([], [], marker="s", ls="", color=ccol[k], markersize=8, label=f"clone {label[k]}") for k in order]
    handles.append(Line2D([], [], marker="s", ls="", color=MINOR, markersize=8, label="small lineages"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.legend(handles=key, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.055))
    out = HERE.parents[2] / "SpOTr-overleaf" / "figures" / f"growth_example_{args.outer}{'_' + args.tag if args.tag else ''}.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}; agreement {agree:.2f}; alpha {best['a']:.3f} at step {best['step']}; err {best['err']:.0f} um")
    print("clone, motile, true L, learned L:", [(label[k], bool(motile[k]), round(float(Ltrue[k])), round(float(Lhat[k]))) for k in order])


if __name__ == "__main__":
    main()
