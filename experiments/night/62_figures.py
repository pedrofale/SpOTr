"""Figures for the 2026-10-12 results page.  Reads only the result CSVs written by 52_-61_ (plus the cached
PEtracer object for the law curves and the per-clade comparison).  Each figure is rendered twice, for a
light and a dark page.  Output: <out>/<name>_{light,dark}.png and <out>/numbers.json.
Usage: python 62_figures.py <out_dir>
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
R = lambda f: pd.read_csv(HERE / f)
NUM = {}

THEMES = {
    "light": dict(surf="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
                  s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]),
    "dark": dict(surf="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34342f",
                 s=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]),
}


def style(th):
    t = THEMES[th]
    plt.rcParams.update({
        "figure.facecolor": t["surf"], "axes.facecolor": t["surf"], "savefig.facecolor": t["surf"],
        "axes.edgecolor": t["grid"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "text.color": t["ink"], "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlecolor": t["ink"], "legend.frameon": False, "lines.linewidth": 2,
        "font.family": "DejaVu Sans", "axes.axisbelow": True,
    })
    return t


def save(fig, name, th):
    fig.savefig(OUT / f"{name}_{th}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def loso_pick(df, key, metric, better="min"):
    """per held-out section, the w chosen on the other two sections by mean metric"""
    out = []
    for sec in df.sec.unique():
        o = df[df.sec != sec].groupby(key)[metric].mean()
        out.append(df[(df.sec == sec) & (df[key] == (o.idxmin() if better == "min" else o.idxmax()))])
    return pd.concat(out)


# ---------------------------------------------------------------- data
et = R("gonogo_etracer_density.csv"); em = R("etracer_map.csv")
fgw_et = {}
for tag, g in em.groupby("tag"):
    h = g[(g.arm == "FGW") & (g.w > 0)]; v = []
    for s in h.seed.unique():
        w = h[h.seed != s].groupby("w").r.mean().idxmax(); v.append(h[(h.seed == s) & (h.w == w)].r.iloc[0])
    fgw_et[tag] = float(np.mean(v))
NUM["etracer_fgw"] = fgw_et

pp = R("gonogo_petracer_pairwise.csv")
key = ["sec", "dens", "seed"]
mix = pp[pp.arm == "mixture (unary only)"].drop_duplicates(subset=key)
arms_pp = {"Neighbour vote": pp[pp.arm == "kNN vote (k LOO)"], "Density model, no tree": mix}
for arm, lab in [("POTTS pairwise", "+ pairwise, same/different clade"), ("TREE pairwise (divergence time)", "+ pairwise, divergence-time classes")]:
    h = pd.concat([pp[pp.arm == arm], mix.assign(arm=arm)])
    sel = []
    for dens in h.dens.unique():
        sel.append(loso_pick(h[h.dens == dens], "lam", "macro", "max"))
    arms_pp[lab] = pd.concat(sel)

dc = R("decomposition_check.csv"); hh = R("headtohead.csv").rename(columns={"section": "sec"})
dec = {"Expression only": dc[dc.arm == "expression only"]}
for a, lab in [("FGW", "FGW"), ("decomposed, ground truth", "Decomposed, weights from truth"),
               ("decomposed, law", "Decomposed, weights from law"), ("law, continuous", "Law, per pair")]:
    dec[lab] = loso_pick(dc[dc.arm == a], "w", "err")
gf = hh[(hh.arm == "learned rank 2") & (hh.w > 0) & hh.seed.isin([0, 1, 2])]
dec["Generalised FGW (reference)"] = loso_pick(gf, "w", "err")

la57 = R("loss_alignment.csv"); la58 = R("loss_alignment_heldout.csv"); la58b = R("loss_alignment_spatialprog.csv")
aa = R("anchor_alignment.csv")


def share(df, crit, sign=1, group=("sec", "seed")):
    e0s, ebs, eps = [], [], []
    for _, g in df.groupby(list(group)):
        g = g.dropna(subset=[crit])
        if g.empty:
            continue
        e0s.append(g[g.w == 0].err.iloc[0]); ebs.append(g.err.min()); eps.append(g.loc[(sign * g[crit]).idxmax(), "err"])
    e0s, ebs, eps = map(np.array, (e0s, ebs, eps))
    return float((e0s - eps).sum() / (e0s - ebs).sum())


crit_rows = [
    ("Niche, PEtracer's own modules (leaks positions)", share(la57.dropna(subset=["w"]), "niche", group=("sec",)), "leak"),
    ("Niche, states from scRNA-seq arm", share(la58[la58.variant == "all genes"], "niche"), "bio"),
    ("Niche, programs from spatial arm", share(la58b[la58b.variant == "all genes"], "niche"), "bio"),
    ("Growth: fitness vs edge distance", share(la58[la58.variant == "all genes"], "growth", -1), "bio"),
]
for cov in [0.5, 0.2, 0.1, 0.05, 0.02]:
    crit_rows.append((f"Anchor, {cov:.0%} of spots labelled", share(aa[aa["cov"] == cov], "xent", group=("sec", "seed", "draw")), "anchor"))
NUM["criteria"] = [(a, round(b, 3)) for a, b, _ in crit_rows]

af2, af3, af4 = R("anchor_fast2.csv"), R("anchor_fast3.csv"), R("anchor_fast4.csv")
FGW_REF = 410.6

# ---------------------------------------------------------------- law curves (cheap, from the cached object)
spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
from scipy.spatial.distance import pdist
law = {}
fits = {"M3-4": (160, 1.74), "M3-1": (300, 1.34), "M3-2": (305, 1.34)}
for sec in ["M3-4", "M3-1", "M3-2"]:
    S = m55.load(sec); AGE = S["AGE"]; n = len(AGE); rho = n / S["area"]
    iu = np.triu_indices(n, 1)
    Mr = m55.measured_r(AGE[iu], pdist(S["xy_sc"]))
    pack = np.array([np.sqrt(np.average((AGE <= a + 1e-9).sum(1) / (2 * np.pi * rho), weights=(AGE <= a + 1e-9).sum(1))) for a, _ in Mr])
    L, c = fits[sec]
    law[sec] = dict(age=Mr[:, 0], meas=Mr[:, 1], pack=pack, fit=np.sqrt(L * L + c * c * pack ** 2), L=L, c=c)

# per-clade learned vs measured (anchor_fast3: supervised and anchor arms)
spec = importlib.util.spec_from_file_location("m60", HERE / "60_anchor_learn.py")
m60 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m60)
pcl = []
for _, r in af3[af3.arm.isin(["pc-sup", "pc-free"])].iterrows():
    z = np.load(HERE / "anchor_fast" / f"{r.tag}.npz"); P = m60.prep(r.sec, int(r.seed))
    for k in range(P["K"]):
        idx = np.where(P["cl"] == k)[0]
        if len(idx) < 30:
            continue
        iu = np.triu_indices(len(idx), 1)
        d2 = ((P["xy_true"][idx][:, None] - P["xy_true"][idx][None]) ** 2).sum(-1)[iu]
        pk = P["NC"][np.ix_(idx, idx)][iu] / (2 * np.pi * P["rho"])
        pcl.append(dict(arm=r.arm, sec=r.sec, seed=r.seed, learned=float(z["Lk"][k]),
                        measured=float(np.sqrt(max(np.mean(d2 / 2 - float(z["c"]) ** 2 * pk), 1.0)))))
pcl = pd.DataFrame(pcl)

# ---------------------------------------------------------------- figures
for th in THEMES:
    t = style(th); s = t["s"]

    # 1. eTRACER: neighbour vote vs FGW across barcode density
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, tag, name in zip(axes, ["LE", "EE"], ["Late escape", "Early escape"]):
        g = et[et.tag == tag].groupby("frac").r.agg(["mean", "std"]).reset_index()
        ax.plot(g.frac * 100, g["mean"], "-o", color=s[0], ms=6, label="Neighbour vote among barcoded spots")
        ax.fill_between(g.frac * 100, g["mean"] - g["std"], g["mean"] + g["std"], color=s[0], alpha=0.15, lw=0)
        ax.axhline(fgw_et[tag], color=s[1], ls="--", label="FGW mapping (uses no barcodes)")
        ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20, 50]); ax.set_xticklabels(["2", "5", "10", "20", "50"])
        ax.axvspan(21 if tag == "LE" else 38, 55, color=t["grid"], alpha=0.5, lw=0)
        ax.text(23 if tag == "LE" else 39, 0.315, "real\ncoverage", color=t["ink2"], fontsize=9, va="bottom")
        ax.set_title(name); ax.set_xlabel("Spots carrying a clone barcode (%)")
    axes[0].set_ylabel("Clone-density agreement (r)"); axes[0].set_ylim(0.3, 0.92)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.06))
    save(fig, "etracer_vote", th)

    # 2. PEtracer in situ: macro accuracy and lineage error by reference density
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    dens = [0.45, 0.20, 0.05]
    for i, (lab, d) in enumerate(arms_pp.items()):
        m_ = d.groupby("dens").macro.mean().reindex(dens); e_ = d.groupby("dens").lin_err.mean().reindex(dens)
        axes[0].plot(range(3), m_.values, "-o", color=s[i], ms=6, label=lab)
        axes[1].plot(range(3), e_.values, "-o", color=s[i], ms=6, label=lab)
    for ax in axes:
        ax.set_xticks(range(3)); ax.set_xticklabels(["45%", "20%", "5%"]); ax.set_xlabel("Cells with a barcode (reference)")
    axes[0].set_ylabel("Macro clade accuracy (higher is better)"); axes[1].set_ylabel("Lineage error, days (lower is better)")
    axes[0].set_title("Clade accuracy"); axes[1].set_title("Lineage error")
    axes[0].legend(fontsize=8.5, loc="lower left")
    save(fig, "petracer_insitu", th)

    # 3. growth law vs measured spread
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, sec in zip(axes, ["M3-4", "M3-1", "M3-2"]):
        d = law[sec]
        ax.plot(d["age"], d["meas"], "o", color=s[0], ms=6, label="Measured spread r(a)")
        ax.plot(d["age"], d["pack"], "--", color=s[1], label="Packing only")
        ax.plot(d["age"], d["fit"], "-", color=s[2], label="Packing + mixing length")
        ax.set_yscale("log"); ax.set_xlabel("Divergence age a (days)")
        ax.set_title(f"{sec}   L = {d['L']} µm, c = {d['c']}")
    axes[0].set_ylabel("Spread of a clone (µm, log)"); axes[0].legend(fontsize=8.5, loc="lower right")
    save(fig, "growth_law", th)

    # 4. decomposition check: error by arm per section
    order = ["Expression only", "FGW", "Decomposed, weights from truth", "Decomposed, weights from law", "Law, per pair", "Generalised FGW (reference)"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True)
    for ax, sec in zip(axes, ["M3-1", "M3-2", "M3-4"]):
        vals = [dec[a][dec[a].sec == sec].err for a in order]
        y = np.arange(len(order))[::-1]
        for yi, v, a in zip(y, vals, order):
            col = s[1] if a.startswith("Law") or "law" in a else (t["ink2"] if a in ("Expression only", "Generalised FGW (reference)") else s[0])
            ax.scatter(v, np.full(len(v), yi) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), s=16, color=col, alpha=0.55, lw=0)
            ax.plot([v.mean()] * 2, [yi - 0.3, yi + 0.3], color=col, lw=3, solid_capstyle="round")
        ax.set_yticks(y); ax.set_yticklabels(order if sec == "M3-1" else [""] * len(order), fontsize=9)
        ax.set_title(sec); ax.set_xlabel("Median placement error (µm)")
        ax.axvline(dec["FGW"][dec["FGW"].sec == sec].err.mean(), color=s[0], ls=":", lw=1.2)
    save(fig, "decomposition", th)

    # 5. label-free criteria: share of the tree's gain recovered at each criterion's pick
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    labs = [r[0] for r in crit_rows][::-1]; vals = [r[1] for r in crit_rows][::-1]; kinds = [r[2] for r in crit_rows][::-1]
    cols = {"leak": t["ink2"], "bio": s[1], "anchor": s[0]}
    bars = ax.barh(range(len(labs)), [v * 100 for v in vals], color=[cols[k] for k in kinds], height=0.62)
    for b_, k in zip(bars, kinds):
        if k == "leak":
            b_.set_hatch("///"); b_.set_edgecolor(t["surf"]); b_.set_alpha(0.6)
    for i, v in enumerate(vals):
        ax.text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=9, color=t["ink"])
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=9); ax.set_xlim(0, 108)
    ax.set_xlabel("Share of the tree's gain over expression-only recovered at the criterion's pick (%)")
    ax.axvline(100, color=t["ink2"], lw=1, ls=":")
    save(fig, "criteria", th)

    # 6. one weight path, M3-4 seed 0: error and criteria (standardised)
    g58 = la58[(la58.sec == "M3-4") & (la58.seed == 0) & (la58.variant == "all genes")].sort_values("w")
    ga = aa[(aa.sec == "M3-4") & (aa.seed == 0) & (aa["cov"] == 0.2)].groupby("w").xent.mean().reindex(g58.w.values)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.1]))
    x = g58.w.replace(0, 0.01).values
    axes[0].plot(x, g58.err, "-o", color=t["ink"], ms=5); axes[0].set_ylabel("Placement error (µm)")
    best = x[np.argmin(g58.err.values)]
    for ax in axes:
        ax.axvline(best, color=t["ink2"], ls=":", lw=1.2)
    z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
    axes[1].plot(x, z(ga.values), "-o", color=s[0], ms=5, label="Anchor, 20% of spots")
    axes[1].plot(x, z(g58.niche.values), "-o", color=s[1], ms=5, label="Niche (scRNA-seq states)")
    axes[1].plot(x, z(-g58.growth.values), "-o", color=s[2], ms=5, label="Growth")
    axes[1].set_ylabel("Criterion (standardised,\nhigher = preferred)"); axes[1].set_xscale("log")
    axes[1].set_xlabel("Tree weight w (0 plotted at 0.01)"); axes[1].legend(fontsize=8.5, loc="lower left")
    axes[0].set_title("M3-4, seed 0: dotted line marks the accuracy optimum")
    save(fig, "path_example", th)

    # 7. anchor coverage sweep
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    covs = [0.02, 0.05, 0.1, 0.2, 0.5]
    for i, (crit, lab) in enumerate([("xent", "Multinomial composition likelihood"), ("dens", "Clade-density correlation")]):
        ax.plot([c * 100 for c in covs], [share(aa[aa["cov"] == c], crit, group=("sec", "seed", "draw")) * 100 for c in covs], "-o", color=s[i], ms=6, label=lab)
    ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20, 50]); ax.set_xticklabels(["2", "5", "10", "20", "50"])
    ax.set_xlabel("Spots with a known clade composition (%)"); ax.set_ylabel("Share of tree's gain at pick (%)")
    ax.set_ylim(40, 100); ax.legend(fontsize=9, loc="lower right")
    save(fig, "anchor_coverage", th)

    # 8. learning results
    rows = [("Anchor 5%, weight only", af2[(af2["cov"] == 0.05) & (af2.arm == "a-only")].err_exact),
            ("Anchor 5%, + per-clade L", af2[(af2["cov"] == 0.05) & (af2.arm == "pc-law")].err_exact),
            ("Anchor 5%, random groups", af2[(af2["cov"] == 0.05) & (af2.arm == "rand-law")].err_exact),
            ("Anchor 20%, weight only", af2[(af2["cov"] == 0.2) & (af2.arm == "a-only")].err_exact),
            ("Anchor 20%, + per-clade L", af2[(af2["cov"] == 0.2) & (af2.arm == "pc-law")].err_exact),
            ("Anchor 20%, random groups", af2[(af2["cov"] == 0.2) & (af2.arm == "rand-law")].err_exact),
            ("Supervised, weight only", af4[af4.arm == "a-sup"].err_exact_heldout),
            ("Supervised, + per-clade L", af4[af4.arm == "pc-sup"].err_exact_heldout),
            ("Supervised, two-stage per-clade", af4[af4.arm == "pc-sup2"].err_exact_heldout)]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v) in zip(y, rows):
        col = s[1] if ("per-clade" in lab or "random" in lab) else s[0]
        ax.scatter(v.clip(upper=620), np.full(len(v), yi) + np.random.default_rng(1).uniform(-0.12, 0.12, len(v)), s=18, color=col, alpha=0.6, lw=0)
        ax.plot([v.mean()] * 2, [yi - 0.3, yi + 0.3], color=col, lw=3, solid_capstyle="round")
    ax.axvline(FGW_REF, color=t["ink2"], ls="--", lw=1.2)
    ax.text(FGW_REF + 3, y[0] + 0.45, "FGW, honestly chosen weight (411 µm)", color=t["ink2"], fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Median placement error, exact solver (µm)")
    save(fig, "learning", th)

    # 9. learned vs measured per-clade mixing length
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)
    for ax, (arm, lab, col) in zip(axes, [("pc-sup", "Supervised (true positions of 30% of cells)", s[0]),
                                          ("pc-free", "Anchor, 50% of spots", s[1])]):
        d = pcl[pcl.arm == arm]
        rhos = [spearmanr(g.learned, g.measured)[0] for _, g in d.groupby(["sec", "seed"]) if g.learned.std() > 0]
        ax.scatter(d.measured, d.learned, s=20, color=col, alpha=0.7, lw=0)
        ax.set_xlabel("Mixing length measured from true positions (µm)"); ax.set_title(f"{lab}\nmean within-split Spearman {np.nanmean(rhos):+.2f}")
    axes[0].set_ylabel("Mixing length learned (µm)")
    save(fig, "perclade_L", th)

    # 10. entropic vs exact evaluation of the same learned parameters
    allr = pd.concat([af2.assign(e=af2.err_entropic, x=af2.err_exact), af4.assign(e=af4.err_entropic_heldout, x=af4.err_exact_heldout)])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(allr.x, allr.e, s=18, color=s[0], alpha=0.6, lw=0)
    lo, hi = 300, 720
    ax.plot([lo, hi], [lo, hi], color=t["ink2"], lw=1, ls=":")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Error, exact solver (µm)"); ax.set_ylabel("Error, entropic solver (µm)")
    ax.set_title("Same parameters, two solvers")
    save(fig, "solver_noise", th)

NUM["perclade_rho"] = {a: float(np.nanmean([spearmanr(g.learned, g.measured)[0] for _, g in pcl[pcl.arm == a].groupby(["sec", "seed"]) if g.learned.std() > 0])) for a in ["pc-sup", "pc-free"]}
NUM["dec_means"] = {a: round(float(v.err.mean()), 1) for a, v in dec.items()}
json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1)
print(json.dumps(NUM, indent=1))
