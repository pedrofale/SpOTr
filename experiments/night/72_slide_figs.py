"""Figures for the four-slide summary (2026-10-13).  Light theme only (the deck commits to one look).
Usage: python 72_slide_figs.py <out_dir> <compare.csv dir> [real]
  without 'real': method and simulation figures; with 'real': PEtracer and eTRACER figures from spotr_out/.
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
CMP = Path(sys.argv[2])
INK, INK2, GRID, SURF = "#1d1d1b", "#5b5a55", "#e2e1dc", "#fbfaf7"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
PURPLE = "#6250d6"
SELECT_FROM = ["path-law", "per-clade", "per-clade-mom"]   # the candidates the anchor chooses between
ORACLE_ARM = "path-oracle-rel"          # the ceiling row: true per-clade spreads at the level of L0 (70_spotr.py)
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
                     "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
                     "font.size": 17, "axes.labelsize": 17, "axes.titlesize": 18, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 15, "lines.linewidth": 2.6,
                     "xtick.labelsize": 16, "ytick.labelsize": 16,
                     "axes.axisbelow": True})
NUM = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def method_and_simulation():
    # growth law on M3-4 (measured spread vs law), from the cached data as in 62_
    spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
    m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
    from scipy.spatial.distance import pdist
    S = m55.load("M3-4"); AGE = S["AGE"]; n = len(AGE); rho = n / S["area"]; iu = np.triu_indices(n, 1)
    Mr = m55.measured_r(AGE[iu], pdist(S["xy_sc"]))
    pack = np.array([np.sqrt(np.average((AGE <= a + 1e-9).sum(1) / (2 * np.pi * rho), weights=(AGE <= a + 1e-9).sum(1))) for a, _ in Mr])
    L, c = 160, 1.74
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(Mr[:, 0], Mr[:, 1], "o", color=BLUE, ms=8, label="measured, PEtracer M3-4")
    ax.plot(Mr[:, 0], pack, "--", color=GREY, label="clone packing only")
    ax.plot(Mr[:, 0], np.sqrt(L * L + c * c * pack ** 2), "-", color=ORANGE, label="packing + mixing length")
    ax.set_yscale("log"); ax.set_xlabel("Divergence age a (days)"); ax.set_ylabel("Spread of the clone, r(a) (µm)")
    ax.legend(loc="lower right")
    save(fig, "s2_growth_law")

    # identifiability on the planted control: law vs the paper's per-clade alpha / theta (same anchor)
    R = pd.read_csv(CMP / "compare.csv")
    S_ = R.groupby(["arm", "sigma"]).recovered.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for arm, col, lab in [("Law: mixing length L", BLUE, "SpOTr: per-clade mixing length L"),
                          ("Paper: tree weight α", ORANGE, "paper: per-clade tree weight α"),
                          ("Paper: within-clade weight θ", GREY, "paper: per-clade within-clade weight θ")]:
        ax.plot(S_.index, 100 * S_[arm], "-o", color=col, ms=9, label=lab)
    ax.text(500, 26, "ties count as not found\n20 draws per point", color=INK2, fontsize=15, ha="right")
    ax.set_xticks([0, 300, 500]); ax.set_ylim(0, 105)
    ax.set_xlabel("Planted extra dispersal of half the clades, σ (µm)"); ax.set_ylabel("Dispersed group found (%)")
    ax.legend(loc="upper left", fontsize=12.5)
    save(fig, "s3_identifiability")
    NUM["identifiability"] = (100 * S_).round(0).to_dict()

    # per-clade gradient learning on the planted control: AUC vs sigma, and learned vs measured at sigma = 500
    res = json.load(open(HERE / "spotr_out" / "plantedK.json"))
    rows, pts = [], []
    for r in res:
        Lk = np.array(r["L"]); inB = np.array(r["inB"], bool); ok = np.array(r["sizes"]) >= 30
        auc = mannwhitneyu(np.log(Lk[inB & ok]), np.log(Lk[~inB & ok]), alternative="greater").statistic / ((inB & ok).sum() * (~inB & ok).sum())
        meas = np.array([r["meas"].get(str(k), np.nan) for k in range(len(Lk))])
        rows.append(dict(sigma=r["sigma"], gdraw=r["gdraw"], auc=auc, rho=spearmanr(Lk[ok], meas[ok])[0]))
        if r["sigma"] == 500:
            for k in np.where(ok)[0]:
                pts.append(dict(meas=meas[k], learned=Lk[k], dispersed=bool(inB[k]), gdraw=r["gdraw"]))
    A = pd.DataFrame(rows); P = pd.DataFrame(pts); NUM["plantedK"] = A.round(3).to_dict(orient="records")
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for disp, col, lab in [(False, BLUE, "undisturbed clades"), (True, ORANGE, "clades given extra dispersal")]:
        d = P[P.dispersed == disp]
        ax.scatter(d.meas, d.learned, s=90, color=col, alpha=0.85, lw=0, label=lab)
    ax.set_xlabel("Measured mixing length (µm)"); ax.set_ylabel("Learned from anchor (µm)")
    a5 = A[A.sigma == 500].auc.values; a3 = A[A.sigma == 300].auc.values; a0 = A[A.sigma == 0].auc.values
    ax.set_title(f"σ = 500 µm, both group draws", fontsize=17)
    ax.legend(loc="upper left")
    save(fig, "s3_perclade")
    NUM["auc"] = dict(s0=a0.tolist(), s300=a3.tolist(), s500=a5.tolist())


def real():
    res = []
    for f in ["real.json", "real_more.json", "real_controls.json", "real_oracle.json", "real_mom.json", "real_s47.json"]:
        if (HERE / "spotr_out" / f).exists():
            res += json.load(open(HERE / "spotr_out" / f))
    rows = []
    for r in res:
        if r["arm"].startswith("path-"):
            P = pd.DataFrame(r["path"])
            pick = P.loc[P.nll_val.idxmin()]
            rows.append(dict(sec=r["sec"], seed=r["seed"], arm=r["arm"], err=pick.err, agree=pick.agree, w=pick.w,
                             nll_val=pick.nll_val))
            if r["arm"] == "path-fgw":
                e0 = P[P.w == 0].iloc[0]
                rows.append(dict(sec=r["sec"], seed=r["seed"], arm="expression", err=e0.err, agree=e0.agree, w=0.0))
        else:
            nv = r["log"][r["best_step"]]["nll_val"] if r.get("log") else np.nan
            rows.append(dict(sec=r["sec"], seed=r["seed"], arm=r["arm"], err=r["err"], agree=r["agree"], w=np.nan, nll_val=nv))
    for f in ["tangram.json", "tangram_s47.json"]:                  # 73_tangram.py, same splits and metrics
        if (HERE / "spotr_out" / f).exists():
            rows += [dict(sec=r["sec"], seed=r["seed"], arm="tangram", err=r["err"], agree=r["agree"], w=np.nan)
                     for r in json.load(open(HERE / "spotr_out" / f)) if r["data"] == "PEtracer"]
    D = pd.DataFrame(rows)
    # SpOTr as run: per split, the candidate with the lowest validation-anchor NLL (no true position involved).
    # Rule fixed after seeds 0-3 were seen; seeds 4-7 are its confirmation.
    cand = D[D.arm.isin(SELECT_FROM)]
    sel = cand.loc[cand.groupby(["sec", "seed"]).nll_val.idxmin()].copy()
    sel["picked"] = sel.arm; sel["arm"] = "selected"
    D = pd.concat([D, sel], ignore_index=True)
    D["half"] = np.where(D.seed < 4, "seeds 0-3 (rule designed)", "seeds 4-7 (rule confirmed)")
    D.to_csv(OUT / "real_summary.csv", index=False)
    NUM["selection"] = {h: g.groupby("arm").err.mean().round(1).to_dict() for h, g in D.groupby("half")}
    NUM["selection_picks"] = sel.picked.value_counts().to_dict()
    order = [("expression", "Expression only", GREY), ("tangram", "Tangram", PURPLE), ("path-fgw", "FGW, square loss", ORANGE),
             ("path-law", "SpOTr, shared L", AQUA), ("per-clade", "SpOTr, per-clade L", "#184f95"),
             (ORACLE_ARM, "ceiling: true per-clade L", GREY),
             ("per-random", "control: random groups", GREY), ("path-law-shuffled", "control: shuffled tree", GREY)]
    order = [o for o in order if (D.arm == o[0]).any()]
    plt.rcParams.update({"font.size": 21, "axes.labelsize": 21, "xtick.labelsize": 20, "ytick.labelsize": 21})
    fig, ax = plt.subplots(figsize=(5.4, 6.6))
    y = np.arange(len(order))[::-1]
    for yi, (arm, lab, col) in zip(y, order):
        v = D[D.arm == arm].err.values
        side = lab.startswith(("control", "ceiling"))
        mk = "o" if not side else "D"
        ax.scatter(v, yi + np.random.default_rng(0).uniform(-0.14, 0.14, len(v)), s=70, color=col, alpha=0.7, lw=0, marker=mk)
        ax.plot([v.mean()] * 2, [yi - 0.33, yi + 0.33], color=col if not side else INK2, lw=5, solid_capstyle="round")
        ax.text(v.max() + 14, yi, f"{v.mean():.0f}", ha="left", color=INK, fontsize=19, va="center")
    ax.set_yticks(y); ax.set_yticklabels([o[1] for o in order])
    nspl = D[D.arm == "selected"].shape[0]
    ax.set_xlabel("Placement error (µm)")
    ax.set_xticks([400, 600, 800]); ax.set_xlim(right=ax.get_xlim()[1] + 60)
    save(fig, "s4_petracer")
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 16})
    NUM["petracer"] = {"err": D.groupby("arm").err.mean().round(1).to_dict(), "agree": D.groupby("arm").agree.mean().round(3).to_dict(),
                       "n_splits": int(nspl)}
    fg = D[D.arm == "path-fgw"].set_index(["sec", "seed"]).err; lw_ = D[D.arm == "path-law"].set_index(["sec", "seed"]).err
    for arm in ["per-clade", "per-clade-mom", "selected", "tangram"]:
        v = D[D.arm == arm].set_index(["sec", "seed"]).err
        NUM["petracer"][f"{arm}_beats_fgw"] = f"{int((v < fg.reindex(v.index)).sum())}/{len(v)}"
        NUM["petracer"][f"{arm}_beats_shared"] = f"{int((v < lw_.reindex(v.index)).sum())}/{len(v)}"
    # per-clade learned vs measured, centred within each split
    rr, pts = [], []
    for r in res:
        if r["arm"] not in ("per-clade", "per-clade-mom"):
            continue
        Lk = np.array(r["L"]); ok = np.array(r["sizes"]) >= 30
        meas = np.array([r["meas"].get(str(k), np.nan) for k in range(len(Lk))]); ok &= np.isfinite(meas)
        rho = spearmanr(Lk[ok], meas[ok])[0]
        null = np.array([spearmanr(np.random.default_rng(i).permutation(Lk[ok]), meas[ok])[0] for i in range(2000)])
        row = dict(arm=r["arm"], sec=r["sec"], seed=r["seed"], rho=float(rho), p=float(np.mean(null >= rho)), n=int(ok.sum()))
        if r.get("L_init"):                                   # the anchor-moment start alone, before any learning
            row["rho_start"] = float(spearmanr(np.array(r["L_init"])[ok], meas[ok])[0])
        rr.append(row)
        if r["arm"] == "per-clade-mom":
            lx, ly = np.log(meas[ok]), np.log(Lk[ok])
            for a_, b_ in zip(lx - lx.mean(), ly - ly.mean()):
                pts.append(dict(sec=r["sec"], x=a_, y=b_))
    NUM["real_perclade"] = rr
    Rr = pd.DataFrame(rr)
    NUM["real_perclade_summary"] = {a: dict(mean_rho=round(g.rho.mean(), 3), positive=f"{int((g.rho > 0).sum())}/{len(g)}",
                                            p05=f"{int((g.p < 0.05).sum())}/{len(g)}",
                                            mean_rho_start=round(g.rho_start.mean(), 3) if "rho_start" in g and g.rho_start.notna().any() else None)
                                    for a, g in Rr.groupby("arm")}
    # spread recovery per split: learned from the shared start; the anchor's moments alone; learned from them
    cols = [("per-clade", "rho", "learned,\nshared start", "#184f95"), ("per-clade-mom", "rho_start", "anchor\nalone", GREY),
            ("per-clade-mom", "rho", "learned,\nanchor start", BLUE)]
    fig, ax = plt.subplots(figsize=(4.9, 4.6))
    for i, (arm, key, lab, col) in enumerate(cols):
        v = Rr[Rr.arm == arm][key].dropna().values
        ax.scatter(i + np.random.default_rng(1).uniform(-0.17, 0.17, len(v)), v, s=55, color=col, alpha=0.75, lw=0)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=col if col != GREY else INK2, lw=5, solid_capstyle="round")
        ax.text(i, 1.06, f"{v.mean():+.2f}", ha="center", va="bottom", color=INK, fontsize=17)
    ax.axhline(0, color=INK2, lw=1, ls=":")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[2] for c in cols], fontsize=15)
    ax.set_ylim(-0.75, 1.25); ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.set_ylabel("Spearman with true spread")
    save(fig, "s4_perclade")
    et = [r for r in json.load(open(HERE / "spotr_out" / "etracer.json")) if r["arm"] in ("expression", "fgw")]
    for f in ["etracer2.json", "etracer3.json", "etracer_s37.json"]:   # SpOTr arms rerun with the across-clone spread
        if (HERE / "spotr_out" / f).exists():                     # fixed (2), anchor-moment start (3), anchor draws 3-7
            et += json.load(open(HERE / "spotr_out" / f))
    for f in ["tangram.json", "tangram_et37.json"]:
        if (HERE / "spotr_out" / f).exists():
            et += [r for r in json.load(open(HERE / "spotr_out" / f)) if r["data"] == "eTRACER"]
    E = pd.DataFrame([dict(tag=r["tag"], seed=r["seed"], arm=r["arm"], dens=r["dens"]) for r in et])
    E.to_csv(OUT / "etracer_summary.csv", index=False)
    order_e = [("expression", "expr.", GREY), ("tangram", "Tangram", PURPLE), ("fgw", "FGW", ORANGE), ("law-path", "shared L", AQUA),
               ("per-clone", "per-clone L", "#184f95")]
    order_e = [o for o in order_e if (E.arm == o[0]).any()]
    fig, ax = plt.subplots(figsize=(4.9, 4.6)); axes = [ax]
    for j, (tag, name, mk) in enumerate([("LE", "late escape", "o"), ("EE", "early escape", "s")]):
        means = []
        for i, (arm, lab, col) in enumerate(order_e):
            v = E[(E.tag == tag) & (E.arm == arm)].dens.values
            ax.scatter(np.full(len(v), i) + (j - 0.5) * 0.3, v, s=55, color=col, alpha=0.8, lw=0, marker=mk)
            means.append(v.mean())
        ax.plot(np.arange(len(order_e)) + (j - 0.5) * 0.3, means, color=INK2, lw=1.4, ls="-" if tag == "LE" else "--",
                label=name)
    ax.set_xticks(range(len(order_e))); ax.set_xticklabels([o[1] for o in order_e], fontsize=14, rotation=30, ha="right")
    ax.legend(loc="lower center", fontsize=14, bbox_to_anchor=(0.6, 0.0))
    axes[0].set_ylabel("clone-density agreement (r)")
    save(fig, "s4_etracer")
    NUM["etracer"] = {t: g.groupby("arm").dens.mean().round(3).to_dict() for t, g in E.groupby("tag")}


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "real":
        real()
    else:
        method_and_simulation()
    prev = json.load(open(OUT / "numbers.json")) if (OUT / "numbers.json").exists() else {}
    prev.update({k: v for k, v in NUM.items()})
    json.dump(prev, open(OUT / "numbers.json", "w"), indent=1, default=str)
    print(json.dumps(NUM, indent=1, default=str))
