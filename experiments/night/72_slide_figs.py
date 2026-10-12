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
    for f in ["real.json", "real_more.json", "real_controls.json"]:
        if (HERE / "spotr_out" / f).exists():
            res += json.load(open(HERE / "spotr_out" / f))
    rows = []
    for r in res:
        if r["arm"].startswith("path-"):
            P = pd.DataFrame(r["path"])
            pick = P.loc[P.nll_val.idxmin()]
            rows.append(dict(sec=r["sec"], seed=r["seed"], arm=r["arm"], err=pick.err, agree=pick.agree, w=pick.w))
            if r["arm"] == "path-fgw":
                e0 = P[P.w == 0].iloc[0]
                rows.append(dict(sec=r["sec"], seed=r["seed"], arm="expression", err=e0.err, agree=e0.agree, w=0.0))
        else:
            rows.append(dict(sec=r["sec"], seed=r["seed"], arm=r["arm"], err=r["err"], agree=r["agree"], w=np.nan))
    D = pd.DataFrame(rows); D.to_csv(OUT / "real_summary.csv", index=False)
    order = [("expression", "Expression only", GREY), ("path-fgw", "FGW, square loss", ORANGE),
             ("path-law", "SpOTr, shared L", AQUA), ("weight", "SpOTr, learned α", BLUE),
             ("per-clade", "SpOTr, per-clade L", "#184f95"),
             ("per-random", "control: random groups", GREY), ("path-law-shuffled", "control: shuffled tree", GREY)]
    order = [o for o in order if (D.arm == o[0]).any()]
    plt.rcParams.update({"font.size": 21, "axes.labelsize": 21, "xtick.labelsize": 20, "ytick.labelsize": 21})
    fig, ax = plt.subplots(figsize=(5.4, 6.6))
    y = np.arange(len(order))[::-1]
    for yi, (arm, lab, col) in zip(y, order):
        v = D[D.arm == arm].err.values
        mk = "o" if not lab.startswith("control") else "D"
        ax.scatter(v, yi + np.random.default_rng(0).uniform(-0.14, 0.14, len(v)), s=70, color=col, alpha=0.7, lw=0, marker=mk)
        ax.plot([v.mean()] * 2, [yi - 0.33, yi + 0.33], color=col if not lab.startswith("control") else INK2, lw=5, solid_capstyle="round")
        ax.text(v.mean(), yi + 0.36, f"{v.mean():.0f}", ha="center", color=INK, fontsize=19, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([o[1] for o in order])
    nspl = D[D.arm == "per-clade"].shape[0]
    ax.set_xlabel(f"Placement error (µm), {nspl} splits")
    ax.set_xticks([400, 500, 600, 700, 800])
    save(fig, "s4_petracer")
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 16})
    NUM["petracer"] = {"err": D.groupby("arm").err.mean().round(1).to_dict(), "agree": D.groupby("arm").agree.mean().round(3).to_dict(),
                       "n_splits": int(nspl)}
    pc = D[D.arm == "per-clade"].set_index(["sec", "seed"]).err; fg = D[D.arm == "path-fgw"].set_index(["sec", "seed"]).err
    lw_ = D[D.arm == "path-law"].set_index(["sec", "seed"]).err
    NUM["petracer"]["perclade_beats_fgw"] = int((pc < fg.reindex(pc.index)).sum())
    NUM["petracer"]["perclade_beats_shared"] = int((pc < lw_.reindex(pc.index)).sum())
    # per-clade learned vs measured, centred within each split
    rr, pts = [], []
    for r in res:
        if r["arm"] != "per-clade":
            continue
        Lk = np.array(r["L"]); ok = np.array(r["sizes"]) >= 30
        meas = np.array([r["meas"].get(str(k), np.nan) for k in range(len(Lk))]); ok &= np.isfinite(meas)
        rho = spearmanr(Lk[ok], meas[ok])[0]
        null = np.array([spearmanr(np.random.default_rng(i).permutation(Lk[ok]), meas[ok])[0] for i in range(2000)])
        rr.append(dict(sec=r["sec"], seed=r["seed"], rho=float(rho), p=float(np.mean(null >= rho)), n=int(ok.sum())))
        lx, ly = np.log(meas[ok]), np.log(Lk[ok])
        for a_, b_ in zip(lx - lx.mean(), ly - ly.mean()):
            pts.append(dict(sec=r["sec"], x=a_, y=b_))
    NUM["real_perclade"] = rr
    Pp = pd.DataFrame(pts)
    fig, ax = plt.subplots(figsize=(4.9, 4.6))
    for sec, col in [("M3-4", BLUE), ("M3-1", ORANGE), ("M3-2", AQUA)]:
        d = Pp[Pp.sec == sec]; ax.scatter(np.exp(d.x), np.exp(d.y), s=60, color=col, alpha=0.75, lw=0, label=sec)
    ax.set_xscale("log"); ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×")); a.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.7, 1, 1.4]); ax.set_yticks([0.5, 0.7, 1, 1.4, 2])
    ax.set_xlabel("measured spread (relative)"); ax.set_ylabel("learned spread (relative)")
    ax.legend(loc="upper left", fontsize=14, handletextpad=0.2)
    save(fig, "s4_perclade")
    et = [r for r in json.load(open(HERE / "spotr_out" / "etracer.json")) if r["arm"] in ("expression", "fgw")]
    if (HERE / "spotr_out" / "etracer2.json").exists():          # SpOTr arms rerun with the across-clone spread fixed
        et += json.load(open(HERE / "spotr_out" / "etracer2.json"))
    E = pd.DataFrame([dict(tag=r["tag"], seed=r["seed"], arm=r["arm"], dens=r["dens"]) for r in et])
    E.to_csv(OUT / "etracer_summary.csv", index=False)
    order_e = [("expression", "expr.", GREY), ("fgw", "FGW", ORANGE), ("law-path", "shared", AQUA),
               ("weight", "learned", BLUE), ("per-clone", "per-clone", "#184f95")]
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
