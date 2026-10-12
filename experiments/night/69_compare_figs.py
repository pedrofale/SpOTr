"""Law formulation (66_) vs the paper's clade-aware FGW (68_) on the planted control, same anchor loss.
Direction recovered = the dispersed group B gets the 'less tree-bound' parameter: law L_B > L_A; paper alpha_B < alpha_A;
paper theta_B < theta_A.  Placement = error at the anchor's pick, per-group vs restricted to the shared diagonal.
Usage: python 69_compare_figs.py <66_ out dir> <68_ out dir> <fig out dir>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

law = pd.read_csv(Path(sys.argv[1]) / "planted_grid.csv").rename(columns={"LA": "A", "LB": "B"})
pap = pd.read_csv(Path(sys.argv[2]) / "paper_grid.csv"); OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
arms = {"Law: mixing length L": (law, "B>A"), "Paper: tree weight α": (pap[pap.phase == 1], "B<A"),
        "Paper: within-clade weight θ": (pap[pap.phase == 2], "B<A")}
rows = []
for arm, (d, sense) in arms.items():
    for (s, g), h in d.groupby(["sigma", "gdraw"]):
        diag = h[h.A == h.B]
        for cov in (20, 5):
            for r in range(5):
                c = f"nll_{cov}_{r}"
                pk = h.loc[h[c].idxmin()]; pd_ = diag.loc[diag[c].idxmin()]
                rec = (pk.B > pk.A) if sense == "B>A" else (pk.B < pk.A)
                rows.append(dict(arm=arm, sigma=s, gdraw=g, cov=cov, draw=r, recovered=bool(rec), tie=bool(pk.A == pk.B),
                                 err_pick=pk.err, err_shared_pick=pd_.err, err_best=h.err.min(), err_best_shared=diag.err.min()))
R = pd.DataFrame(rows); R.to_csv(OUT / "compare.csv", index=False)
S = R.groupby(["arm", "sigma"]).agg(recovered=("recovered", "sum"), ties=("tie", "sum"), n=("recovered", "size"),
                                    err_pick=("err_pick", "mean"), err_shared_pick=("err_shared_pick", "mean"),
                                    err_best=("err_best", "mean"), err_best_shared=("err_best_shared", "mean")).reset_index()
print(S.round(1).to_string(index=False))
json.dump(S.to_dict(orient="records"), open(OUT / "compare_summary.json", "w"), indent=1)

THEMES = {"light": dict(surf="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", s=["#2a78d6", "#eb6834", "#1baf7a"]),
          "dark": dict(surf="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34342f", s=["#3987e5", "#d95926", "#199e70"])}
for th, t in THEMES.items():
    plt.rcParams.update({"figure.facecolor": t["surf"], "axes.facecolor": t["surf"], "savefig.facecolor": t["surf"],
                         "axes.edgecolor": t["grid"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
                         "text.color": t["ink"], "axes.grid": True, "grid.color": t["grid"], "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": t["ink"],
                         "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for i, arm in enumerate(arms):
        h = S[S.arm == arm].sort_values("sigma")
        axes[0].plot(h.sigma, 100 * h.recovered / h.n, "-o", color=t["s"][i], ms=7, label=arm)
        axes[1].plot(h.sigma, h.err_pick, "-o", color=t["s"][i], ms=7, label=f"{arm}, per group")
        axes[1].plot(h.sigma, h.err_shared_pick, "--", color=t["s"][i], lw=1.4, label=f"{arm}, shared")
    axes[0].axhline(50, color=t["ink2"], ls=":", lw=1); axes[0].text(5, 52, "chance", color=t["ink2"], fontsize=9)
    axes[0].set_ylim(0, 105); axes[0].set_xticks([0, 300, 500]); axes[1].set_xticks([0, 300, 500])
    axes[0].set_xlabel("Planted dispersal σ (µm)"); axes[1].set_xlabel("Planted dispersal σ (µm)")
    axes[0].set_ylabel("Anchor picks the right direction (%)"); axes[1].set_ylabel("Placement error at the anchor's pick (µm)")
    axes[0].set_title("Does the anchor find the dispersed group?"); axes[1].set_title("What it costs in placement (solid: per group, dashed: shared)")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / f"compare_{th}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
