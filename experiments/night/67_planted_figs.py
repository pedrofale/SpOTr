"""Figure for 66_planted_control.py: anchor loss and placement error over (L_A, L_B), one row per planted
dispersal, group draw 0.  Cross = mixing lengths measured from the moved true positions; circle = minimum of
the panel's surface.  Usage: python 67_planted_figs.py <dir with planted_grid.csv>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

D = Path(sys.argv[1]); df = pd.read_csv(D / "planted_grid.csv")
for cov in (20, 5):
    df[f"nll_{cov}"] = df[[c for c in df.columns if c.startswith(f"nll_{cov}_")]].mean(1)
grid = sorted(df.LA.unique()); idx = {v: i for i, v in enumerate(grid)}
SEQ_L = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
THEMES = {"light": dict(surf="#fcfcfb", ink="#0b0b0b", ink2="#52514e", mark="#eb6834", seq=SEQ_L),
          "dark": dict(surf="#1a1a19", ink="#f0efec", ink2="#c3c2b7", mark="#d95926", seq=SEQ_L)}

summary = []
for (s, g), h in df.groupby(["sigma", "gdraw"]):
    row = dict(sigma=s, gdraw=g, measA=h.measA.iloc[0], measB=h.measB.iloc[0],
               err_shared=float(h[(h.LA == 305) & (h.LB == 305)].err.iloc[0]))
    b = h.loc[h.err.idxmin()]; row.update(best_err=b.err, best_LA=b.LA, best_LB=b.LB)
    for cov in (20, 5):
        a = h.loc[h[f"nll_{cov}"].idxmin()]
        picks = [h.loc[h[f"nll_{cov}_{r}"].idxmin()] for r in range(5)]
        row.update({f"anchor{cov}_LA": a.LA, f"anchor{cov}_LB": a.LB, f"anchor{cov}_err": a.err,
                    f"anchor{cov}_draws_LB_gt_LA": int(sum(p.LB > p.LA for p in picks)),
                    f"anchor{cov}_draws_err": float(np.mean([p.err for p in picks]))})
    summary.append(row)
S = pd.DataFrame(summary); S.to_csv(D / "planted_summary.csv", index=False)
print(S.round(1).to_string(index=False))

for th, t in THEMES.items():
    plt.rcParams.update({"figure.facecolor": t["surf"], "axes.facecolor": t["surf"], "savefig.facecolor": t["surf"],
                         "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
                         "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.titlecolor": t["ink"]})
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    sig = sorted(df.sigma.unique())
    fig, axes = plt.subplots(len(sig), 2, figsize=(9.4, 3.2 * len(sig)))
    for r, s in enumerate(sig):
        h = df[(df.sigma == s) & (df.gdraw == 0)]
        for c, (col, lab) in enumerate([("nll_20", "Anchor loss, 20% of spots: what the method sees"),
                                         ("err", "Placement error (µm): the truth, not seen")]):
            ax = axes[r, c]; Z = np.full((len(grid), len(grid)), np.nan)
            for _, q in h.iterrows():
                Z[idx[q.LB], idx[q.LA]] = q[col]
            im = ax.imshow(Z, origin="lower", cmap=cmap.reversed() if th == "light" else cmap, aspect="auto")
            ax.set_xticks(range(len(grid))); ax.set_xticklabels([f"{v:.0f}" for v in grid], fontsize=8.5)
            ax.set_yticks(range(len(grid))); ax.set_yticklabels([f"{v:.0f}" for v in grid], fontsize=8.5)
            ax.set_xlabel("L_A, undisturbed clades (µm)" if r == len(sig) - 1 else "")
            ax.set_ylabel(f"σ = {s:.0f} µm\nL_B, dispersed clades (µm)" if c == 0 else "")
            mA, mB = h.measA.iloc[0], h.measB.iloc[0]
            fx = np.interp(mA, grid, range(len(grid))); fy = np.interp(mB, grid, range(len(grid)))
            ax.plot(fx, fy, "x", color=t["mark"], ms=11, mew=2.5)
            iy, ix = np.unravel_index(np.nanargmin(Z), Z.shape)
            ax.plot(ix, iy, "o", ms=13, mfc="none", mec=t["ink"], mew=2)
            ax.plot([0, len(grid) - 1], [0, len(grid) - 1], ":", color=t["ink2"], lw=1)
            if r == 0:
                ax.set_title(lab, fontsize=10.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=8, colors=t["ink2"]); cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(D / f"planted_{th}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
