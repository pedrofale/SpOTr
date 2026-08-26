import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
FG, BG, MID = "#1b1b1f", "#ffffff", "#9a9aa4"
ACC, OK, WARN, COOL = "#b03a2e", "#2e7d5b", "#c8891a", "#31688e"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MID, "axes.labelcolor": FG,
                     "text.color": FG, "xtick.color": MID, "ytick.color": MID,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "font.size": 9})
fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))

# 1. variance control
u = pd.read_csv(HERE / "replicate_confusable.csv"); u["g"] = u.pair_o - u.pair_g
c = pd.read_csv(HERE / "replicate_controlled.csv"); c["g"] = c.ao - c.ag
a = ax[0]
for i, (df, nm, col) in enumerate([(u, "uncontrolled", MID), (c, "controlled", ACC)]):
    x = np.random.default_rng(0).normal(i, 0.06, len(df))
    a.scatter(x, df.g, s=46, color=col, zorder=3, alpha=.85)
    a.hlines(df.g.mean(), i - .22, i + .22, color=col, lw=2.4, zorder=4)
a.axhline(0, color=FG, lw=.9)
a.set_xticks([0, 1]); a.set_xticklabels(["uncontrolled\n4/8 seeds", "controlled\n11/11 seeds"])
a.set_ylabel("oracle per-clade  −  best global α")
a.set_title("1. Controlling the simulator\nturned a coin-flip into 11/11", fontsize=10, loc="left")

# 2. mechanism
m = pd.read_csv(HERE / "q12_mechanism.csv").groupby("alpha")[["all", "compact", "dispersed"]].mean()
a = ax[1]
for col, nm, cl in [("compact", "compact clades", OK), ("dispersed", "dispersed clades", COOL),
                    ("all", "all", ACC)]:
    a.plot(m.index, m[col], "o-", color=cl, lw=2, ms=4.5, label=nm)
for v, cl, lab in [(0.350, OK, None), (0.182, ACC, None), (0.095, COOL, None)]:
    a.axvline(v, color=cl, ls=":", lw=1.3)
a.set_xlabel("global α"); a.set_ylabel("clade accuracy")
a.set_title("2. Compact wants α=0.35,\ndispersed α=0.10 — global compromises", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8)

# 3. oracle vs controls
a = ax[2]
names = ["no tree\n(α=0)", "best\nglobal α", "ORACLE\nper-clade", "shuffled", "inverted"]
vals = [c.a0.mean(), c.ag.mean(), c.ao.mean(), c.shuf.mean(), c.inv.mean()]
sds = [c.a0.std(), c.ag.std(), c.ao.std(), c.shuf.std(), c.inv.std()]
a.bar(names, vals, yerr=sds, color=[MID, COOL, ACC, "#c9c9d1", "#e2e2e8"],
      error_kw=dict(lw=1, capsize=3, ecolor=FG))
a.set_ylabel("clade accuracy"); a.set_ylim(0, 0.42)
a.set_title("3. Oracle > global > shuffled >\ninverted, in 11/11 seeds", fontsize=10, loc="left")

# 4. learnability gap
q = pd.read_csv(HERE / "q13_heldout.csv")
a = ax[3]
g = q.groupby("alpha")
a2 = a.twinx(); a2.spines["top"].set_visible(False)
for col, cl, nm in [("loss_fit", MID, "training loss"), ("loss_ev", COOL, "held-out loss")]:
    z = g[col].mean(); a.plot(z.index, (z - z.min()) / (z.max() - z.min()), "o-",
                              color=cl, lw=2, ms=4, label=nm)
z = g.acc.mean()
a2.plot(z.index, z, "o-", color=ACC, lw=2.4, ms=5, label="true accuracy")
a2.set_ylabel("clade accuracy", color=ACC); a2.tick_params(axis="y", colors=ACC)
a.set_xlabel("global α"); a.set_ylabel("loss (rescaled)")
a.set_title("4. Both losses are monotone —\nargmin pinned at α=0, 11/11", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.0))
a2.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 0.06))
a.text(.97, .40, "the two loss curves\ncoincide", transform=a.transAxes, fontsize=7.5,
       color=MID, ha="right")
a2.axvline(0.2, color=ACC, ls=":", lw=1.2)
fig.tight_layout()
fig.savefig(HERE / "controlled_summary.png", dpi=170, bbox_inches="tight", facecolor=BG)
print("wrote controlled_summary.png")
