import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
FG, BG, MID = "#1b1b1f", "#ffffff", "#9a9aa4"
ACC, OK, COOL, WARN = "#b03a2e", "#2e7d5b", "#31688e", "#c8891a"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MID, "axes.labelcolor": FG,
                     "text.color": FG, "xtick.color": MID, "ytick.color": MID,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "font.size": 9})
fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))

# 1. which alpha does each objective pick
s = pd.read_csv(HERE / "q15_cnv_sweep.csv")
per = []
for sd, sub in s.groupby("seed"):
    q = sub.set_index("alpha")
    per.append((q.loss_expr.idxmin(), q.loss_cnv.idxmin(), q.acc.idxmax()))
p = np.array(per)
a = ax[0]
rng = np.random.default_rng(0)
for i, (col, nm) in enumerate([(0, "expression\nloss"), (1, "CNV\nloss"), (2, "true\naccuracy")]):
    a.scatter(rng.normal(i, .05, len(p)), p[:, col], s=52,
              color=[MID, COOL, ACC][i], alpha=.8, zorder=3)
    a.hlines(np.mean(p[:, col]), i - .22, i + .22, color=[MID, COOL, ACC][i], lw=2.4, zorder=4)
a.set_xticks([0, 1, 2]); a.set_xticklabels(["expression\nloss", "CNV\nloss", "true\naccuracy"])
a.set_ylabel("selected α"); a.set_ylim(-0.03, 0.42)
a.set_title("1. The CNV anchor unpins\nthe objective from α=0", fontsize=10, loc="left")
a.text(.03, .93, "expression loss: α=0 in 10/10", transform=a.transAxes, fontsize=8, color=MID)

# 2. init sweep
q = pd.read_csv(HERE / "q17_init_sweep.csv"); q = q[q.iters == 300]
g = q.groupby("init").auc
a = ax[1]
a.errorbar(g.mean().index, g.mean(), yerr=g.std(), color=ACC, lw=2, marker="o", ms=5,
           capsize=3, ecolor=MID)
a.axhline(0.5, color=FG, ls="--", lw=1)
a.axvspan(0.0, 0.42, color=OK, alpha=.10)
a.text(0.20, 0.545, "usable range\nAUC 0.96–0.98\n10/10 seeds", fontsize=8, color=OK, ha="center")
a.set_xlabel("α initialisation"); a.set_ylabel("AUC(learned α → compact)")
a.set_ylim(0.3, 1.04)
a.set_title("2. A plateau, not a cliff —\ninit-independent below 0.5", fontsize=10, loc="left")

# 3. learned alpha separation
a = ax[2]
w = 0.36; x = np.arange(len(g.mean().index))
qq = q.groupby("init")[["a_c", "a_d"]].mean()
a.bar(x - w/2, qq.a_c, w, color=OK, label="compact clades")
a.bar(x + w/2, qq.a_d, w, color=COOL, label="dispersed clades")
a.set_xticks(x); a.set_xticklabels([f"{v:.2f}" for v in qq.index], fontsize=8)
a.set_xlabel("α initialisation"); a.set_ylabel("learned α")
a.set_title("3. Learned α separates the\nclade types 6–10× when init is low", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8)

# 4. anchor robustness
h = pd.read_csv(HERE / "q16_anchor_quality.csv")
piv = h.pivot_table(index="noise", columns="bins", values="pos", aggfunc="mean") * 100
a = ax[3]
im = a.imshow(piv.values, cmap="BuGn", vmin=0, vmax=70, aspect="auto")
a.set_xticks(range(len(piv.columns))); a.set_xticklabels(piv.columns)
a.set_yticks(range(len(piv.index))); a.set_yticklabels([f"×{v:.1f}" for v in piv.index])
a.set_xlabel("CNV segments retained"); a.set_ylabel("extra noise")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        a.text(j, i, f"{piv.values[i, j]:.0f}%", ha="center", va="center", fontsize=8.5, color=FG)
a.set_title("4. Robust to a degraded anchor\n(% of seeds selecting α>0)", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(HERE / "cnv_summary.png", dpi=170, bbox_inches="tight", facecolor=BG)
print("wrote cnv_summary.png")
