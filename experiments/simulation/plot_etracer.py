import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
E = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
FG, BG, MID = "#1b1b1f", "#ffffff", "#9a9aa4"
ACC, OK, COOL, WARN = "#b03a2e", "#2e7d5b", "#31688e", "#c8891a"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MID, "axes.labelcolor": FG,
                     "text.color": FG, "xtick.color": MID, "ytick.color": MID,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))

le = pd.read_csv(E / "le_semirelaxed.csv"); ee = pd.read_csv(E / "ee_semirelaxed.csv")
cols = {"balanced": MID, "1.0": COOL, "0.1": WARN, "0.01": ACC}
for a, df, nm, sub in [(ax[0], le, "Late escape", "4,120 cells · TVD 0.053"),
                       (ax[1], ee, "Early escape", "735 cells · TVD 0.374")]:
    for t, s in df.groupby("tau", sort=False):
        s = s.sort_values("alpha")
        a.plot(s.alpha, s.acc, "o-", ms=4, lw=1.9, color=cols.get(str(t), MID),
               label=f"τ = {t}")
    a.set_xlabel("α  (tree weight)"); a.set_ylabel("clone accuracy on held-out bins")
    a.set_title(f"{nm}\n{sub}", fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8)
ax[0].set_ylim(0.36, 0.50); ax[1].set_ylim(0.17, 0.31)

a = ax[2]
g = [("LE\nbalanced", 0.3917, 0.4410, MID), ("LE\nτ=0.01", 0.3772, 0.4850, ACC),
     ("EE\nbalanced", 0.2286, 0.2871, MID), ("EE\nτ=0.1", 0.2286, 0.2925, ACC)]
x = np.arange(len(g)); w = 0.36
a.bar(x - w/2, [v[1] for v in g], w, color="#d5d5dc", label="no tree (α=0)")
a.bar(x + w/2, [v[2] for v in g], w, color=[v[3] for v in g], label="best α")
for i, v in enumerate(g):
    a.text(i + w/2, v[2] + 0.008, f"+{v[2]-v[1]:.3f}", ha="center", fontsize=8.5, color=v[3])
a.set_xticks(x); a.set_xticklabels([v[0] for v in g], fontsize=8.5)
a.set_ylabel("clone accuracy"); a.set_ylim(0, 0.56)
a.set_title("The tree helps at both stages,\nat every τ", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "etracer_semirelaxed.png", dpi=170,
            bbox_inches="tight", facecolor=BG)
print("wrote etracer_semirelaxed.png")
