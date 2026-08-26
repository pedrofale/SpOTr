"""Why none of the BCSA datasets can demonstrate SpOTr: four preconditions, all failed."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "dataset_requirements.csv")
lab = [d.replace("BCSA", "") for d in df["dataset"]]
x = np.arange(len(df))

fig, axs = plt.subplots(1, 4, figsize=(18, 4.4))
specs = [
    ("coherence", "spatial coherence of clone labels", 0.30, "need >0.3", "0 = intermixed, 1 = clean territories", False),
    ("r_mantel", "tree vs territory agreement (Mantel r)", None, None, "SpOTr's premise; none significant", False),
    ("comp_TVD", "SS3 vs Visium composition mismatch", 0.20, "need <0.2", "total variation distance", True),
    ("visium_frac_unreferenced", "spots whose clone has <10 SS3 cells", 0.10, "need <0.1", "fraction of the section", True),
]
for ax, (col, title, thr, thrlab, sub, lower_better) in zip(axs, specs):
    v = df[col].to_numpy()
    if col == "r_mantel":
        colors = ["silver"] * len(v)          # none significant
    elif lower_better:
        colors = ["seagreen" if a < thr else "indianred" for a in v]
    else:
        colors = ["seagreen" if a > thr else "indianred" for a in v]
    ax.bar(x, v, color=colors, edgecolor="k", linewidth=0.5)
    if thr is not None:
        ax.axhline(thr, ls="--", c="k", lw=1.2)
        ax.text(len(v) - 0.4, thr, f" {thrlab}", va="bottom", ha="right", fontsize=8)
    if col == "r_mantel":
        for i, (r, p) in enumerate(zip(v, df["p_mantel"])):
            ax.text(i, r + (0.02 if r >= 0 else -0.05), f"p={p:.2f}", ha="center", fontsize=7)
        ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(sub, fontsize=8, color="dimgray")

fig.suptitle("None of the BCSA datasets meets the preconditions for demonstrating SpOTr "
             "(red = fails)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.9])
out = HERE / "bcsa_dataset_requirements.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
